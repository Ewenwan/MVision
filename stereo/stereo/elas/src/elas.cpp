/*
算法基本思想为：通过计算一些支持点组成稀疏视差图，
对这些支持点在图像坐标空间进行三角剖分，构建视差的先验值。

由于支持点可被精确匹配，避免了使用其余点进行匹配造成的匹配模糊。
进而可以通过有效利用视差搜索空间，重建精确的稠密视差图，而不必进行全局优化。
算法分为以下几个部分：
1.匹配支持点
	-首先确定支持点匹配的特征描述算子，文中采用简单的9X9尺寸的sobel滤波并连结周围像素窗口的sobel值组成特征。
	特征算子维度为1+11+5=17，作者有提到使用更复杂的surf特征对提高匹配的精度并无益处，反而使得速度更慢。
	匹配方法为L1向量距离，并进行从左到右及从右到左两次匹配。
	为防止多个匹配点歧义，剔除最大匹配点与次匹配点匹配得分比超过一定阀值的点。
	另外则是增加图像角点作为支持点，角点视差取其最近邻点的值。

2.立体匹配生成模型
	这里所谓的生成模型，简单来讲就是基于上面确定的支持点集，
	也可以扩展一些角点，再对这些支持点集进行三角剖分，形成多个三角形区域。
	在每个三角形内基于三个已知顶点的 精确 视差值进行MAP最大后验估计插值该三角区域内的其他点视差。

3.视差估计
	视差估计依赖最大后验估计（MAP）来计算其余观察点的视差值。
4.提纯
	后面主要是对E(d)进行条件约束

*/

#include "elas.h"
#include <math.h>
#include "descriptor.h"
#include "triangle.h"
#include "matrix.h"

using namespace std;

// 匹配函数 matching function
// 输入: 指针  left (I1) and right (I2) intensity image (uint8, input)  左右输入图像
//       指针  left (D1) and right (D2) disparity image (float, output) 左右输出视差图
//       数组  dims[0] = width of I1 and I2  宽度
//             dims[1] = height of I1 and I2 高度
//             dims[2] = 存储设置 每一行的byte  bpl    = width + 15-(width-1)%16;
void Elas::process (uint8_t* I1_,uint8_t* I2_,float* D1,float* D2,const int32_t* dims){

  // get width, height and bytes per line
  width  = dims[0];
  height = dims[1];
  bpl    = width + 15-(width-1)%16;

  // copy images to byte aligned memory
  I1 = (uint8_t*)_mm_malloc(bpl*height*sizeof(uint8_t),16);
  I2 = (uint8_t*)_mm_malloc(bpl*height*sizeof(uint8_t),16);
  memset (I1,0,bpl*height*sizeof(uint8_t));
  memset (I2,0,bpl*height*sizeof(uint8_t));
  if (bpl==dims[2]) {
    memcpy(I1,I1_,bpl*height*sizeof(uint8_t));
    memcpy(I2,I2_,bpl*height*sizeof(uint8_t));
  } else {
    for (int32_t v=0; v<height; v++) {
      memcpy(I1+v*bpl,I1_+v*dims[2],width*sizeof(uint8_t));
      memcpy(I2+v*bpl,I2_+v*dims[2],width*sizeof(uint8_t));
    }
  }

#ifdef PROFILE
  timer.start("Descriptor");
#endif
//=======1=========
  Descriptor desc1(I1,width,height,bpl,param.subsampling);
  Descriptor desc2(I2,width,height,bpl,param.subsampling);

#ifdef PROFILE
  timer.start("Support Matches");
#endif
//========2=========
  vector<support_pt> p_support = computeSupportMatches(desc1.I_desc,desc2.I_desc);

  // if not enough support points for triangulation
  if (p_support.size()<3) {
    cout << "ERROR: Need at least 3 support points!" << endl;
    _mm_free(I1);
    _mm_free(I2);
    return;
  }

#ifdef PROFILE
  timer.start("Delaunay Triangulation");
#endif
  vector<triangle> tri_1 = computeDelaunayTriangulation(p_support,0);
  vector<triangle> tri_2 = computeDelaunayTriangulation(p_support,1);

#ifdef PROFILE
  timer.start("Disparity Planes");
#endif
  computeDisparityPlanes(p_support,tri_1,0);
  computeDisparityPlanes(p_support,tri_2,1);

#ifdef PROFILE
  timer.start("Grid");
#endif

  // allocate memory for disparity grid
  int32_t grid_width   = (int32_t)ceil((float)width/(float)param.grid_size);
  int32_t grid_height  = (int32_t)ceil((float)height/(float)param.grid_size);
  int32_t grid_dims[3] = {param.disp_max+2,grid_width,grid_height};
  int32_t* disparity_grid_1 = (int32_t*)calloc((param.disp_max+2)*grid_height*grid_width,sizeof(int32_t));
  int32_t* disparity_grid_2 = (int32_t*)calloc((param.disp_max+2)*grid_height*grid_width,sizeof(int32_t));

  createGrid(p_support,disparity_grid_1,grid_dims,0);
  createGrid(p_support,disparity_grid_2,grid_dims,1);

#ifdef PROFILE
  timer.start("Matching");
#endif
  computeDisparity(p_support,tri_1,disparity_grid_1,grid_dims,desc1.I_desc,desc2.I_desc,0,D1);
  computeDisparity(p_support,tri_2,disparity_grid_2,grid_dims,desc1.I_desc,desc2.I_desc,1,D2);

#ifdef PROFILE
  timer.start("L/R Consistency Check");
#endif
  leftRightConsistencyCheck(D1,D2);

#ifdef PROFILE
  timer.start("Remove Small Segments");
#endif
  removeSmallSegments(D1);
  if (!param.postprocess_only_left)
    removeSmallSegments(D2);

#ifdef PROFILE
  timer.start("Gap Interpolation");
#endif
  gapInterpolation(D1);
  if (!param.postprocess_only_left)
    gapInterpolation(D2);

  if (param.filter_adaptive_mean) {
#ifdef PROFILE
    timer.start("Adaptive Mean");
#endif
    adaptiveMean(D1);
    if (!param.postprocess_only_left)
      adaptiveMean(D2);
  }

  if (param.filter_median) {
#ifdef PROFILE
    timer.start("Median");
#endif
    median(D1);
    if (!param.postprocess_only_left)
      median(D2);
  }

#ifdef PROFILE
  timer.plot();
#endif

  // release memory
  free(disparity_grid_1);
  free(disparity_grid_2);
  _mm_free(I1);
  _mm_free(I2);
}

// 支持点 support point functions
void Elas::removeInconsistentSupportPoints (int16_t* D_can,int32_t D_can_width,int32_t D_can_height) {

  // for all valid support points do
  for (int32_t u_can=0; u_can<D_can_width; u_can++) {
    for (int32_t v_can=0; v_can<D_can_height; v_can++) {
      int16_t d_can = *(D_can+getAddressOffsetImage(u_can,v_can,D_can_width));
      if (d_can>=0) {

        // compute number of other points supporting the current point
        int32_t support = 0;
        for (int32_t u_can_2=u_can-param.incon_window_size; u_can_2<=u_can+param.incon_window_size; u_can_2++) {
          for (int32_t v_can_2=v_can-param.incon_window_size; v_can_2<=v_can+param.incon_window_size; v_can_2++) {
            if (u_can_2>=0 && v_can_2>=0 && u_can_2<D_can_width && v_can_2<D_can_height) {
              int16_t d_can_2 = *(D_can+getAddressOffsetImage(u_can_2,v_can_2,D_can_width));
              if (d_can_2>=0 && abs(d_can-d_can_2)<=param.incon_threshold)
                support++;
            }
          }
        }

        // invalidate support point if number of supporting points is too low
        if (support<param.incon_min_support)
          *(D_can+getAddressOffsetImage(u_can,v_can,D_can_width)) = -1;
      }
    }
  }
}

void Elas::removeRedundantSupportPoints(int16_t* D_can,int32_t D_can_width,int32_t D_can_height,
                                        int32_t redun_max_dist, int32_t redun_threshold, bool vertical) {

  // parameters
  int32_t redun_dir_u[2] = {0,0};
  int32_t redun_dir_v[2] = {0,0};
  if (vertical) {
    redun_dir_v[0] = -1;
    redun_dir_v[1] = +1;
  } else {
    redun_dir_u[0] = -1;
    redun_dir_u[1] = +1;
  }

  // for all valid support points do
  for (int32_t u_can=0; u_can<D_can_width; u_can++) {
    for (int32_t v_can=0; v_can<D_can_height; v_can++) {
      int16_t d_can = *(D_can+getAddressOffsetImage(u_can,v_can,D_can_width));
      if (d_can>=0) {

        // check all directions for redundancy
        bool redundant = true;
        for (int32_t i=0; i<2; i++) {

          // search for support
          int32_t u_can_2 = u_can;
          int32_t v_can_2 = v_can;
          int16_t d_can_2;
          bool support = false;
          for (int32_t j=0; j<redun_max_dist; j++) {
            u_can_2 += redun_dir_u[i];
            v_can_2 += redun_dir_v[i];
            if (u_can_2<0 || v_can_2<0 || u_can_2>=D_can_width || v_can_2>=D_can_height)
              break;
            d_can_2 = *(D_can+getAddressOffsetImage(u_can_2,v_can_2,D_can_width));
            if (d_can_2>=0 && abs(d_can-d_can_2)<=redun_threshold) {
              support = true;
              break;
            }
          }

          // if we have no support => point is not redundant
          if (!support) {
            redundant = false;
            break;
          }
        }

        // invalidate support point if it is redundant
        if (redundant)
          *(D_can+getAddressOffsetImage(u_can,v_can,D_can_width)) = -1;
      }
    }
  }
}

void Elas::addCornerSupportPoints(vector<support_pt> &p_support) {

  // list of border points
  vector<support_pt> p_border;
  p_border.push_back(support_pt(0,0,0));
  p_border.push_back(support_pt(0,height-1,0));
  p_border.push_back(support_pt(width-1,0,0));
  p_border.push_back(support_pt(width-1,height-1,0));

  // find closest d
  for (int32_t i=0; i<p_border.size(); i++) {
    int32_t best_dist = 10000000;
    for (int32_t j=0; j<p_support.size(); j++) {
      int32_t du = p_border[i].u-p_support[j].u;
      int32_t dv = p_border[i].v-p_support[j].v;
      int32_t curr_dist = du*du+dv*dv;
      if (curr_dist<best_dist) {
        best_dist = curr_dist;
        p_border[i].d = p_support[j].d;
      }
    }
  }

  // for right image
  p_border.push_back(support_pt(p_border[2].u+p_border[2].d,p_border[2].v,p_border[2].d));
  p_border.push_back(support_pt(p_border[3].u+p_border[3].d,p_border[3].v,p_border[3].d));

  // add border points to support points
  for (int32_t i=0; i<p_border.size(); i++)
    p_support.push_back(p_border[i]);
}

inline int16_t Elas::computeMatchingDisparity (const int32_t &u,const int32_t &v,uint8_t* I1_desc,uint8_t* I2_desc,const bool &right_image) {

  const int32_t u_step      = 2;
  const int32_t v_step      = 2;
  const int32_t window_size = 3;

  int32_t desc_offset_1 = -16*u_step-16*width*v_step;
  int32_t desc_offset_2 = +16*u_step-16*width*v_step;
  int32_t desc_offset_3 = -16*u_step+16*width*v_step;
  int32_t desc_offset_4 = +16*u_step+16*width*v_step;

  __m128i xmm1,xmm2,xmm3,xmm4,xmm5,xmm6;

  // check if we are inside the image region
  if (u>=window_size+u_step && u<=width-window_size-1-u_step && v>=window_size+v_step && v<=height-window_size-1-v_step) {

    // compute desc and start addresses
    int32_t  line_offset = 16*width*v;
    uint8_t *I1_line_addr,*I2_line_addr;
    if (!right_image) {
      I1_line_addr = I1_desc+line_offset;
      I2_line_addr = I2_desc+line_offset;
    } else {
      I1_line_addr = I2_desc+line_offset;
      I2_line_addr = I1_desc+line_offset;
    }

    // compute I1 block start addresses
    uint8_t* I1_block_addr = I1_line_addr+16*u;
    uint8_t* I2_block_addr;

    // we require at least some texture
    int32_t sum = 0;
    for (int32_t i=0; i<16; i++)
      sum += abs((int32_t)(*(I1_block_addr+i))-128);
    if (sum<param.support_texture)
      return -1;

    // load first blocks to xmm registers
    xmm1 = _mm_load_si128((__m128i*)(I1_block_addr+desc_offset_1));
    xmm2 = _mm_load_si128((__m128i*)(I1_block_addr+desc_offset_2));
    xmm3 = _mm_load_si128((__m128i*)(I1_block_addr+desc_offset_3));
    xmm4 = _mm_load_si128((__m128i*)(I1_block_addr+desc_offset_4));

    // declare match energy for each disparity
    int32_t u_warp;

    // best match
    int16_t min_1_E = 32767;
    int16_t min_1_d = -1;
    int16_t min_2_E = 32767;
    int16_t min_2_d = -1;

    // get valid disparity range
    int32_t disp_min_valid = max(param.disp_min,0);
    int32_t disp_max_valid = param.disp_max;
    if (!right_image) disp_max_valid = min(param.disp_max,u-window_size-u_step);
    else              disp_max_valid = min(param.disp_max,width-u-window_size-u_step);

    // assume, that we can compute at least 10 disparities for this pixel
    if (disp_max_valid-disp_min_valid<10)
      return -1;

    // for all disparities do
    for (int16_t d=disp_min_valid; d<=disp_max_valid; d++) {

      // warp u coordinate
      if (!right_image) u_warp = u-d;
      else              u_warp = u+d;

      // compute I2 block start addresses
      I2_block_addr = I2_line_addr+16*u_warp;

      // compute match energy at this disparity
      xmm6 = _mm_load_si128((__m128i*)(I2_block_addr+desc_offset_1));
      xmm6 = _mm_sad_epu8(xmm1,xmm6);
      xmm5 = _mm_load_si128((__m128i*)(I2_block_addr+desc_offset_2));
      xmm6 = _mm_add_epi16(_mm_sad_epu8(xmm2,xmm5),xmm6);
      xmm5 = _mm_load_si128((__m128i*)(I2_block_addr+desc_offset_3));
      xmm6 = _mm_add_epi16(_mm_sad_epu8(xmm3,xmm5),xmm6);
      xmm5 = _mm_load_si128((__m128i*)(I2_block_addr+desc_offset_4));
      xmm6 = _mm_add_epi16(_mm_sad_epu8(xmm4,xmm5),xmm6);
      sum  = _mm_extract_epi16(xmm6,0)+_mm_extract_epi16(xmm6,4);

      // best + second best match
      if (sum<min_1_E) {
        min_1_E = sum;
        min_1_d = d;
      } else if (sum<min_2_E) {
        min_2_E = sum;
        min_2_d = d;
      }
    }

    // check if best and second best match are available and if matching ratio is sufficient
    if (min_1_d>=0 && min_2_d>=0 && (float)min_1_E<param.support_threshold*(float)min_2_E)
      return min_1_d;
    else
      return -1;

  } else
    return -1;
}

vector<Elas::support_pt> Elas::computeSupportMatches (uint8_t* I1_desc,uint8_t* I2_desc) {

  // be sure that at half resolution we only need data
  // from every second line!
  int32_t D_candidate_stepsize = param.candidate_stepsize;
  if (param.subsampling)
    D_candidate_stepsize += D_candidate_stepsize%2;

  // create matrix for saving disparity candidates
  int32_t D_can_width  = 0;
  int32_t D_can_height = 0;
  for (int32_t u=0; u<width;  u+=D_candidate_stepsize) D_can_width++;
  for (int32_t v=0; v<height; v+=D_candidate_stepsize) D_can_height++;
  int16_t* D_can = (int16_t*)calloc(D_can_width*D_can_height,sizeof(int16_t));

  // loop variables
  int32_t u,v;
  int16_t d,d2;

  // for all point candidates in image 1 do
  for (int32_t u_can=1; u_can<D_can_width; u_can++) {
    u = u_can*D_candidate_stepsize;
    for (int32_t v_can=1; v_can<D_can_height; v_can++) {
      v = v_can*D_candidate_stepsize;

      // initialize disparity candidate to invalid
      *(D_can+getAddressOffsetImage(u_can,v_can,D_can_width)) = -1;

      // find forwards
      d = computeMatchingDisparity(u,v,I1_desc,I2_desc,false);
      if (d>=0) {

        // find backwards
        d2 = computeMatchingDisparity(u-d,v,I1_desc,I2_desc,true);
        if (d2>=0 && abs(d-d2)<=param.lr_threshold)
          *(D_can+getAddressOffsetImage(u_can,v_can,D_can_width)) = d;
      }
    }
  }

  // remove inconsistent support points
  removeInconsistentSupportPoints(D_can,D_can_width,D_can_height);

  // remove support points on straight lines, since they are redundant
  // this reduces the number of triangles a little bit and hence speeds up
  // the triangulation process
  removeRedundantSupportPoints(D_can,D_can_width,D_can_height,5,1,true);
  removeRedundantSupportPoints(D_can,D_can_width,D_can_height,5,1,false);

  // move support points from image representation into a vector representation
  vector<support_pt> p_support;
  for (int32_t u_can=1; u_can<D_can_width; u_can++)
    for (int32_t v_can=1; v_can<D_can_height; v_can++)
      if (*(D_can+getAddressOffsetImage(u_can,v_can,D_can_width))>=0)
        p_support.push_back(support_pt(u_can*D_candidate_stepsize,
                                       v_can*D_candidate_stepsize,
                                       *(D_can+getAddressOffsetImage(u_can,v_can,D_can_width))));

  // if flag is set, add support points in image corners
  // with the same disparity as the nearest neighbor support point
  if (param.add_corners)
    addCornerSupportPoints(p_support);

  // free memory
  free(D_can);

  // return support point vector
  return p_support;
}

// 三角形化 triangulation & grid
vector<Elas::triangle> Elas::computeDelaunayTriangulation (vector<support_pt> p_support,int32_t right_image) {

  // input/output structure for triangulation
  struct triangulateio in, out;
  int32_t k;

  // inputs
  in.numberofpoints = p_support.size();
  in.pointlist = (float*)malloc(in.numberofpoints*2*sizeof(float));
  k=0;
  if (!right_image) {
    for (int32_t i=0; i<p_support.size(); i++) {
      in.pointlist[k++] = p_support[i].u;
      in.pointlist[k++] = p_support[i].v;
    }
  } else {
    for (int32_t i=0; i<p_support.size(); i++) {
      in.pointlist[k++] = p_support[i].u-p_support[i].d;
      in.pointlist[k++] = p_support[i].v;
    }
  }
  in.numberofpointattributes = 0;
  in.pointattributelist      = NULL;
  in.pointmarkerlist         = NULL;
  in.numberofsegments        = 0;
  in.numberofholes           = 0;
  in.numberofregions         = 0;
  in.regionlist              = NULL;

  // outputs
  out.pointlist              = NULL;
  out.pointattributelist     = NULL;
  out.pointmarkerlist        = NULL;
  out.trianglelist           = NULL;
  out.triangleattributelist  = NULL;
  out.neighborlist           = NULL;
  out.segmentlist            = NULL;
  out.segmentmarkerlist      = NULL;
  out.edgelist               = NULL;
  out.edgemarkerlist         = NULL;

  // do triangulation (z=zero-based, n=neighbors, Q=quiet, B=no boundary markers)
  char parameters[] = "zQB";
  triangulate(parameters, &in, &out, NULL);

  // put resulting triangles into vector tri
  vector<triangle> tri;
  k=0;
  for (int32_t i=0; i<out.numberoftriangles; i++) {
    tri.push_back(triangle(out.trianglelist[k],out.trianglelist[k+1],out.trianglelist[k+2]));
    k+=3;
  }

  // free memory used for triangulation
  free(in.pointlist);
  free(out.pointlist);
  free(out.trianglelist);

  // return triangles
  return tri;
}

void Elas::computeDisparityPlanes (vector<support_pt> p_support,vector<triangle> &tri,int32_t right_image) {

  // init matrices
  Matrix A(3,3);
  Matrix b(3,1);

  // for all triangles do
  for (int32_t i=0; i<tri.size(); i++) {

    // get triangle corner indices
    int32_t c1 = tri[i].c1;
    int32_t c2 = tri[i].c2;
    int32_t c3 = tri[i].c3;

    // compute matrix A for linear system of left triangle
    A.val[0][0] = p_support[c1].u;
    A.val[1][0] = p_support[c2].u;
    A.val[2][0] = p_support[c3].u;
    A.val[0][1] = p_support[c1].v; A.val[0][2] = 1;
    A.val[1][1] = p_support[c2].v; A.val[1][2] = 1;
    A.val[2][1] = p_support[c3].v; A.val[2][2] = 1;

    // compute vector b for linear system (containing the disparities)
    b.val[0][0] = p_support[c1].d;
    b.val[1][0] = p_support[c2].d;
    b.val[2][0] = p_support[c3].d;

    // on success of gauss jordan elimination
    if (b.solve(A)) {

      // grab results from b
      tri[i].t1a = b.val[0][0];
      tri[i].t1b = b.val[1][0];
      tri[i].t1c = b.val[2][0];

    // otherwise: invalid
    } else {
      tri[i].t1a = 0;
      tri[i].t1b = 0;
      tri[i].t1c = 0;
    }

    // compute matrix A for linear system of right triangle
    A.val[0][0] = p_support[c1].u-p_support[c1].d;
    A.val[1][0] = p_support[c2].u-p_support[c2].d;
    A.val[2][0] = p_support[c3].u-p_support[c3].d;
    A.val[0][1] = p_support[c1].v; A.val[0][2] = 1;
    A.val[1][1] = p_support[c2].v; A.val[1][2] = 1;
    A.val[2][1] = p_support[c3].v; A.val[2][2] = 1;

    // compute vector b for linear system (containing the disparities)
    b.val[0][0] = p_support[c1].d;
    b.val[1][0] = p_support[c2].d;
    b.val[2][0] = p_support[c3].d;

    // on success of gauss jordan elimination
    if (b.solve(A)) {

      // grab results from b
      tri[i].t2a = b.val[0][0];
      tri[i].t2b = b.val[1][0];
      tri[i].t2c = b.val[2][0];

    // otherwise: invalid
    } else {
      tri[i].t2a = 0;
      tri[i].t2b = 0;
      tri[i].t2c = 0;
    }
  }
}

void Elas::createGrid(vector<support_pt> p_support,int32_t* disparity_grid,int32_t* grid_dims,bool right_image) {

  // get grid dimensions
  int32_t grid_width  = grid_dims[1];
  int32_t grid_height = grid_dims[2];

  // allocate temporary memory
  int32_t* temp1 = (int32_t*)calloc((param.disp_max+1)*grid_height*grid_width,sizeof(int32_t));
  int32_t* temp2 = (int32_t*)calloc((param.disp_max+1)*grid_height*grid_width,sizeof(int32_t));

  // for all support points do
  for (int32_t i=0; i<p_support.size(); i++) {

    // compute disparity range to fill for this support point
    int32_t x_curr = p_support[i].u;
    int32_t y_curr = p_support[i].v;
    int32_t d_curr = p_support[i].d;
    int32_t d_min  = max(d_curr-1,0);
    int32_t d_max  = min(d_curr+1,param.disp_max);

    // fill disparity grid helper
    for (int32_t d=d_min; d<=d_max; d++) {
      int32_t x;
      if (!right_image)
        x = floor((float)(x_curr/param.grid_size));
      else
        x = floor((float)(x_curr-d_curr)/(float)param.grid_size);
      int32_t y = floor((float)y_curr/(float)param.grid_size);

      // point may potentially lay outside (corner points)
      if (x>=0 && x<grid_width &&y>=0 && y<grid_height) {
        int32_t addr = getAddressOffsetGrid(x,y,d,grid_width,param.disp_max+1);
        *(temp1+addr) = 1;
      }
    }
  }

  // diffusion pointers
  const int32_t* tl = temp1 + (0*grid_width+0)*(param.disp_max+1);
  const int32_t* tc = temp1 + (0*grid_width+1)*(param.disp_max+1);
  const int32_t* tr = temp1 + (0*grid_width+2)*(param.disp_max+1);
  const int32_t* cl = temp1 + (1*grid_width+0)*(param.disp_max+1);
  const int32_t* cc = temp1 + (1*grid_width+1)*(param.disp_max+1);
  const int32_t* cr = temp1 + (1*grid_width+2)*(param.disp_max+1);
  const int32_t* bl = temp1 + (2*grid_width+0)*(param.disp_max+1);
  const int32_t* bc = temp1 + (2*grid_width+1)*(param.disp_max+1);
  const int32_t* br = temp1 + (2*grid_width+2)*(param.disp_max+1);

  int32_t* result    = temp2 + (1*grid_width+1)*(param.disp_max+1);
  int32_t* end_input = temp1 + grid_width*grid_height*(param.disp_max+1);

  // diffuse temporary grid
  for( ; br != end_input; tl++, tc++, tr++, cl++, cc++, cr++, bl++, bc++, br++, result++ )
    *result = *tl | *tc | *tr | *cl | *cc | *cr | *bl | *bc | *br;

  // for all grid positions create disparity grid
  for (int32_t x=0; x<grid_width; x++) {
    for (int32_t y=0; y<grid_height; y++) {

      // start with second value (first is reserved for count)
      int32_t curr_ind = 1;

      // for all disparities do
      for (int32_t d=0; d<=param.disp_max; d++) {

        // if yes => add this disparity to current cell
        if (*(temp2+getAddressOffsetGrid(x,y,d,grid_width,param.disp_max+1))>0) {
          *(disparity_grid+getAddressOffsetGrid(x,y,curr_ind,grid_width,param.disp_max+2))=d;
          curr_ind++;
        }
      }

      // finally set number of indices
      *(disparity_grid+getAddressOffsetGrid(x,y,0,grid_width,param.disp_max+2))=curr_ind-1;
    }
  }

  // release temporary memory
  free(temp1);
  free(temp2);
}
// 匹配 matching  先验知识
inline void Elas::updatePosteriorMinimum(__m128i* I2_block_addr,const int32_t &d,const int32_t &w,
                                         const __m128i &xmm1,__m128i &xmm2,int32_t &val,int32_t &min_val,int32_t &min_d) {
  xmm2 = _mm_load_si128(I2_block_addr);
  xmm2 = _mm_sad_epu8(xmm1,xmm2);
  val  = _mm_extract_epi16(xmm2,0)+_mm_extract_epi16(xmm2,4)+w;
  if (val<min_val) {
    min_val = val;
    min_d   = d;
  }
}
// 
inline void Elas::updatePosteriorMinimum(__m128i* I2_block_addr,const int32_t &d,
                                         const __m128i &xmm1,__m128i &xmm2,int32_t &val,int32_t &min_val,int32_t &min_d) {
  xmm2 = _mm_load_si128(I2_block_addr);
  xmm2 = _mm_sad_epu8(xmm1,xmm2);
  val  = _mm_extract_epi16(xmm2,0)+_mm_extract_epi16(xmm2,4);
  if (val<min_val) {
    min_val = val;
    min_d   = d;
  }
}

inline void Elas::findMatch(int32_t &u,int32_t &v,float &plane_a,float &plane_b,float &plane_c,
                            int32_t* disparity_grid,int32_t *grid_dims,uint8_t* I1_desc,uint8_t* I2_desc,
                            int32_t *P,int32_t &plane_radius,bool &valid,bool &right_image,float* D){

  // get image width and height
  const int32_t disp_num    = grid_dims[0]-1;
  const int32_t window_size = 2;

  // address of disparity we want to compute
  uint32_t d_addr;
  if (param.subsampling) d_addr = getAddressOffsetImage(u/2,v/2,width/2);
  else                   d_addr = getAddressOffsetImage(u,v,width);

  // check if u is ok
  if (u<window_size || u>=width-window_size)
    return;

  // compute line start address
  int32_t  line_offset = 16*width*max(min(v,height-3),2);
  uint8_t *I1_line_addr,*I2_line_addr;
  if (!right_image) {
    I1_line_addr = I1_desc+line_offset;
    I2_line_addr = I2_desc+line_offset;
  } else {
    I1_line_addr = I2_desc+line_offset;
    I2_line_addr = I1_desc+line_offset;
  }

  // compute I1 block start address
  uint8_t* I1_block_addr = I1_line_addr+16*u;

  // does this patch have enough texture?
  int32_t sum = 0;
  for (int32_t i=0; i<16; i++)
    sum += abs((int32_t)(*(I1_block_addr+i))-128);
  if (sum<param.match_texture)
    return;

  // compute disparity, min disparity and max disparity of plane prior
  int32_t d_plane     = (int32_t)(plane_a*(float)u+plane_b*(float)v+plane_c);
  int32_t d_plane_min = max(d_plane-plane_radius,0);
  int32_t d_plane_max = min(d_plane+plane_radius,disp_num-1);

  // get grid pointer
  int32_t  grid_x    = (int32_t)floor((float)u/(float)param.grid_size);
  int32_t  grid_y    = (int32_t)floor((float)v/(float)param.grid_size);
  uint32_t grid_addr = getAddressOffsetGrid(grid_x,grid_y,0,grid_dims[1],grid_dims[0]);
  int32_t  num_grid  = *(disparity_grid+grid_addr);
  int32_t* d_grid    = disparity_grid+grid_addr+1;

  // loop variables
  int32_t d_curr, u_warp, val;
  int32_t min_val = 10000;
  int32_t min_d   = -1;
  __m128i xmm1    = _mm_load_si128((__m128i*)I1_block_addr);
  __m128i xmm2;

  // left image
  if (!right_image) {
    for (int32_t i=0; i<num_grid; i++) {
      d_curr = d_grid[i];
      if (d_curr<d_plane_min || d_curr>d_plane_max) {
        u_warp = u-d_curr;
        if (u_warp<window_size || u_warp>=width-window_size)
          continue;
        updatePosteriorMinimum((__m128i*)(I2_line_addr+16*u_warp),d_curr,xmm1,xmm2,val,min_val,min_d);
      }
    }
    for (d_curr=d_plane_min; d_curr<=d_plane_max; d_curr++) {
      u_warp = u-d_curr;
      if (u_warp<window_size || u_warp>=width-window_size)
        continue;
      updatePosteriorMinimum((__m128i*)(I2_line_addr+16*u_warp),d_curr,valid?*(P+abs(d_curr-d_plane)):0,xmm1,xmm2,val,min_val,min_d);
    }

  // right image
  } else {
    for (int32_t i=0; i<num_grid; i++) {
      d_curr = d_grid[i];
      if (d_curr<d_plane_min || d_curr>d_plane_max) {
        u_warp = u+d_curr;
        if (u_warp<window_size || u_warp>=width-window_size)
          continue;
        updatePosteriorMinimum((__m128i*)(I2_line_addr+16*u_warp),d_curr,xmm1,xmm2,val,min_val,min_d);
      }
    }
    for (d_curr=d_plane_min; d_curr<=d_plane_max; d_curr++) {
      u_warp = u+d_curr;
      if (u_warp<window_size || u_warp>=width-window_size)
        continue;
      updatePosteriorMinimum((__m128i*)(I2_line_addr+16*u_warp),d_curr,valid?*(P+abs(d_curr-d_plane)):0,xmm1,xmm2,val,min_val,min_d);
    }
  }

  // set disparity value
  if (min_d>=0) *(D+d_addr) = min_d; // MAP value (min neg-Log probability)
  else          *(D+d_addr) = -1;    // invalid disparity
}

// TODO: %2 => more elegantly
void Elas::computeDisparity(vector<support_pt> p_support,vector<triangle> tri,int32_t* disparity_grid,int32_t *grid_dims,
                            uint8_t* I1_desc,uint8_t* I2_desc,bool right_image,float* D) {

  // number of disparities
  const int32_t disp_num  = grid_dims[0]-1;

  // descriptor window_size
  int32_t window_size = 2;

  // init disparity image to -10
  if (param.subsampling) {
    for (int32_t i=0; i<(width/2)*(height/2); i++)
      *(D+i) = -10;
  } else {
    for (int32_t i=0; i<width*height; i++)
      *(D+i) = -10;
  }

  // pre-compute prior
  float two_sigma_squared = 2*param.sigma*param.sigma;
  int32_t* P = new int32_t[disp_num];
  for (int32_t delta_d=0; delta_d<disp_num; delta_d++)
    P[delta_d] = (int32_t)((-log(param.gamma+exp(-delta_d*delta_d/two_sigma_squared))+log(param.gamma))/param.beta);
  int32_t plane_radius = (int32_t)max((float)ceil(param.sigma*param.sradius),(float)2.0);

  // loop variables
  int32_t c1, c2, c3;
  float plane_a,plane_b,plane_c,plane_d;

  // for all triangles do
  for (uint32_t i=0; i<tri.size(); i++) {

    // get plane parameters
    uint32_t p_i = i*3;
    if (!right_image) {
      plane_a = tri[i].t1a;
      plane_b = tri[i].t1b;
      plane_c = tri[i].t1c;
      plane_d = tri[i].t2a;
    } else {
      plane_a = tri[i].t2a;
      plane_b = tri[i].t2b;
      plane_c = tri[i].t2c;
      plane_d = tri[i].t1a;
    }

    // triangle corners
    c1 = tri[i].c1;
    c2 = tri[i].c2;
    c3 = tri[i].c3;

    // sort triangle corners wrt. u (ascending)
    float tri_u[3];
    if (!right_image) {
      tri_u[0] = p_support[c1].u;
      tri_u[1] = p_support[c2].u;
      tri_u[2] = p_support[c3].u;
    } else {
      tri_u[0] = p_support[c1].u-p_support[c1].d;
      tri_u[1] = p_support[c2].u-p_support[c2].d;
      tri_u[2] = p_support[c3].u-p_support[c3].d;
    }
    float tri_v[3];
    tri_v[0] = p_support[c1].v;
    tri_v[1] = p_support[c2].v;
    tri_v[2] = p_support[c3].v;

    for (uint32_t j=0; j<3; j++) {
      for (uint32_t k=0; k<j; k++) {
        if (tri_u[k]>tri_u[j]) {
          float tri_u_temp = tri_u[j]; tri_u[j] = tri_u[k]; tri_u[k] = tri_u_temp;
          float tri_v_temp = tri_v[j]; tri_v[j] = tri_v[k]; tri_v[k] = tri_v_temp;
        }
      }
    }

    // rename corners
    float A_u = tri_u[0]; float A_v = tri_v[0];
    float B_u = tri_u[1]; float B_v = tri_v[1];
    float C_u = tri_u[2]; float C_v = tri_v[2];

    // compute straight lines connecting triangle corners
    float AB_a = 0; float AC_a = 0; float BC_a = 0;
    if ((int32_t)(A_u)!=(int32_t)(B_u)) AB_a = (A_v-B_v)/(A_u-B_u);
    if ((int32_t)(A_u)!=(int32_t)(C_u)) AC_a = (A_v-C_v)/(A_u-C_u);
    if ((int32_t)(B_u)!=(int32_t)(C_u)) BC_a = (B_v-C_v)/(B_u-C_u);
    float AB_b = A_v-AB_a*A_u;
    float AC_b = A_v-AC_a*A_u;
    float BC_b = B_v-BC_a*B_u;

    // a plane is only valid if itself and its projection
    // into the other image is not too much slanted
    bool valid = fabs(plane_a)<0.7 && fabs(plane_d)<0.7;

    // first part (triangle corner A->B)
    if ((int32_t)(A_u)!=(int32_t)(B_u)) {
      for (int32_t u=max((int32_t)A_u,0); u<min((int32_t)B_u,width); u++){
        if (!param.subsampling || u%2==0) {
          int32_t v_1 = (uint32_t)(AC_a*(float)u+AC_b);
          int32_t v_2 = (uint32_t)(AB_a*(float)u+AB_b);
          for (int32_t v=min(v_1,v_2); v<max(v_1,v_2); v++)
            if (!param.subsampling || v%2==0) {
              findMatch(u,v,plane_a,plane_b,plane_c,disparity_grid,grid_dims,
                        I1_desc,I2_desc,P,plane_radius,valid,right_image,D);
            }
        }
      }
    }

    // second part (triangle corner B->C)
    if ((int32_t)(B_u)!=(int32_t)(C_u)) {
      for (int32_t u=max((int32_t)B_u,0); u<min((int32_t)C_u,width); u++){
        if (!param.subsampling || u%2==0) {
          int32_t v_1 = (uint32_t)(AC_a*(float)u+AC_b);
          int32_t v_2 = (uint32_t)(BC_a*(float)u+BC_b);
          for (int32_t v=min(v_1,v_2); v<max(v_1,v_2); v++)
            if (!param.subsampling || v%2==0) {
              findMatch(u,v,plane_a,plane_b,plane_c,disparity_grid,grid_dims,
                        I1_desc,I2_desc,P,plane_radius,valid,right_image,D);
            }
        }
      }
    }

  }

  delete[] P;
}

// L/R consistency check
void Elas::leftRightConsistencyCheck(float* D1,float* D2) {

  // get disparity image dimensions
  int32_t D_width  = width;
  int32_t D_height = height;
  if (param.subsampling) {
    D_width  = width/2;
    D_height = height/2;
  }

  // make a copy of both images
  float* D1_copy = (float*)malloc(D_width*D_height*sizeof(float));
  float* D2_copy = (float*)malloc(D_width*D_height*sizeof(float));
  memcpy(D1_copy,D1,D_width*D_height*sizeof(float));
  memcpy(D2_copy,D2,D_width*D_height*sizeof(float));

  // loop variables
  uint32_t addr,addr_warp;
  float    u_warp_1,u_warp_2,d1,d2;

  // for all image points do
  for (int32_t u=0; u<D_width; u++) {
    for (int32_t v=0; v<D_height; v++) {

      // compute address (u,v) and disparity value
      addr     = getAddressOffsetImage(u,v,D_width);
      d1       = *(D1_copy+addr);
      d2       = *(D2_copy+addr);
      if (param.subsampling) {
        u_warp_1 = (float)u-d1/2;
        u_warp_2 = (float)u+d2/2;
      } else {
        u_warp_1 = (float)u-d1;
        u_warp_2 = (float)u+d2;
      }


      // check if left disparity is valid
      if (d1>=0 && u_warp_1>=0 && u_warp_1<D_width) {

        // compute warped image address
        addr_warp = getAddressOffsetImage((int32_t)u_warp_1,v,D_width);

        // if check failed
        if (fabs(*(D2_copy+addr_warp)-d1)>param.lr_threshold)
          *(D1+addr) = -10;

      // set invalid
      } else
        *(D1+addr) = -10;

      // check if right disparity is valid
      if (d2>=0 && u_warp_2>=0 && u_warp_2<D_width) {

        // compute warped image address
        addr_warp = getAddressOffsetImage((int32_t)u_warp_2,v,D_width);

        // if check failed
        if (fabs(*(D1_copy+addr_warp)-d2)>param.lr_threshold)
          *(D2+addr) = -10;

      // set invalid
      } else
        *(D2+addr) = -10;
    }
  }

  // release memory
  free(D1_copy);
  free(D2_copy);
}

// 后处理 postprocessing
void Elas::removeSmallSegments (float* D) {

  // get disparity image dimensions
  int32_t D_width        = width;
  int32_t D_height       = height;
  int32_t D_speckle_size = param.speckle_size;
  if (param.subsampling) {
    D_width        = width/2;
    D_height       = height/2;
    D_speckle_size = sqrt((float)param.speckle_size)*2;
  }

  // allocate memory on heap for dynamic programming arrays
  int32_t *D_done     = (int32_t*)calloc(D_width*D_height,sizeof(int32_t));
  int32_t *seg_list_u = (int32_t*)calloc(D_width*D_height,sizeof(int32_t));
  int32_t *seg_list_v = (int32_t*)calloc(D_width*D_height,sizeof(int32_t));
  int32_t seg_list_count;
  int32_t seg_list_curr;
  int32_t u_neighbor[4];
  int32_t v_neighbor[4];
  int32_t u_seg_curr;
  int32_t v_seg_curr;

  // declare loop variables
  int32_t addr_start, addr_curr, addr_neighbor;

  // for all pixels do
  for (int32_t u=0; u<D_width; u++) {
    for (int32_t v=0; v<D_height; v++) {

      // get address of first pixel in this segment
      addr_start = getAddressOffsetImage(u,v,D_width);

      // if this pixel has not already been processed
      if (*(D_done+addr_start)==0) {

        // init segment list (add first element
        // and set it to be the next element to check)
        *(seg_list_u+0) = u;
        *(seg_list_v+0) = v;
        seg_list_count  = 1;
        seg_list_curr   = 0;

        // add neighboring segments as long as there
        // are none-processed pixels in the seg_list;
        // none-processed means: seg_list_curr<seg_list_count
        while (seg_list_curr<seg_list_count) {

          // get current position from seg_list
          u_seg_curr = *(seg_list_u+seg_list_curr);
          v_seg_curr = *(seg_list_v+seg_list_curr);

          // get address of current pixel in this segment
          addr_curr = getAddressOffsetImage(u_seg_curr,v_seg_curr,D_width);

          // fill list with neighbor positions
          u_neighbor[0] = u_seg_curr-1; v_neighbor[0] = v_seg_curr;
          u_neighbor[1] = u_seg_curr+1; v_neighbor[1] = v_seg_curr;
          u_neighbor[2] = u_seg_curr;   v_neighbor[2] = v_seg_curr-1;
          u_neighbor[3] = u_seg_curr;   v_neighbor[3] = v_seg_curr+1;

          // for all neighbors do
          for (int32_t i=0; i<4; i++) {

            // check if neighbor is inside image
            if (u_neighbor[i]>=0 && v_neighbor[i]>=0 && u_neighbor[i]<D_width && v_neighbor[i]<D_height) {

              // get neighbor pixel address
              addr_neighbor = getAddressOffsetImage(u_neighbor[i],v_neighbor[i],D_width);

              // check if neighbor has not been added yet and if it is valid
              if (*(D_done+addr_neighbor)==0 && *(D+addr_neighbor)>=0) {

                // is the neighbor similar to the current pixel
                // (=belonging to the current segment)
                if (fabs(*(D+addr_curr)-*(D+addr_neighbor))<=param.speckle_sim_threshold) {

                  // add neighbor coordinates to segment list
                  *(seg_list_u+seg_list_count) = u_neighbor[i];
                  *(seg_list_v+seg_list_count) = v_neighbor[i];
                  seg_list_count++;

                  // set neighbor pixel in I_done to "done"
                  // (otherwise a pixel may be added 2 times to the list, as
                  //  neighbor of one pixel and as neighbor of another pixel)
                  *(D_done+addr_neighbor) = 1;
                }
              }

            }
          }

          // set current pixel in seg_list to "done"
          seg_list_curr++;

          // set current pixel in I_done to "done"
          *(D_done+addr_curr) = 1;

        } // end: while (seg_list_curr<seg_list_count)

        // if segment NOT large enough => invalidate pixels
        if (seg_list_count<D_speckle_size) {

          // for all pixels in current segment invalidate pixels
          for (int32_t i=0; i<seg_list_count; i++) {
            addr_curr = getAddressOffsetImage(*(seg_list_u+i),*(seg_list_v+i),D_width);
            *(D+addr_curr) = -10;
          }
        }
      } // end: if (*(I_done+addr_start)==0)

    }
  }

  // free memory
  free(D_done);
  free(seg_list_u);
  free(seg_list_v);
}

void Elas::gapInterpolation(float* D) {

  // get disparity image dimensions
  int32_t D_width          = width;
  int32_t D_height         = height;
  int32_t D_ipol_gap_width = param.ipol_gap_width;
  if (param.subsampling) {
    D_width          = width/2;
    D_height         = height/2;
    D_ipol_gap_width = param.ipol_gap_width/2+1;
  }

  // discontinuity threshold
  float discon_threshold = 3.0;

  // declare loop variables
  int32_t count,addr,v_first,v_last,u_first,u_last;
  float   d1,d2,d_ipol;

  // 1. Row-wise:
  // for each row do
  for (int32_t v=0; v<D_height; v++) {

    // init counter
    count = 0;

    // for each element of the row do
    for (int32_t u=0; u<D_width; u++) {

      // get address of this location
      addr = getAddressOffsetImage(u,v,D_width);

      // if disparity valid
      if (*(D+addr)>=0) {

        // check if speckle is small enough
        if (count>=1 && count<=D_ipol_gap_width) {

          // first and last value for interpolation
          u_first = u-count;
          u_last  = u-1;

          // if value in range
          if (u_first>0 && u_last<D_width-1) {

            // compute mean disparity
            d1 = *(D+getAddressOffsetImage(u_first-1,v,D_width));
            d2 = *(D+getAddressOffsetImage(u_last+1,v,D_width));
            if (fabs(d1-d2)<discon_threshold) d_ipol = (d1+d2)/2;
            else                              d_ipol = min(d1,d2);

            // set all values to d_ipol
            for (int32_t u_curr=u_first; u_curr<=u_last; u_curr++)
              *(D+getAddressOffsetImage(u_curr,v,D_width)) = d_ipol;
          }

        }

        // reset counter
        count = 0;

      // otherwise increment counter
      } else {
        count++;
      }
    }

    // if full size disp map requested
    if (param.add_corners) {

      // extrapolate to the left
      for (int32_t u=0; u<D_width; u++) {

        // get address of this location
        addr = getAddressOffsetImage(u,v,D_width);

        // if disparity valid
        if (*(D+addr)>=0) {
          for (int32_t u2=max(u-D_ipol_gap_width,0); u2<u; u2++)
            *(D+getAddressOffsetImage(u2,v,D_width)) = *(D+addr);
          break;
        }
      }

      // extrapolate to the right
      for (int32_t u=D_width-1; u>=0; u--) {

        // get address of this location
        addr = getAddressOffsetImage(u,v,D_width);

        // if disparity valid
        if (*(D+addr)>=0) {
          for (int32_t u2=u; u2<=min(u+D_ipol_gap_width,D_width-1); u2++)
            *(D+getAddressOffsetImage(u2,v,D_width)) = *(D+addr);
          break;
        }
      }
    }
  }

  // 2. Column-wise:
  // for each column do
  for (int32_t u=0; u<D_width; u++) {

    // init counter
    count = 0;

    // for each element of the column do
    for (int32_t v=0; v<D_height; v++) {

      // get address of this location
      addr = getAddressOffsetImage(u,v,D_width);

      // if disparity valid
      if (*(D+addr)>=0) {

        // check if gap is small enough
        if (count>=1 && count<=D_ipol_gap_width) {

          // first and last value for interpolation
          v_first = v-count;
          v_last  = v-1;

          // if value in range
          if (v_first>0 && v_last<D_height-1) {

            // compute mean disparity
            d1 = *(D+getAddressOffsetImage(u,v_first-1,D_width));
            d2 = *(D+getAddressOffsetImage(u,v_last+1,D_width));
            if (fabs(d1-d2)<discon_threshold) d_ipol = (d1+d2)/2;
            else                              d_ipol = min(d1,d2);

            // set all values to d_ipol
            for (int32_t v_curr=v_first; v_curr<=v_last; v_curr++)
              *(D+getAddressOffsetImage(u,v_curr,D_width)) = d_ipol;
          }

        }

        // reset counter
        count = 0;

      // otherwise increment counter
      } else {
        count++;
      }
    }
  }
}

// 可选后处理 optional postprocessing
// implements approximation to bilateral filtering
void Elas::adaptiveMean (float* D) {

  // get disparity image dimensions
  int32_t D_width          = width;
  int32_t D_height         = height;
  if (param.subsampling) {
    D_width          = width/2;
    D_height         = height/2;
  }

  // allocate temporary memory
  float* D_copy = (float*)malloc(D_width*D_height*sizeof(float));
  float* D_tmp  = (float*)malloc(D_width*D_height*sizeof(float));
  memcpy(D_copy,D,D_width*D_height*sizeof(float));

  // zero input disparity maps to -10 (this makes the bilateral
  // weights of all valid disparities to 0 in this region)
  for (int32_t i=0; i<D_width*D_height; i++) {
    if (*(D+i)<0) {
      *(D_copy+i) = -10;
      *(D_tmp+i)  = -10;
    }
  }

  __m128 xconst0 = _mm_set1_ps(0);
  __m128 xconst4 = _mm_set1_ps(4);
  __m128 xval,xweight1,xweight2,xfactor1,xfactor2;

  float *val     = (float *)_mm_malloc(8*sizeof(float),16);
  float *weight  = (float*)_mm_malloc(4*sizeof(float),16);
  float *factor  = (float*)_mm_malloc(4*sizeof(float),16);

  // set absolute mask
  __m128 xabsmask = _mm_set1_ps(0x7FFFFFFF);

  // when doing subsampling: 4 pixel bilateral filter width
  if (param.subsampling) {

    // horizontal filter
    for (int32_t v=3; v<D_height-3; v++) {

      // init
      for (int32_t u=0; u<3; u++)
        val[u] = *(D_copy+v*D_width+u);

      // loop
      for (int32_t u=3; u<D_width; u++) {

        // set
        float val_curr = *(D_copy+v*D_width+(u-1));
        val[u%4] = *(D_copy+v*D_width+u);

        xval     = _mm_load_ps(val);
        xweight1 = _mm_sub_ps(xval,_mm_set1_ps(val_curr));
        xweight1 = _mm_and_ps(xweight1,xabsmask);
        xweight1 = _mm_sub_ps(xconst4,xweight1);
        xweight1 = _mm_max_ps(xconst0,xweight1);
        xfactor1 = _mm_mul_ps(xval,xweight1);

        _mm_store_ps(weight,xweight1);
        _mm_store_ps(factor,xfactor1);

        float weight_sum = weight[0]+weight[1]+weight[2]+weight[3];
        float factor_sum = factor[0]+factor[1]+factor[2]+factor[3];

        if (weight_sum>0) {
          float d = factor_sum/weight_sum;
          if (d>=0) *(D_tmp+v*D_width+(u-1)) = d;
        }
      }
    }

    // vertical filter
    for (int32_t u=3; u<D_width-3; u++) {

      // init
      for (int32_t v=0; v<3; v++)
        val[v] = *(D_tmp+v*D_width+u);

      // loop
      for (int32_t v=3; v<D_height; v++) {

        // set
        float val_curr = *(D_tmp+(v-1)*D_width+u);
        val[v%4] = *(D_tmp+v*D_width+u);

        xval     = _mm_load_ps(val);
        xweight1 = _mm_sub_ps(xval,_mm_set1_ps(val_curr));
        xweight1 = _mm_and_ps(xweight1,xabsmask);
        xweight1 = _mm_sub_ps(xconst4,xweight1);
        xweight1 = _mm_max_ps(xconst0,xweight1);
        xfactor1 = _mm_mul_ps(xval,xweight1);

        _mm_store_ps(weight,xweight1);
        _mm_store_ps(factor,xfactor1);

        float weight_sum = weight[0]+weight[1]+weight[2]+weight[3];
        float factor_sum = factor[0]+factor[1]+factor[2]+factor[3];

        if (weight_sum>0) {
          float d = factor_sum/weight_sum;
          if (d>=0) *(D+(v-1)*D_width+u) = d;
        }
      }
    }

  // full resolution: 8 pixel bilateral filter width
  } else {


    // horizontal filter
    for (int32_t v=3; v<D_height-3; v++) {

      // init
      for (int32_t u=0; u<7; u++)
        val[u] = *(D_copy+v*D_width+u);

      // loop
      for (int32_t u=7; u<D_width; u++) {

        // set
        float val_curr = *(D_copy+v*D_width+(u-3));
        val[u%8] = *(D_copy+v*D_width+u);

        xval     = _mm_load_ps(val);
        xweight1 = _mm_sub_ps(xval,_mm_set1_ps(val_curr));
        xweight1 = _mm_and_ps(xweight1,xabsmask);
        xweight1 = _mm_sub_ps(xconst4,xweight1);
        xweight1 = _mm_max_ps(xconst0,xweight1);
        xfactor1 = _mm_mul_ps(xval,xweight1);

        xval     = _mm_load_ps(val+4);
        xweight2 = _mm_sub_ps(xval,_mm_set1_ps(val_curr));
        xweight2 = _mm_and_ps(xweight2,xabsmask);
        xweight2 = _mm_sub_ps(xconst4,xweight2);
        xweight2 = _mm_max_ps(xconst0,xweight2);
        xfactor2 = _mm_mul_ps(xval,xweight2);

        xweight1 = _mm_add_ps(xweight1,xweight2);
        xfactor1 = _mm_add_ps(xfactor1,xfactor2);

        _mm_store_ps(weight,xweight1);
        _mm_store_ps(factor,xfactor1);

        float weight_sum = weight[0]+weight[1]+weight[2]+weight[3];
        float factor_sum = factor[0]+factor[1]+factor[2]+factor[3];

        if (weight_sum>0) {
          float d = factor_sum/weight_sum;
          if (d>=0) *(D_tmp+v*D_width+(u-3)) = d;
        }
      }
    }

    // vertical filter
    for (int32_t u=3; u<D_width-3; u++) {

      // init
      for (int32_t v=0; v<7; v++)
        val[v] = *(D_tmp+v*D_width+u);

      // loop
      for (int32_t v=7; v<D_height; v++) {

        // set
        float val_curr = *(D_tmp+(v-3)*D_width+u);
        val[v%8] = *(D_tmp+v*D_width+u);

        xval     = _mm_load_ps(val);
        xweight1 = _mm_sub_ps(xval,_mm_set1_ps(val_curr));
        xweight1 = _mm_and_ps(xweight1,xabsmask);
        xweight1 = _mm_sub_ps(xconst4,xweight1);
        xweight1 = _mm_max_ps(xconst0,xweight1);
        xfactor1 = _mm_mul_ps(xval,xweight1);

        xval     = _mm_load_ps(val+4);
        xweight2 = _mm_sub_ps(xval,_mm_set1_ps(val_curr));
        xweight2 = _mm_and_ps(xweight2,xabsmask);
        xweight2 = _mm_sub_ps(xconst4,xweight2);
        xweight2 = _mm_max_ps(xconst0,xweight2);
        xfactor2 = _mm_mul_ps(xval,xweight2);

        xweight1 = _mm_add_ps(xweight1,xweight2);
        xfactor1 = _mm_add_ps(xfactor1,xfactor2);

        _mm_store_ps(weight,xweight1);
        _mm_store_ps(factor,xfactor1);

        float weight_sum = weight[0]+weight[1]+weight[2]+weight[3];
        float factor_sum = factor[0]+factor[1]+factor[2]+factor[3];

        if (weight_sum>0) {
          float d = factor_sum/weight_sum;
          if (d>=0) *(D+(v-3)*D_width+u) = d;
        }
      }
    }
  }

  // free memory
  _mm_free(val);
  _mm_free(weight);
  _mm_free(factor);
  free(D_copy);
  free(D_tmp);
}

void Elas::median (float* D) {

  // get disparity image dimensions
  int32_t D_width          = width;
  int32_t D_height         = height;
  if (param.subsampling) {
    D_width          = width/2;
    D_height         = height/2;
  }

  // temporary memory
  float *D_temp = (float*)calloc(D_width*D_height,sizeof(float));

  int32_t window_size = 3;

  float *vals = new float[window_size*2+1];
  int32_t i,j;
  float temp;

  // first step: horizontal median filter
  for (int32_t u=window_size; u<D_width-window_size; u++) {
    for (int32_t v=window_size; v<D_height-window_size; v++) {
      if (*(D+getAddressOffsetImage(u,v,D_width))>=0) {
        j = 0;
        for (int32_t u2=u-window_size; u2<=u+window_size; u2++) {
          temp = *(D+getAddressOffsetImage(u2,v,D_width));
          i = j-1;
          while (i>=0 && *(vals+i)>temp) {
            *(vals+i+1) = *(vals+i);
            i--;
          }
          *(vals+i+1) = temp;
          j++;
        }
        *(D_temp+getAddressOffsetImage(u,v,D_width)) = *(vals+window_size);
      } else {
        *(D_temp+getAddressOffsetImage(u,v,D_width)) = *(D+getAddressOffsetImage(u,v,D_width));
      }

    }
  }

  // second step: vertical median filter
  for (int32_t u=window_size; u<D_width-window_size; u++) {
    for (int32_t v=window_size; v<D_height-window_size; v++) {
      if (*(D+getAddressOffsetImage(u,v,D_width))>=0) {
        j = 0;
        for (int32_t v2=v-window_size; v2<=v+window_size; v2++) {
          temp = *(D_temp+getAddressOffsetImage(u,v2,D_width));
          i = j-1;
          while (i>=0 && *(vals+i)>temp) {
            *(vals+i+1) = *(vals+i);
            i--;
          }
          *(vals+i+1) = temp;
          j++;
        }
        *(D+getAddressOffsetImage(u,v,D_width)) = *(vals+window_size);
      } else {
        *(D+getAddressOffsetImage(u,v,D_width)) = *(D+getAddressOffsetImage(u,v,D_width));
      }
    }
  }

  free(D_temp);
  free(vals);
}

// 上述过程包装类 
StereoELAS::StereoELAS(int mindis, int dispRange)
{
	minDisparity = mindis;//最小视差
	disparityRange = dispRange;// 视差 Range
	elas.param.disp_min=mindis;
	elas.param.disp_max=mindis + dispRange;//最大视差

	elas.param.postprocess_only_left = false;
}
void StereoELAS::operator()(cv::Mat& leftimg, cv::Mat& rightimg, cv::Mat& leftdisp, cv::Mat& rightdisp, int bd)
{
	Mat l,r;
// 原图像转换为灰度图像
	if(leftimg.channels()==3){cvtColor(leftimg,l,CV_BGR2GRAY);cout<<"convert gray"<<endl;}
	else l=leftimg;
	if(rightimg.channels()==3)cvtColor(rightimg,r,CV_BGR2GRAY);
	else r=rightimg;
// 对左右视图的左边进行边界延拓，以获取与原始视图相同大小的有效视差区域
	Mat lb,rb;
	cv::copyMakeBorder(l,lb,0,0,bd,bd,cv::BORDER_REPLICATE);
	cv::copyMakeBorder(r,rb,0,0,bd,bd,cv::BORDER_REPLICATE);

	const cv::Size imsize = lb.size();
	const int32_t dims[3] = {imsize.width,imsize.height,imsize.width}; // bytes per line = width

	cv::Mat leftdpf = cv::Mat::zeros(imsize,CV_32F);
	cv::Mat rightdpf = cv::Mat::zeros(imsize,CV_32F);
	elas.process(lb.data,rb.data,leftdpf.ptr<float>(0),rightdpf.ptr<float>(0),dims);

// 截取与原始画面对应的视差区域（舍去加宽的部分）
	Mat disp;
	Mat(leftdpf(cv::Rect(bd,0,leftimg.cols,leftimg.rows))).copyTo(disp);
	disp.convertTo(leftdisp,CV_16S,16);
	Mat(rightdpf(cv::Rect(bd,0,leftimg.cols,leftimg.rows))).copyTo(disp);
	disp.convertTo(rightdisp,CV_16S,16);
}

void StereoELAS::operator()(cv::Mat& leftimg, cv::Mat& rightimg, cv::Mat& leftdisp, int bd)
{
	Mat temp;
	StereoELAS::operator()(leftimg,rightimg,leftdisp,temp,bd);
}
/*
void StereoELAS::check(Mat& leftim, Mat& rightim, Mat& disp, StereoEval& eval)
{
	string wname = "ELAS";
	namedWindow(wname);
	int nsigma = 0;
	createTrackbar("N sigma",wname,&nsigma,1000);

	int key = 0;
	Mat disp16;
	Mat dispr16;
	Mat lim,rim;

	Stat st;
	while(key!='q')
	{
		addNoise(leftim,lim,nsigma/10.0);
		addNoise(rightim,rim,nsigma/10.0);
		
		operator()(lim,rim,disp16,dispr16,16);

		Mat disp8;
		disp16.convertTo(disp8,CV_8U,2/16.0);
		eval(disp8,1.0,true);
		st.push_back(eval.all);
		st.show();
		imshow(wname,disp);
		if(key=='r')st.clear();
		key = waitKey(1);
	}
}*/

