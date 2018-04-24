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

// Main header file. Include this to use libelas in your code.

#ifndef __ELAS_H__
#define __ELAS_H__

#pragma once
#include <opencv2/opencv.hpp>
using namespace cv;

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <vector>
#include <emmintrin.h>

// define fixed-width datatypes for Visual Studio projects
#ifndef _MSC_VER
  #include <stdint.h>
#else
  typedef __int8            int8_t;
  typedef __int16           int16_t;
  typedef __int32           int32_t;
  typedef __int64           int64_t;
  typedef unsigned __int8   uint8_t;
  typedef unsigned __int16  uint16_t;
  typedef unsigned __int32  uint32_t;
  typedef unsigned __int64  uint64_t;
#endif

#ifdef PROFILE
#include "timer.h"
#endif

class Elas {
  
public:
  
  enum setting {ROBOTICS,MIDDLEBURY};
  
  // 参数设置 parameter settings
  struct parameters {
    int32_t disp_min;               // min disparity 最小视差
    int32_t disp_max;               // max disparity  视差
    float   support_threshold;      // max. uniqueness ratio (best vs. second best support match)
    int32_t support_texture;        // min texture for support points
    int32_t candidate_stepsize;     // step size of regular grid on which support points are matched
    int32_t incon_window_size;      // window size of inconsistent support point check
    int32_t incon_threshold;        // disparity similarity threshold for support point to be considered consistent
    int32_t incon_min_support;      // minimum number of consistent support points
    bool    add_corners;            // add support points at image corners with nearest neighbor disparities
    int32_t grid_size;              // size of neighborhood for additional support point extrapolation
    float   beta;                   // image likelihood parameter
    float   gamma;                  // prior constant
    float   sigma;                  // prior sigma
    float   sradius;                // prior sigma radius
    int32_t match_texture;          // min texture for dense matching
    int32_t lr_threshold;           // disparity threshold for left/right consistency check
    float   speckle_sim_threshold;  // similarity threshold for speckle segmentation
    int32_t speckle_size;           // maximal size of a speckle (small speckles get removed)
    int32_t ipol_gap_width;         // interpolate small gaps (left<->right, top<->bottom)
    bool    filter_median;          // optional median filter (approximated)
    bool    filter_adaptive_mean;   // optional adaptive mean filter (approximated)
    bool    postprocess_only_left;  // saves time by not postprocessing the right image
    bool    subsampling;            // saves time by only computing disparities for each 2nd pixel
                                    // note: for this option D1 and D2 must be passed with size
                                    //       width/2 x height/2 (rounded towards zero)
    
    // 默认参数设置 constructor
    parameters (setting s=ROBOTICS) {
      
      // default settings in a robotics environment
      // (do not produce results in half-occluded areas
      //  and are a bit more robust towards lighting etc.)
      // 下列参数对光照更鲁棒
      if (s==ROBOTICS) {
        disp_min              = 0;
        disp_max              = 255;// 视差范围
        support_threshold     = 0.85f;//支持特征点最小阈值 最好匹配距离小于次好匹配距离 0.85
        support_texture       = 10;
        candidate_stepsize    = 5;
        incon_window_size     = 5;
        incon_threshold       = 5;
        incon_min_support     = 5;
        add_corners           = 0;
        grid_size             = 20;
        beta                  = 0.02f;
        gamma                 = 3;
        sigma                 = 1;
        sradius               = 2;
        match_texture         = 1;
        lr_threshold          = 2;
        speckle_sim_threshold = 1;
        speckle_size          = 200;
        ipol_gap_width        = 3;
        filter_median         = 0;
        filter_adaptive_mean  = 1;
        postprocess_only_left = 1;
        subsampling           = 0;
        
      // default settings for middlebury benchmark
      // (interpolate all missing disparities)
      } else {
        disp_min              = 0;
        disp_max              = 255;
        support_threshold     = 0.95f;
        support_texture       = 10;
        candidate_stepsize    = 5;
        incon_window_size     = 5;
        incon_threshold       = 5;
        incon_min_support     = 5;
        add_corners           = 1;
        grid_size             = 20;
        beta                  = 0.02f;
        gamma                 = 5;
        sigma                 = 1;
        sradius               = 3;
        match_texture         = 0;
        lr_threshold          = 2;
        speckle_sim_threshold = 1;
        speckle_size          = 200;
        ipol_gap_width        = 5000;
        filter_median         = 1;
        filter_adaptive_mean  = 0;
        postprocess_only_left = 0;
        subsampling           = 0;
      }
    }
  };
  // 默认构造函数 constructor, input: parameters  
  Elas (parameters param2) : param(param2) {}
  Elas () :param(MIDDLEBURY){}
  // Elas () :param(ROBOTICS){}
  // 默认析构函数 deconstructor
  ~Elas () {}
  
  // 匹配函数 matching function
  // inputs: pointers to left (I1) and right (I2) intensity image (uint8, input)  左右输入图像
  //         pointers to left (D1) and right (D2) disparity image (float, output) 左右输出视差图
  //         dims[0] = width of I1 and I2  宽度
  //         dims[1] = height of I1 and I2 高度
  //         dims[2] = bytes per line (often equal to width, but allowed to differ)
  //         note: D1 and D2 must be allocated before (bytes per line = width)
  //               if subsampling is not active their size is width x height,
  //               otherwise width/2 x height/2 (rounded towards zero)
  void process (uint8_t* I1,uint8_t* I2,float* D1,float* D2,const int32_t* dims);
  // 参数设置 parameter set
  parameters param; 
// 私有 
private:
  //支持点 结构体
  struct support_pt {
    int32_t u;//像素坐标
    int32_t v;
    int32_t d;//深度
    support_pt(int32_t u,int32_t v,int32_t d):u(u),v(v),d(d){}
  };
  // 支持点组成的三角形
  struct triangle {
    int32_t c1,c2,c3;
    float   t1a,t1b,t1c;
    float   t2a,t2b,t2c;
    triangle(int32_t c1,int32_t c2,int32_t c3):c1(c1),c2(c2),c3(c3){}
  };

  inline uint32_t getAddressOffsetImage (const int32_t& u,const int32_t& v,const int32_t& width) {
    return v*width+u;
  }

  inline uint32_t getAddressOffsetGrid (const int32_t& x,const int32_t& y,const int32_t& d,const int32_t& width,const int32_t& disp_num) {
    return (y*width+x)*disp_num+d;
  }

  // 支持点 support point functions
  void removeInconsistentSupportPoints (int16_t* D_can,int32_t D_can_width,int32_t D_can_height);
  void removeRedundantSupportPoints (int16_t* D_can,int32_t D_can_width,int32_t D_can_height,
                                     int32_t redun_max_dist, int32_t redun_threshold, bool vertical);
  void addCornerSupportPoints (std::vector<support_pt> &p_support);
  inline int16_t computeMatchingDisparity (const int32_t &u,const int32_t &v,uint8_t* I1_desc,uint8_t* I2_desc,const bool &right_image);
  std::vector<support_pt> computeSupportMatches (uint8_t* I1_desc,uint8_t* I2_desc);

  // 三角形化 triangulation & grid
  std::vector<triangle> computeDelaunayTriangulation (std::vector<support_pt> p_support,int32_t right_image);
  void computeDisparityPlanes (std::vector<support_pt> p_support,std::vector<triangle> &tri,int32_t right_image);
  void createGrid (std::vector<support_pt> p_support,int32_t* disparity_grid,int32_t* grid_dims,bool right_image);

  // 匹配 matching
  inline void updatePosteriorMinimum (__m128i* I2_block_addr,const int32_t &d,const int32_t &w,
                                      const __m128i &xmm1,__m128i &xmm2,int32_t &val,int32_t &min_val,int32_t &min_d);
  inline void updatePosteriorMinimum (__m128i* I2_block_addr,const int32_t &d,
                                      const __m128i &xmm1,__m128i &xmm2,int32_t &val,int32_t &min_val,int32_t &min_d);
  inline void findMatch (int32_t &u,int32_t &v,float &plane_a,float &plane_b,float &plane_c,
                         int32_t* disparity_grid,int32_t *grid_dims,uint8_t* I1_desc,uint8_t* I2_desc,
                         int32_t *P,int32_t &plane_radius,bool &valid,bool &right_image,float* D);
  void computeDisparity (std::vector<support_pt> p_support,std::vector<triangle> tri,int32_t* disparity_grid,int32_t* grid_dims,
                         uint8_t* I1_desc,uint8_t* I2_desc,bool right_image,float* D);

  // L/R consistency check
  void leftRightConsistencyCheck (float* D1,float* D2);
  
  // 后处理 postprocessing
  void removeSmallSegments (float* D);
  void gapInterpolation (float* D);

  // 可选后处理 optional postprocessing
  void adaptiveMean (float* D);// implements approximation to bilateral filtering
  void median (float* D);//median filter
  
  // 参数设置 parameter set  
//  parameters param;// private -->  public
  
  // 内存分配 memory aligned input images + dimensions
  uint8_t *I1,*I2;
  int32_t width,height,bpl;//维度
  
  // profiling timer
#ifdef PROFILE
  Timer timer;
#endif
};
// 上述过程包装类 
class StereoELAS
{
public:
	//Elas::parameters param(Elas::MIDDLEBURY);

	int minDisparity;//最小视差
	int disparityRange;// 视差 Range
public:
	Elas elas;//elas类
	StereoELAS(int mindis, int dispRange);

        //StereoELAS(void);//:minDisparity(0),disparityRange(128){
	//	elas.param.disp_min = minDisparity;
	//	elas.param.disp_max = minDisparity + disparityRange;//最大视差
	//	elas.param.postprocess_only_left = false;
        //}
        //  ()括号运算符 定义
	void operator()(cv::Mat& leftimg, cv::Mat& rightimg, cv::Mat& leftdisp, cv::Mat& rightdisp, int border);
	void operator()(cv::Mat& leftimg, cv::Mat& rightimg, cv::Mat& leftdisp, int border);
//	void StereoEfficientLargeScale::check(Mat& leftim, Mat& rightim, Mat& disp, StereoEval& eval);
};

#endif
