/*
 * PnPProblem.cpp
 *  2d-3d点对匹配 Rt变换求解器
 */
#include <iostream>
#include <sstream>

#include "PnPProblem.h"
#include "Mesh.h"

#include <opencv2/calib3d/calib3d.hpp>

/* Functions headers */
cv::Point3f CROSS(cv::Point3f v1, cv::Point3f v2);//向量叉乘
double DOT(cv::Point3f v1, cv::Point3f v2);//向量点乘法 
cv::Point3f SUB(cv::Point3f v1, cv::Point3f v2);//向量减法
// 找最近的3d点
cv::Point3f get_nearest_3D_point(std::vector<cv::Point3f> &points_list, cv::Point3f origin);
//=================向量 叉乘===============================
// Functions for Möller–Trumbore intersection algorithm
// (v1.x v1.y v1.z)  
// 得到向量　　叉乘 
// (v2.x v2.y v2.z)
// ====>
//     v1.y * v2.z - v1.z * v2.y
//     v1.z * v2.x - v1.x * v2.z
//     v1.x * v2.y - v1.y * v2.x
//=====>写成叉乘矩阵　
//
//     0    -v1.z   v1.y       v2.x
//    v1.z    0    -v1.x    *  v2.y
//    -v1.y v1.x    0          v2.z
cv::Point3f CROSS(cv::Point3f v1, cv::Point3f v2)
{
  cv::Point3f tmp_p;
  tmp_p.x =  v1.y*v2.z - v1.z*v2.y;
  tmp_p.y =  v1.z*v2.x - v1.x*v2.z;
  tmp_p.z =  v1.x*v2.y - v1.y*v2.x;
  return tmp_p;
}
//=========向量点乘=====================
//得到常数　　对应元素相乘后加在一起
double DOT(cv::Point3f v1, cv::Point3f v2)
{
  return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}
//========向量相减====================
//得到向量　　对应元素相减
cv::Point3f SUB(cv::Point3f v1, cv::Point3f v2)
{
  cv::Point3f tmp_p;
  tmp_p.x =  v1.x - v2.x;
  tmp_p.y =  v1.y - v2.y;
  tmp_p.z =  v1.z - v2.z;
  return tmp_p;
}

//End functions for Möller–Trumbore intersection algorithm
// 在给定的3d点容器里(就存储了两个点)　找一个与　特定3d点　最相邻　的点　
// Function to get the nearest 3D point to the Ray origin
// 注意这里　3d点容器　为引用类型　避免拷贝
cv::Point3f get_nearest_3D_point(std::vector<cv::Point3f> &points_list, cv::Point3f origin)
{
  cv::Point3f p1 = points_list[0];//　第一个点
  cv::Point3f p2 = points_list[1];//　第二个点

  double d1 = std::sqrt( std::pow(p1.x-origin.x, 2) + std::pow(p1.y-origin.y, 2) + std::pow(p1.z-origin.z, 2) );
  double d2 = std::sqrt( std::pow(p2.x-origin.x, 2) + std::pow(p2.y-origin.y, 2) + std::pow(p2.z-origin.z, 2) );

  if(d1 < d2)//与第一个点最近
  {
      return p1;
  }
  else//与第二个点最近
  {
      return p2;
  }
}

// Custom constructor given the intrinsic camera parameters
// PnP求解器类　默认构造函数
PnPProblem::PnPProblem(const double params[])
{
  // PnP求解器参数初始化
  // 相机内参数矩阵
  _A_matrix = cv::Mat::zeros(3, 3, CV_64FC1);   // intrinsic camera parameters
  _A_matrix.at<double>(0, 0) = params[0];       //      [ fx   0  cx ]
  _A_matrix.at<double>(1, 1) = params[1];       //      [  0  fy  cy ]
  _A_matrix.at<double>(0, 2) = params[2];       //      [  0   0   1 ]
  _A_matrix.at<double>(1, 2) = params[3];
  _A_matrix.at<double>(2, 2) = 1;
  // 旋转矩阵  R   需要求解的
  _R_matrix = cv::Mat::zeros(3, 3, CV_64FC1);   // rotation matrix
  // 平移向量　t
  _t_matrix = cv::Mat::zeros(3, 1, CV_64FC1);   // translation matrix
  // 变换矩阵　P = [R t]
  _P_matrix = cv::Mat::zeros(3, 4, CV_64FC1);   // rotation-translation matrix

}

// PnP求解器类　默认析构函数
PnPProblem::~PnPProblem()
{
  // TODO Auto-generated destructor stub
}

// 由　旋转矩阵R　和　平移向量t　　构造变换矩阵P = [R t]　
void PnPProblem::set_P_matrix( const cv::Mat &R_matrix, const cv::Mat &t_matrix)
{
  // Rotation-Translation Matrix Definition
  _P_matrix.at<double>(0,0) = R_matrix.at<double>(0,0);
  _P_matrix.at<double>(0,1) = R_matrix.at<double>(0,1);
  _P_matrix.at<double>(0,2) = R_matrix.at<double>(0,2);
  _P_matrix.at<double>(1,0) = R_matrix.at<double>(1,0);
  _P_matrix.at<double>(1,1) = R_matrix.at<double>(1,1);
  _P_matrix.at<double>(1,2) = R_matrix.at<double>(1,2);
  _P_matrix.at<double>(2,0) = R_matrix.at<double>(2,0);
  _P_matrix.at<double>(2,1) = R_matrix.at<double>(2,1);
  _P_matrix.at<double>(2,2) = R_matrix.at<double>(2,2);
  _P_matrix.at<double>(0,3) = t_matrix.at<double>(0);
  _P_matrix.at<double>(1,3) = t_matrix.at<double>(1);
  _P_matrix.at<double>(2,3) = t_matrix.at<double>(2);
}

// 2D-3D　单个点对　估计变换矩阵　cv::solvePnP() 得到　旋转向量 ----> 旋转矩阵
// Estimate the pose given a list of 2D/3D correspondences and the method to use
bool PnPProblem::estimatePose( const std::vector<cv::Point3f> &list_points3d,
                               const std::vector<cv::Point2f> &list_points2d,
                               int flags)
{
  cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64FC1);// distCoeffs是4×1、1×4、5×1或1×5的相机畸变向量
  cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);// 3*1 旋转向量 方向表示旋转轴　　大小表示　旋转角度
  cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);// 3*1 平移向量
  bool useExtrinsicGuess = false;//按初始值不断优化
  // 变量useExtrinsicGuess的值为true，函数将使用所提供的rvec和tvec向量作为旋转和平移的初始值，然后会不断优化它们。
  // 函数按最小化重投影误差来计算摄像机的变换，即让所得的投影向量imagePoints与被投影向量之间的距离平方和最小。
  // Pose estimation
  bool correspondence = cv::solvePnP( list_points3d,// objectPoints 对象所在空间的三维点　 
				      list_points2d,// imagePoints是相应图像点（或投影）
				      _A_matrix,    // cameraMatrix是3×3的摄像机内部参数矩阵
				      distCoeffs,   // distCoeffs是4×1、1×4、5×1或1×5的相机畸变向量
				      rvec,    
// rvec是输出的旋转向量，该向量将点由模型坐标系统 转换为 摄像机坐标系统  	worid 到camera相机
				      tvec,        // tvec是输出的平移向量
				      useExtrinsicGuess, 
				      flags);
// 欧拉角的表示方式里，三个旋转轴一般是随着刚体在运动 row pitch yaw
// 旋转向量　转到 旋转矩阵(旋转矩阵　也叫　方向余弦矩阵(Direction Cosine Matrix)，简称DCM　)
// 直观来讲，一个四维向量(theta,x,y,z)就可以表示出三维空间任意的旋转。
// 注意，这里的三维向量(x,y,z)只是用来表示旋转轴axis的方向朝向，
// 因此更紧凑的表示方式是用一个单位向量来表示方向axis，
// 而用该三维向量的长度来表示角度值theta。
// 这样以来，可以用一个三维向量(theta*x, theta*y, theta*z)就可以表示出三维空间任意的旋转，
// 前提是其中(x,y,z)是单位向量。这就是旋转向量(Rotation Vector)的表示方式，
// ===============四元素==========================================
// 同上，假设(x,y,z)是axis方向的单位向量，theta是绕axis转过的角度，
// 那么四元数可以表示为[cos(theta/2), x*sin(theta/2), y*sin(theta/2), z*sin(theta/2)]。
  Rodrigues(rvec,_R_matrix);// 旋转向量 ----> 旋转矩阵
  _t_matrix = tvec;// 平移向量

  // 调用类的函数　设置　变换矩阵　Set projection matrix
  this->set_P_matrix(_R_matrix, _t_matrix);

  return correspondence;
}

// 随机序列采样　一致性　估计　变换矩阵
// 得到符合变换的内点外点数量找到　内点比例最大的　一个变换
// Estimate the pose given a list of 2D/3D correspondences with RANSAC and the method to use
void PnPProblem::estimatePoseRANSAC( const std::vector<cv::Point3f> &list_points3d,   // 对象所在空间的三维点　　　列表
                                     const std::vector<cv::Point2f> &list_points2d,   // 相应图像二维点（或投影）　列表
                                     int flags, cv::Mat &inliers, int iterationsCount,// PnP method; inliers container
                                     float reprojectionError, double confidence )     // Ransac parameters
{
  cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64FC1);// distCoeffs是4×1、1×4、5×1或1×5的相机畸变向量
  cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);      // 3*1 旋转向量 方向表示旋转轴　　大小表示　旋转角度
  cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);      // 3*1 平移向量
  bool useExtrinsicGuess = false;
  // 变量useExtrinsicGuess的值为true，函数将使用所提供的rvec和tvec向量作为旋转和平移的初始值，然后会不断优化它们。
  // 函数按最小化重投影误差来计算摄像机的变换，即让所得的投影向量imagePoints与被投影向量之间的距离平方和最小。
  cv::solvePnPRansac( list_points3d, // 对象所在空间的三维点　　　列表
                      list_points2d, // 相应图像二维点（或投影）　列表
                      _A_matrix,     // cameraMatrix是3×3的摄像机内部参数矩阵
                      distCoeffs,    // distCoeffs是4×1、1×4、5×1或1×5的相机畸变向量
                      rvec, // rvec是输出的旋转向量，该向量将点由模型坐标系统 转换为 摄像机坐标系统  worid 到camera相机
                      tvec, // tvec是输出的平移向量
                      useExtrinsicGuess, // 使用初始优化？
                      iterationsCount,   // 随机采样迭代计数
                      reprojectionError, // 投影误差
                      confidence,        // 内点所占比例　　可信度
		      inliers,           // 内点数量
		      flags );

  Rodrigues(rvec,_R_matrix);// 旋转向量 ----> 旋转矩阵
  _t_matrix = tvec;         // 平移向量
  // 调用类的函数　设置　变换矩阵　Set projection matrix
  this->set_P_matrix(_R_matrix, _t_matrix); // set rotation-translation matrix
}
//　网格文件　顶点　三维坐标　--> 2d　
// Given the mesh, backproject the 3D points to 2D to verify the pose estimation
std::vector<cv::Point2f> PnPProblem::verify_points(Mesh *mesh)
{
  std::vector<cv::Point2f> verified_points_2d;
  for( int i = 0; i < mesh->getNumVertices(); i++)
  {
    cv::Point3f point3d = mesh->getVertex(i);//　网格　顶点　三维坐标
    cv::Point2f point2d = this->backproject3DPoint(point3d);//3d点反投影到2d点
    verified_points_2d.push_back(point2d);//3d --> 2d
  }
  return verified_points_2d;
}

// 反投影　3d --> 2d　　　[u v 1]'　＝　K * P * [x y z 1]'　　再归一化　　得到二维像素点坐标[u v]'
// Backproject a 3D point to 2D using the estimated pose parameters
cv::Point2f PnPProblem::backproject3DPoint(const cv::Point3f &point3d)
{
  // 三维点　的　齐次表达式　3D point vector [x y z 1]'
  cv::Mat point3d_vec = cv::Mat(4, 1, CV_64FC1);
  point3d_vec.at<double>(0) = point3d.x;
  point3d_vec.at<double>(1) = point3d.y;
  point3d_vec.at<double>(2) = point3d.z;
  point3d_vec.at<double>(3) = 1;

  // 二维点　的　齐次表达式2D point vector [u v 1]'
  cv::Mat point2d_vec = cv::Mat(4, 1, CV_64FC1);
  point2d_vec = _A_matrix * _P_matrix * point3d_vec;

  // 将第三个量归一化　得到2d 点　Normalization of [u v]'
  cv::Point2f point2d;
  point2d.x = (float)(point2d_vec.at<double>(0) / point2d_vec.at<double>(2));
  point2d.y = (float)(point2d_vec.at<double>(1) / point2d_vec.at<double>(2));

  return point2d;
}

// 反投影　2d -> 3d 查看是否在　为物体网格模型的平面上
// Back project a 2D point to 3D and returns if it's on the object surface
bool PnPProblem::backproject2DPoint(const Mesh *mesh, const cv::Point2f &point2d, cv::Point3f &point3d)
{
  // 三角形面 顶点 索引  n*3
  std::vector<std::vector<int> > triangles_list = mesh->getTrianglesList();

  double lambda = 8;
  double u = point2d.x;// 二维像素点坐标[u v]'
  double v = point2d.y;

  // 二维点　的　齐次表达式
  cv::Mat point2d_vec = cv::Mat::ones(3, 1, CV_64F); // 3x1
  point2d_vec.at<double>(0) = u * lambda;
  point2d_vec.at<double>(1) = v * lambda;
  point2d_vec.at<double>(2) = lambda;

  // camera相机  coordinates
  cv::Mat X_c = _A_matrix.inv() * point2d_vec ; // 3x1

  // Point in world coordinates
  cv::Mat X_w = _R_matrix.inv() * ( X_c - _t_matrix ); // 3x1

  // Center of projection
  cv::Mat C_op = cv::Mat(_R_matrix.inv()).mul(-1) * _t_matrix; // 3x1

  // Ray direction vector
  cv::Mat ray = X_w - C_op; // 3x1
  ray = ray / cv::norm(ray); // 3x1

  // Set up Ray
  Ray R((cv::Point3f)C_op, (cv::Point3f)ray);

  // A vector to store the intersections found
  std::vector<cv::Point3f> intersections_list;

  // Loop for all the triangles and check the intersection
  for (unsigned int i = 0; i < triangles_list.size(); i++)
  {
    cv::Point3f V0 = mesh->getVertex(triangles_list[i][0]);
    cv::Point3f V1 = mesh->getVertex(triangles_list[i][1]);
    cv::Point3f V2 = mesh->getVertex(triangles_list[i][2]);

    Triangle T(i, V0, V1, V2);

    double out;
    if(this->intersect_MollerTrumbore(R, T, &out))
    {
      cv::Point3f tmp_pt = R.getP0() + out*R.getP1(); // P = O + t*D
      intersections_list.push_back(tmp_pt);
    }
  }

  // If there are intersection, find the nearest one
  if (!intersections_list.empty())
  {
    point3d = get_nearest_3D_point(intersections_list, R.getP0());
    return true;
  }
  else
  {
    return false;
  }
}

// Möller–Trumbore intersection algorithm
bool PnPProblem::intersect_MollerTrumbore(Ray &Ray, Triangle &Triangle, double *out)
{
  const double EPSILON = 0.000001;

  cv::Point3f e1, e2;
  cv::Point3f P, Q, T;
  double det, inv_det, u, v;
  double t;

  cv::Point3f V1 = Triangle.getV0();  // Triangle vertices
  cv::Point3f V2 = Triangle.getV1();
  cv::Point3f V3 = Triangle.getV2();

  cv::Point3f O = Ray.getP0(); // Ray origin
  cv::Point3f D = Ray.getP1(); // Ray direction

  //Find vectors for two edges sharing V1
  e1 = SUB(V2, V1);
  e2 = SUB(V3, V1);

  // Begin calculation determinant - also used to calculate U parameter
  P = CROSS(D, e2);

  // If determinant is near zero, ray lie in plane of triangle
  det = DOT(e1, P);

  //NOT CULLING
  if(det > -EPSILON && det < EPSILON) return false;
  inv_det = 1.f / det;

  //calculate distance from V1 to ray origin
  T = SUB(O, V1);

  //Calculate u parameter and test bound
  u = DOT(T, P) * inv_det;

  //The intersection lies outside of the triangle
  if(u < 0.f || u > 1.f) return false;

  //Prepare to test v parameter
  Q = CROSS(T, e1);

  //Calculate V parameter and test bound
  v = DOT(D, Q) * inv_det;

  //The intersection lies outside of the triangle
  if(v < 0.f || u + v  > 1.f) return false;

  t = DOT(e2, Q) * inv_det;

  if(t > EPSILON) { //ray intersection
    *out = t;
    return true;
  }

  // No hit, no win
  return false;
}
