/*
 *
 */

#ifndef MYSLAM_G2O_TYPES_H//防止头文件重复引用
#define MYSLAM_G2O_TYPES_H//宏定义

#include "myslam/common_include.h"//常用的头文件 放在一起 化繁为简
#include "camera.h"

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
/*使用非线性优化求解相机位姿态*/
namespace myslam//命令空间下 防止定义的出其他库里的同名函数
{
	//边 3D-3D 点对  更新  pose  and   point
      class EdgeProjectXYZRGBD : public g2o::BaseBinaryEdge<3, Eigen::Vector3d, g2o::VertexSBAPointXYZ, g2o::VertexSE3Expmap>
      {
      public:
	  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	  virtual void computeError();//计算误差
	  virtual void linearizeOplus();//雅克比矩阵
	  virtual bool read( std::istream& in ){}
	  virtual bool write( std::ostream& out) const {}
	  
      };

      // only to optimize the pose, no point
      // 边 3D-3D 点对  更新  pose 
      class EdgeProjectXYZRGBDPoseOnly: public g2o::BaseUnaryEdge<3, Eigen::Vector3d, g2o::VertexSE3Expmap >
      {
      public:
	  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	  // Error: measure = R*point+t
	  virtual void computeError();
	  virtual void linearizeOplus();
	  
	  virtual bool read( std::istream& in ){}
	  virtual bool write( std::ostream& out) const {}
	  
	  Vector3d point_;
      };

      // 边 2D-3D 点对  更新  pose 
      class EdgeProjectXYZ2UVPoseOnly: public g2o::BaseUnaryEdge<2, Eigen::Vector2d, g2o::VertexSE3Expmap >
      {
      public:
	  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	  
	  virtual void computeError();
	  virtual void linearizeOplus();
	  
	  virtual bool read( std::istream& in ){}
	  virtual bool write(std::ostream& os) const {};
	  
	  Vector3d point_;
	  Camera* camera_;
      };
      //  // 边 2D-3D 点对  更新  pose  point   g2o 内置函数就有  
      // g2o::EdgeProjectXYZ2UV();  //也可以自己实现

}


#endif // MYSLAM_G2O_TYPES_H
