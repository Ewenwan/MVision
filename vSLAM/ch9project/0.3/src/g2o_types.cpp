#include "myslam/g2o_types.h"
/*
 * 2D-3D点对
 * 对两幅图像 提取特征点对  p  u 得到2D-2D像素坐标点对
 * 第二幅图像 像素坐标点 u  通过 图像对应的深度图 转化成 该相机坐标系下的 	3D点  P
 * 形成 2D-3D点对
 * 
 * 误差 e =   p - p' = p - K * P' =  p - K * T * P  = p - K * exp(f) * P  = p - K * (R * P + t)
 * 【1】 e 对 ∇f导数 可以更新 T 求最优化的 转换矩阵
 *            e 对 ∇f导数 =   e 对  p‘导数 * p‘ 对P’导数 * P’ 对∇f导数 =1 * p‘ 对P’导数 * P’ 对∇f导数
 * 【2】 e 对 P的导数  = e 对 P'导数 * P' 对 P导数  = e 对  p‘导数 * p‘ 对P’导数 * P' 对 P导数   = 1 * p‘ 对P’导数 * R  更新 第二幅图像下的 3维点坐标
 * 
 * 仅有【1】 优化 POSE
 * 【1】【2】都有 优化 POSE 和 POINT
 *  
 *  3D-3D点对
 * 对两幅图像 提取特征点对  p  u 得到2D-2D像素坐标点对
 * 两幅图像的二维像素点坐标 根据 两幅图像对应 的深度图   转换成 在两幅图像 相机坐标系下的 3D坐标  P    Q
 * 形成  3D-3D点对
 * 
 * 误差 e = P - P' = P - T*Q =  P -  exp(f) * Q
 * 【1】 e 对 ∇f导数 可以更新 T 求最优化的 转换矩阵
 *            e 对 ∇f导数 =   e 对  对P’导数 * P’ 对∇f导数 =1 * P’ 对∇f导数
 * 【2】e 对 Q 导数 = e 对  对P’导数 * P’ 对 Q导数 = 1 * T = T    优化 第二幅图像像极坐标系下的 三维点 坐标
 *   仅有【1】 优化 POSE
 * 【1】【2】都有 优化 POSE 和 POINT
 * 
 */
namespace myslam//命令空间下 防止定义的出其他库里的同名函数
{
      //投影误差计算   3D-3D 点对  同时 优化 3D点 和 位姿
      void EdgeProjectXYZRGBD::computeError()
      {
	  const g2o::VertexSBAPointXYZ* point = static_cast<const g2o::VertexSBAPointXYZ*> ( _vertices[0] );// 3维点
	  const g2o::VertexSE3Expmap* pose = static_cast<const g2o::VertexSE3Expmap*> ( _vertices[1] );// 位姿
	  _error = _measurement - pose->estimate().map ( point->estimate() );// 测量值 - T * P 误差
      }
      //雅克比矩阵计算
      void EdgeProjectXYZRGBD::linearizeOplus()
      {
	  g2o::VertexSE3Expmap* pose = static_cast<g2o::VertexSE3Expmap *> ( _vertices[1] );// 位姿
	  g2o::SE3Quat T ( pose->estimate() );
	  g2o::VertexSBAPointXYZ* point = static_cast<g2o::VertexSBAPointXYZ*> ( _vertices[0] );// 3维点
	  Eigen::Vector3d xyz = point->estimate();
	  Eigen::Vector3d xyz_trans = T.map ( xyz );//转换的点
	  double x = xyz_trans[0];
	  double y = xyz_trans[1];
	  double z = xyz_trans[2];

	  // e = P - T*P'
	  // e对 P‘的导数 为T
	  _jacobianOplusXi = - T.rotation().toRotationMatrix();// 旋转矩阵

	  // 3×6的雅克比矩阵  误差  对应的 导数  优化变量更新 增量
	      /*
	      *  e = P - T*P'
	      * // e对 T的导数 为 P'对T的导数
	      *  P'对∇f的偏导数 = [ I  -P'叉乘矩阵] 3*6大小   平移在前  旋转在后
	      *  = [ 1 0  0   0   Z'   -Y' 
	      *       0 1  0  -Z'  0    X'
	      *       0 0  1  Y'   -X   0]
	      * 旋转在前  平移在后
	      *  = [   0   Z'   -Y' 1 0  0 
	      *        -Z'  0    X'  0 1  0 
	      *         Y'   -X   0  0 0  1]
	      * 
	      * J = - P'对∇f的偏导数
	      *  = [   0   -Z'   Y'  -1  0  0 
	      *         Z'   0    -X'  0 -1  0 
	      *         -Y'  X’    0   0  0 -1]
	      */     
	  _jacobianOplusXj ( 0,0 ) = 0;
	  _jacobianOplusXj ( 0,1 ) = -z;
	  _jacobianOplusXj ( 0,2 ) = y;
	  _jacobianOplusXj ( 0,3 ) = -1;
	  _jacobianOplusXj ( 0,4 ) = 0;
	  _jacobianOplusXj ( 0,5 ) = 0;

	  _jacobianOplusXj ( 1,0 ) = z;
	  _jacobianOplusXj ( 1,1 ) = 0;
	  _jacobianOplusXj ( 1,2 ) = -x;
	  _jacobianOplusXj ( 1,3 ) = 0;
	  _jacobianOplusXj ( 1,4 ) = -1;
	  _jacobianOplusXj ( 1,5 ) = 0;

	  _jacobianOplusXj ( 2,0 ) = -y;
	  _jacobianOplusXj ( 2,1 ) = x;
	  _jacobianOplusXj ( 2,2 ) = 0;
	  _jacobianOplusXj ( 2,3 ) = 0;
	  _jacobianOplusXj ( 2,4 ) = 0;
	  _jacobianOplusXj ( 2,5 ) = -1;
      }

      //投影误差计算   3D-3D 点对  仅仅优化 位姿 不优化 3D坐标
      void EdgeProjectXYZRGBDPoseOnly::computeError()
      {
	  const g2o::VertexSE3Expmap* pose = static_cast<const g2o::VertexSE3Expmap*> ( _vertices[0] );
	  _error = _measurement - pose->estimate().map ( point_ );
      }
      //雅克比矩阵计算
      void EdgeProjectXYZRGBDPoseOnly::linearizeOplus()
      {
	  g2o::VertexSE3Expmap* pose = static_cast<g2o::VertexSE3Expmap*> ( _vertices[0] );
	  g2o::SE3Quat T ( pose->estimate() );
	  Vector3d xyz_trans = T.map ( point_ );
	  double x = xyz_trans[0];
	  double y = xyz_trans[1];
	  double z = xyz_trans[2];

	  _jacobianOplusXi ( 0,0 ) = 0;
	  _jacobianOplusXi ( 0,1 ) = -z;
	  _jacobianOplusXi ( 0,2 ) = y;
	  _jacobianOplusXi ( 0,3 ) = -1;
	  _jacobianOplusXi ( 0,4 ) = 0;
	  _jacobianOplusXi ( 0,5 ) = 0;

	  _jacobianOplusXi ( 1,0 ) = z;
	  _jacobianOplusXi ( 1,1 ) = 0;
	  _jacobianOplusXi ( 1,2 ) = -x;
	  _jacobianOplusXi ( 1,3 ) = 0;
	  _jacobianOplusXi ( 1,4 ) = -1;
	  _jacobianOplusXi ( 1,5 ) = 0;

	  _jacobianOplusXi ( 2,0 ) = -y;
	  _jacobianOplusXi ( 2,1 ) = x;
	  _jacobianOplusXi ( 2,2 ) = 0;
	  _jacobianOplusXi ( 2,3 ) = 0;
	  _jacobianOplusXi ( 2,4 ) = 0;
	  _jacobianOplusXi ( 2,5 ) = -1;
      }
      
	//投影误差计算  2D-3D点对	  重投影误差  仅仅优化 位姿
      void EdgeProjectXYZ2UVPoseOnly::computeError()
      {
	  const g2o::VertexSE3Expmap* pose = static_cast<const g2o::VertexSE3Expmap*> ( _vertices[0] );
	  _error = _measurement - camera_->camera2pixel ( 
	      pose->estimate().map(point_) );// T*P 到相机坐标系下 3维点 在转换到 像素坐标系下  重投影到 第二幅图像的 像素坐标系下
      }
      //雅克比矩阵计算
      void EdgeProjectXYZ2UVPoseOnly::linearizeOplus()
      {
	  g2o::VertexSE3Expmap* pose = static_cast<g2o::VertexSE3Expmap*> ( _vertices[0] );
	  g2o::SE3Quat T ( pose->estimate() );
	  Vector3d xyz_trans = T.map ( point_ );
	  double x = xyz_trans[0];
	  double y = xyz_trans[1];
	  double z = xyz_trans[2];
	  double z_2 = z*z;
	  /* P为第二帧像极坐标系下的点坐标 P'为P为第二帧像极坐标系下的点坐标转换到第一帧相机坐标系下
	   * e = p - K*exp(f)*P =p - K * P' = p - u
	   * e对u导数 * u对∇f导数 = 
	   1 * u对∇f的偏导数 = u对P'偏导 *   P'对∇f的偏导数  2*6 矩阵  与图像无关
	   
	    * u对P'的偏导数 = - [ u对X'的偏导数 u对Y'的偏导数 u对Z'的偏导数;
	    *                                   v对X'的偏导数 v对Y'的偏导数  v对Z'的偏导数]  = - [ fx/Z'   0        -fx * X'/Z' ^2 
	    *                                                                                                                        0       fy/Z'    -fy* Y'/Z' ^2]
	    *  P'对∇f的偏导数 = [ I  -P'叉乘矩阵] 3*6大小   平移在前  旋转在后
	    *  = [ 1 0  0   0   Z'   -Y' 
	    *       0 1  0  -Z'  0    X'
	    *       0 0  1  Y'   -X   0]
	    * 有向量 t = [ a1 a2 a3] 其
	    * 叉乘矩阵 = [0  -a3  a2;
	    *                     a3  0  -a1; 
	    *                    -a2 a1  0 ]  
	   
      * =  两者相乘得到 
      
      * = - [fx/Z   0       -fx * X/Z ^2   -fx * X*Y/Z^2      fx + fx * X^2/Z^2    -fx*Y/Z
      *           0    fy/Z   -fy* Y/Z^2    -fy -fy* Y^2/Z^2   fy * X*Y'/Z^2          fy*X/Z   ] 
      * 如果是 旋转在前 平移在后 调换前三列  后三列 
      // 旋转在前 平移在后   g2o   负号乘了进去
      *=  [ fx *X*Y/Z^2           -fx *(1 + X^2/Z^2)   fx*Y/Z  -fx/Z   0        fx * X/Z^2 
      *      fy *(1 + Y^2/Z^2)  -fy * X*Y/Z^2           -fy*X/Z   0      -fy/Z   fy* Y/Z^2     ]    
      */
	  _jacobianOplusXi ( 0,0 ) =  x*y/z_2 *camera_->fx_;
	  _jacobianOplusXi ( 0,1 ) = - ( 1+ ( x*x/z_2 ) ) *camera_->fx_;
	  _jacobianOplusXi ( 0,2 ) = y/z * camera_->fx_;
	  _jacobianOplusXi ( 0,3 ) = -1./z * camera_->fx_;
	  _jacobianOplusXi ( 0,4 ) = 0;
	  _jacobianOplusXi ( 0,5 ) = x/z_2 * camera_->fx_;

	  _jacobianOplusXi ( 1,0 ) = ( 1+y*y/z_2 ) *camera_->fy_;
	  _jacobianOplusXi ( 1,1 ) = -x*y/z_2 *camera_->fy_;
	  _jacobianOplusXi ( 1,2 ) = -x/z *camera_->fy_;
	  _jacobianOplusXi ( 1,3 ) = 0;
	  _jacobianOplusXi ( 1,4 ) = -1./z *camera_->fy_;
	  _jacobianOplusXi ( 1,5 ) = y/z_2 *camera_->fy_;
      }


}
