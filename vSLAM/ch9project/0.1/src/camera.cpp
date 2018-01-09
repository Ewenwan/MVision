/*
 * 相机模型 内参数
 * 像素坐标系、相机坐标系 、世界坐标系坐标相互转换
 */

#include "myslam/camera.h"

namespace myslam
{

	Camera::Camera(){}//默认构造函数

	//使用 Sophus::SE3 类表达相机位姿态(相机外参数)  即转换矩阵

	/*******使用相机外参数********/
	//世界坐标系(3维 ) 到 相机坐标系   
	//输入 世界坐标系坐标 和 T_c_w 转换矩阵      p_c =   T_c_w*p_w;
	Vector3d Camera::world2camera ( const Vector3d& p_w, const SE3& T_c_w )
	{
	    return T_c_w*p_w;
	}
	//相机坐标系 到 世界坐标系           
	//输入 相机坐标系坐标p_c 和 T_c_w 转换矩阵   p_w = T_c_w.inverse() *p_c
	Vector3d Camera::camera2world ( const Vector3d& p_c, const SE3& T_c_w )
	{
	    return T_c_w.inverse() *p_c;
	}

	/*******使用相机内参数*******/
	  //  相机坐标系 到 图像像素坐标系
	  //  像素坐标 u = xx/zz * fx  + cx
	  //  像素坐标 v = yy/zz * fy  + cy 
	Vector2d Camera::camera2pixel ( const Vector3d& p_c )
	{
	    return Vector2d (
		fx_ * p_c ( 0,0 ) / p_c ( 2,0 ) + cx_,
		fy_ * p_c ( 1,0 ) / p_c ( 2,0 ) + cy_
	    );
	}
	//图像像素坐标系(两维) 到 相机坐标系    使用相机内参数
	// x = (u - c_x) * depth / fx
	// y = (v - c_y) * depth / fy
	//z  = depth  // 深度  默认为归一化像极坐标系平面  即 深度为 1
	Vector3d Camera::pixel2camera ( const Vector2d& p_p, double depth )
	{
	    return Vector3d (
		( p_p ( 0,0 )-cx_ ) *depth/fx_,
		( p_p ( 1,0 )-cy_ ) *depth/fy_,
		depth//深度 默认为归一化像极坐标系平面  即 深度为 1
	    );
	}

	//世界坐标系(三维) 到 图像像素坐标系 world 2 camera 2 pixel 
	//p_p = camera2pixel ( world2camera ( p_w, T_c_w ) );
	Vector2d Camera::world2pixel ( const Vector3d& p_w, const SE3& T_c_w )
	{
	    return camera2pixel ( world2camera ( p_w, T_c_w ) );
	}

	//图像像素坐标系(两维)   到 世界坐标系 pixel2camera2world 
	// p_w = camera2world ( pixel2camera ( p_p, depth ), T_c_w ); 
	Vector3d Camera::pixel2world ( const Vector2d& p_p, const SE3& T_c_w, double depth )
	{
	    return camera2world ( pixel2camera ( p_p, depth ), T_c_w );
	}

}
