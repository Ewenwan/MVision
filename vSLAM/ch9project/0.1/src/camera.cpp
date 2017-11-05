/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "myslam/camera.h"

namespace myslam
{

Camera::Camera()
{
}
//使用 Sophus::SE3 类表达相机位姿态(相机外参数)  即转换矩阵

/*******使用相机外参数********/
//世界坐标系(3维 ) 到 相机坐标系   
//输入 世界坐标系坐标 和 T_c_w 转换矩阵  
Vector3d Camera::world2camera ( const Vector3d& p_w, const SE3& T_c_w )
{
    return T_c_w*p_w;
}
//相机坐标系 到 世界坐标系           
//输入 相机坐标系坐标p_c 和 T_c_w 转换矩阵
Vector3d Camera::camera2world ( const Vector3d& p_c, const SE3& T_c_w )
{
    return T_c_w.inverse() *p_c;
}

/*******使用相机内参数*******/
 //相机坐标系 到 图像像素坐标系
Vector2d Camera::camera2pixel ( const Vector3d& p_c )
{
    return Vector2d (
        fx_ * p_c ( 0,0 ) / p_c ( 2,0 ) + cx_,
        fy_ * p_c ( 1,0 ) / p_c ( 2,0 ) + cy_
    );
}
//图像像素坐标系(两维) 到 相机坐标系    使用相机内参数
Vector3d Camera::pixel2camera ( const Vector2d& p_p, double depth )
{
    return Vector3d (
        ( p_p ( 0,0 )-cx_ ) *depth/fx_,
        ( p_p ( 1,0 )-cy_ ) *depth/fy_,
        depth
    );
}

//世界坐标系(三维) 到 图像像素坐标系 world2camera2pixel 
Vector2d Camera::world2pixel ( const Vector3d& p_w, const SE3& T_c_w )
{
    return camera2pixel ( world2camera ( p_w, T_c_w ) );
}

//图像像素坐标系(两维)   到 世界坐标系 pixel2camera2world 
Vector3d Camera::pixel2world ( const Vector2d& p_p, const SE3& T_c_w, double depth )
{
    return camera2world ( pixel2camera ( p_p, depth ), T_c_w );
}


}
