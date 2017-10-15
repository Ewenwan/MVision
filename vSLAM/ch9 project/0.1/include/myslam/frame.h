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

#ifndef FRAME_H //防止头文件重复引用
#define FRAME_H//宏定义

#include "myslam/common_include.h"
#include "myslam/camera.h"
// 视频图像  帧  类
namespace myslam //命令空间下 防止定义的出其他库里的同名函数
{
    
// forward declare 
class MapPoint;// 使用 特征点/路标点 类
class Frame       // 视频图像  帧  类
{
public://
    typedef std::shared_ptr<Frame> Ptr;//共享指针 把 智能指针定义成 Frame的指针类型 以后参数传递时  使用Frame::Ptr 类型就可以了
    unsigned long                  id_;              // id of this frame        标号 编号
    double                         time_stamp_;  // when it is recorded  时间戳 记录
    SE3                            T_c_w_;               // transform from world to camera    w2c坐标变换
    Camera::Ptr                    camera_;     // Pinhole RGBD Camera model          使用相机模型 Camera::Ptr
    Mat                            color_, depth_; // color and depth image                     彩色图 深度图
    
public: // data members  数据成员设置为公有 可以改成私有 private
    Frame();
    Frame( long id, double time_stamp=0, SE3 T_c_w=SE3(), Camera::Ptr camera=nullptr, Mat color=Mat(), Mat depth=Mat() );
    ~Frame();
    
    // factory function
    static Frame::Ptr createFrame(); 
    
    // find the depth in depth map
    double findDepth( const cv::KeyPoint& kp );
    
    // Get Camera Center
    Vector3d getCamCenter() const;
    
    // check if a point is in this frame 
    bool isInFrame( const Vector3d& pt_world );
};

}

#endif // FRAME_H
