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
/*视觉里程记*/
#ifndef VISUALODOMETRY_H//防止头文件重复引用
#define VISUALODOMETRY_H//宏定义

#include "myslam/common_include.h"//常用的头文件 放在一起 化繁为简
#include "myslam/map.h"

#include <opencv2/features2d/features2d.hpp>

namespace myslam //命令空间下 防止定义的出其他库里的同名函数
{
class VisualOdometry//视觉里程计类
{
public:
    typedef shared_ptr<VisualOdometry> Ptr;//共享指针 把 智能指针定义成 VisualOdometry的指针类型 以后参数传递时  使用VisualOdometry::Ptr 类型就可以了
    enum VOState {//枚举 视觉里程计 状态
        INITIALIZING=-1,//初始化标志
        OK=0,//估计成功标志
        LOST//跟丢标志
    };
    
    VOState     state_;      // 状态 current VO status
    Map::Ptr    map_;       // 地图 所有的帧和特征点/地标 map with all frames and map points
    Frame::Ptr  ref_;        // 上一次帧 参考帧 reference frame 
    Frame::Ptr  curr_;      // 当前帧                 current frame 
    
    cv::Ptr<cv::ORB> orb_;  // orb特征检测  orb detector and computer 
    vector<cv::Point3f>     pts_3d_ref_;          // 参考帧 中的点         3d points in reference frame 
    vector<cv::KeyPoint>    keypoints_curr_;// 当前帧 中的 关键点 keypoints in current frame
    Mat                     descriptors_curr_;          // 当前帧中的描述子 descriptor in current frame 
    Mat                     descriptors_ref_;            // 参考帧中的描述子 descriptor in reference frame 
    vector<cv::DMatch>      feature_matches_;//特征点的匹配 特征点对于描述子 之间的 字符串距离
    
    SE3 T_c_r_estimated_;  // 估计的坐标转换   the estimated pose of current frame 
    int num_inliers_;          // 特征数量  number of inlier features in icp
    int num_lost_;              // 丢失次数  number of lost times
    
    // 参数变量 parameters 
    int num_of_features_;   // 每一对帧提取的特征对数量  number of features
    double scale_factor_;    // 图像金字塔尺度  scale in image pyramid
    int level_pyramid_;        // 图像金字塔层级数  number of pyramid levels
    float match_ratio_;        // 选择特征点匹配的阈值 ratio for selecting  good matches
    int max_num_lost_;       // 连续丢失的最大次数 max number of continuous lost times
    int min_inliers_;             // 最少内点数量 minimum inliers
    
    double key_frame_min_rot;     // 最小的旋转 minimal rotation of two key-frames
    double key_frame_min_trans; // 最小平移     minimal translation of two key-frames
    
public: // functions 公有函数  可以改成  私有private  保护protected   函数
    VisualOdometry();//VisualOdometry::VisualOdometry() ://视觉里程计类
    ~VisualOdometry();
    
    bool addFrame( Frame::Ptr frame );      // 添加新的一帧 add a new frame 
    
protected:  
    // inner operation  内部函数
    void extractKeyPoints();       //提取关键点
    void computeDescriptors();//计算描述子
    void featureMatching();       //特征匹配
    void poseEstimationPnP();  //位姿估计
    void setRef3DPoints();	   //设置参考点 三维坐标
    
    void addKeyFrame();//添加关键帧
    bool checkEstimatedPose();//核对估计的位姿
    bool checkKeyFrame();        //检查是否为关键帧
    
};
}

#endif // VISUALODOMETRY_H