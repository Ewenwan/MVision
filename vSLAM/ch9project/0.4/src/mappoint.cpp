/*
 * 地图点类  加入 所处帧  对赢二维图像点的特征描述子 用于 匹配
 */

#ifndef MAPPOINT_H
#define MAPPOINT_H

#include "myslam/common_include.h"

namespace myslam
{
    
class Frame;
class MapPoint
{
public:
    typedef shared_ptr<MapPoint> Ptr;
    unsigned long      id_;        // ID
    static unsigned long factory_id_;    // factory id
    bool        good_;      // wheter a good point 
    Vector3d    pos_;       // Position in world
    Vector3d    norm_;      // Normal of viewing direction 
    Mat         descriptor_; // Descriptor for matching 
    
    list<Frame*>    observed_frames_;   // key-frames that can observe this point 
    
    int         matched_times_;     // being an inliner in pose estimation
    int         visible_times_;     // being visible in current frame 
    
    MapPoint();
    MapPoint( 
        unsigned long id, 
        const Vector3d& position, //位置
        const Vector3d& norm, //方位
        Frame* frame=nullptr, //所属帧
        const Mat& descriptor=Mat() //加入特征描述子
    );
   
   //  inline 标识  常在 较短的函数前面  表示 在执行该函数时 不用新开辟内存空间  之间类似 宏定义 一样替换 节省时间
    inline cv::Point3f getPositionCV() const {
        return cv::Point3f( pos_(0,0), pos_(1,0), pos_(2,0) );//返回 opencv格式下的3维点
    }
    
    static MapPoint::Ptr createMapPoint();
    static MapPoint::Ptr createMapPoint( 
        const Vector3d& pos_world, 
        const Vector3d& norm_,
        const Mat& descriptor,
        Frame* frame );
};
}

#endif // MAPPOINT_H
