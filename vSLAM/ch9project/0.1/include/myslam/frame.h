/*
 *myslam::Frame 
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
    public:// 公有变量数据   data members  数据成员设置为公有 可以改成私有 private
	typedef std::shared_ptr<Frame> Ptr;//共享指针 把 智能指针定义成 Frame的指针类型 以后参数传递时  使用Frame::Ptr 类型就可以了
	unsigned long         id_;         // id of this frame        标号 编号
	double                time_stamp_; // when it is recorded  时间戳 记录
	SE3                   T_c_w_;      // transform from world to camera  世界坐标系到相机坐标系 坐标变换
	Camera::Ptr           amera_;      // Pinhole RGBD Camera model       使用相机模型类  Camera::Ptr 坐标转换
	Mat                    color_, depth_; // color and depth image       彩色图 深度图
	
    public: //  公有函数
	Frame();// 默认 
	Frame( long id, double time_stamp=0, SE3 T_c_w=SE3(), Camera::Ptr camera=nullptr, Mat color=Mat(), Mat depth=Mat() );
	~Frame();//析构函数   退出时  清除函数使用的内存
	
	// factory function 工厂函数
	static Frame::Ptr createFrame();//
	
	// find the depth in depth map
	double findDepth( const cv::KeyPoint& kp );
	
	// Get Camera Center
	// 世界坐标 到 相机坐标  平移矩阵  相机中心坐标
	Vector3d getCamCenter() const;
	
	// check if a point is in this frame 
	// 世界坐标系点是否在 该帧内     相应的 像极坐标系下 z>0  对应的图像像素坐标值在 0到尺寸范围内
	bool isInFrame( const Vector3d& pt_world );
  };

}

#endif // FRAME_H
