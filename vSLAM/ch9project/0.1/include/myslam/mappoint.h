/*
 * myslam::MapPoint   路标点   标号+位置+方向
 */
/*特征点/路标*/
#ifndef MAPPOINT_H//防止头文件重复引用
#define MAPPOINT_H//宏定义

namespace myslam//命令空间下 防止定义的出其他库里的同名函数
{
    
class Frame;
class MapPoint//类
{
	public:// 公有
	  // 变量
	    typedef shared_ptr<MapPoint> Ptr;//共享指针 把 智能指针定义成 MapPoint的指针类型 以后参数传递时  使用MapPoint::Ptr 类型就可以了
	    unsigned long      id_; // ID 标号
	    Vector3d    pos_;       // Position in world 世界坐标系下的位置坐标 (x y z)
	    Vector3d    norm_;    // Normal of viewing direction 方向
	    Mat         descriptor_; // Descriptor for matching        描述子
	    int         observed_times_;    // being observed by feature matching algo. 特征匹配时的时间
	    int         correct_times_;       // being an inliner in pose estimation
	    
	    MapPoint();//【1】默认构造函数
	    // 【2】自定义构造函数  标号  坐标   位姿方向
	    MapPoint( long id, Vector3d position, Vector3d norm );
	    
	    // 【3】factory function 工厂函数
	    static MapPoint::Ptr createMapPoint();
	};
}

#endif // MAPPOINT_H
