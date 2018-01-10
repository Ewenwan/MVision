/*
 * myslam::MapPoint  路标点   标号+位置+方向
 *
 */

#include "myslam/common_include.h"
#include "myslam/mappoint.h"

namespace myslam//命令空间下 防止定义的出其他库里的同名函数
{
// 【1】默认构造函数
    MapPoint::MapPoint()
    : id_(-1), pos_(Vector3d(0,0,0)), norm_(Vector3d(0,0,0)), observed_times_(0), correct_times_(0)  {  }
// 【2】自定义构造函数    标号    坐标   位姿方向
    MapPoint::MapPoint ( long id, Vector3d position, Vector3d norm )
    : id_(id), pos_(position), norm_(norm), observed_times_(0), correct_times_(0) {  }
// 【3】factory function 工厂函数
    MapPoint::Ptr MapPoint::createMapPoint()
    {
	static long factory_id = 0;
	return MapPoint::Ptr( 
	    new MapPoint( factory_id++, Vector3d(0,0,0), Vector3d(0,0,0) )//标号+位置+方向
	);
    }

}
