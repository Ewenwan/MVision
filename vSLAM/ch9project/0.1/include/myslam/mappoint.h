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
/*特征点/路标*/
#ifndef MAPPOINT_H//防止头文件重复引用
#define MAPPOINT_H//宏定义

namespace myslam//命令空间下 防止定义的出其他库里的同名函数
{
    
class Frame;
class MapPoint//类
{
public:
    typedef shared_ptr<MapPoint> Ptr;//共享指针 把 智能指针定义成 MapPoint的指针类型 以后参数传递时  使用MapPoint::Ptr 类型就可以了
    unsigned long      id_; // ID 标号
    Vector3d    pos_;       // Position in world 世界坐标系下的位置坐标 (x y z)
    Vector3d    norm_;      // Normal of viewing direction 方向
    Mat         descriptor_; // Descriptor for matching        描述子
    int         observed_times_;    // being observed by feature matching algo. 特征匹配时的时间
    int         correct_times_;       // being an inliner in pose estimation
    
    MapPoint();
    MapPoint( long id, Vector3d position, Vector3d norm );
    
    // factory function
    static MapPoint::Ptr createMapPoint();
};
}

#endif // MAPPOINT_H
