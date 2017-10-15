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
/*管理 特征点/路标 保存所有的特征点/路标 和关键帧*/
#ifndef MAP_H//防止头文件重复引用
#define MAP_H//宏定义

#include "myslam/common_include.h"
#include "myslam/frame.h"
#include "myslam/mappoint.h"//特征点/路标

namespace myslam//命令空间下 防止定义的出其他库里的同名函数
{
class Map//类
{
public:
    typedef shared_ptr<Map> Ptr;//共享指针 把 智能指针定义成 Map的指针类型 以后参数传递时  使用Map::Ptr 类型就可以了
    unordered_map<unsigned long, MapPoint::Ptr >  map_points_;        // all landmarks  所有的路标/特征点
    unordered_map<unsigned long, Frame::Ptr >     keyframes_;            // all key-frames  所有的关键帧

    Map() {}
    //需要随机访问  使用散列(Hash) 来存储
    void insertKeyFrame( Frame::Ptr frame );              //保存/插入 关键帧
    void insertMapPoint( MapPoint::Ptr map_point ); //保存/插入 路标/特征点
};
}

#endif // MAP_H
