/*
 * Copyright (c) 2017 Intel Corporation
 */

#include <algorithm>
#include <vector>

#include <ros/console.h>

#include "object_analytics_nodelet/model/object3d.h"
#include "object_analytics_nodelet/model/object_utils.h"

namespace object_analytics_nodelet
{
namespace model
{
  
  
// 从XYZRGBA点云中 获取 点云团对应的2d边框 以及/* 点云团 中 最小值3d点、最大值3d点(用于3d(长方体))
// using PointT = pcl::PointXYZRGBA;
//using PointCloudT = pcl::PointCloud<PointT>; // XYZRGBA 点+颜色 点云
Object3D::Object3D(const PointCloudT::ConstPtr& cloud, const std::vector<int>& indices)
{
  pcl::PointCloud<PointXYZPixel>::Ptr seg(new pcl::PointCloud<PointXYZPixel>);
  
  ObjectUtils::copyPointCloud(cloud, indices, seg);// 由点云 indices 除以 / 取余 图像宽度 得到像素坐标
  

  PointXYZPixel x_min_point, x_max_point;
  ObjectUtils::getMinMaxPointsInX(seg, x_min_point, x_max_point);// 3d点中, x坐标值的 最大最小值
  min_.x = x_min_point.x;
  max_.x = x_max_point.x;

  PointXYZPixel y_min_point, y_max_point;
  ObjectUtils::getMinMaxPointsInY(seg, y_min_point, y_max_point);// 3d点中, y坐标值的 最大最小值
  min_.y = y_min_point.y;
  max_.y = y_max_point.y;

  PointXYZPixel z_min_point, z_max_point;
  ObjectUtils::getMinMaxPointsInZ(seg, z_min_point, z_max_point);// 3d点中, z坐标值的 最大最小值
  min_.z = z_min_point.z;
  max_.z = z_max_point.z;

  ObjectUtils::getProjectedROI(seg, this->roi_);// 从 3d+2d点云团里获取 对应2d roi边框======
 //  2d 像素点集 获取 对应的roi边框 min_x, min_y, max_x-min_x, max_y-min_y================
}

Object3D::Object3D(const object_analytics_msgs::ObjectInBox3D& object3d)
  : roi_(object3d.roi), min_(object3d.min), max_(object3d.max)// 点云团对应的2d边框: 最小值3d点、最大值3d点
{
}

std::ostream& operator<<(std::ostream& os, const Object3D& obj)
{
  os << "Object3D[min=" << obj.min_.x << "," << obj.min_.y << "," << obj.min_.z;
  os << " max=" << obj.max_.x << "," << obj.max_.y << "," << obj.max_.z << ", roi=" << obj.roi_.x_offset << ","
     << obj.roi_.y_offset << "," << obj.roi_.width << "," << obj.roi_.height << "]";
  return os;
}
}  // namespace model
}  // namespace object_analytics_nodelet
