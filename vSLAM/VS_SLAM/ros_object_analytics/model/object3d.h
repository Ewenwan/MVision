/*
 * Copyright (c) 2017 Intel Corporation
 *
 */

#ifndef OBJECT_ANALYTICS_NODELET_MODEL_OBJECT3D_H
#define OBJECT_ANALYTICS_NODELET_MODEL_OBJECT3D_H

#define PCL_NO_PRECOMPILE
#include <vector>

#include <geometry_msgs/Point32.h>
#include <object_analytics_msgs/ObjectInBox3D.h>
#include <sensor_msgs/RegionOfInterest.h>

#include <pcl/common/projection_matrix.h>
#include <pcl/point_types.h>

namespace object_analytics_nodelet
{
namespace model
{
using PointT = pcl::PointXYZRGBA;
using PointCloudT = pcl::PointCloud<PointT>;

/** @class Object3D
 * @brief Wrapper of object_analytics_msgs::ObjectInBox3D.
 *
 * There are two scenarios of using this class. One is in SegmenterNodelet, this class is used to hold the result of 3d
 * segmentation and calculate minimum and maximum point in 3d space. Another one is in MergerNodelet, it's build based
 * on segmentation result.
 */
class Object3D
{
public:
  /**
   * @brief Construct a 3D object based on PointCloud segmentation result.
   *
   * Use this constructor to build Object3D objects when 3D segmentation is done.
   *
   * @param[in] cloud       PointCloud got from RGB-D sensor
   * @param[in] indices     Indices vector, each is the indices of one segmentation object
   */
  Object3D(const PointCloudT::ConstPtr& cloud, const std::vector<int>& indices);

  /**
   * @brief Construct a 3D object based on results published by segmenter.
   *
   * Use this constructor in merger to build Object3D object.
   *
   * @param[in] object3d    Result published by segmenter
   */
  explicit Object3D(const object_analytics_msgs::ObjectInBox3D& object3d);

  /** Default destructor */
  ~Object3D() = default;

  /**
   * Inline method. Get the region of interest in image space.
   *
   * @return Roi in image space
   */
  inline sensor_msgs::RegionOfInterest getRoi() const
  {
    return roi_;// 点云团对应的2d边框
  }

  /**
   * Inline method. Get the minimum x, y and z of underlying object in 3d space.
   *
   * @return 3D point contains minimum x, y and z
   */
  inline geometry_msgs::Point32 getMin() const
  {
    return min_; // 点云团 中 最小值3d点 
  }

  /**
   * Inline method. Get the maximum x, y and z of underlying object in 3d space.
   *
   * @return 3D point contains maximum x, y and z
   */
  inline geometry_msgs::Point32 getMax() const
  {
    return max_; // 点云团 中 最大值3d点(用于3d(长方体))
  }

  /**
   * Overload operator << to dump information of underlying information.
   *
   * @param[in,out] os    Standard output stream
   * @param[in]     obj   Object to be dumped
   *
   * @return Standard output stream
   */
  friend std::ostream& operator<<(std::ostream& os, const Object3D& obj);

private:
  sensor_msgs::RegionOfInterest roi_;// 2点云团对应的图像的 roi
  geometry_msgs::Point32 min_;// 三个坐标轴 最小的三个量(用于3d(长方体))
  geometry_msgs::Point32 max_;// 三个坐标轴 最大的三个量
};

using Object3DPtr = std::shared_ptr<Object3D>;
using Object3DConstPtr = std::shared_ptr<Object3D const>;
}  // namespace model
}  // namespace object_analytics_nodelet
#endif  // OBJECT_ANALYTICS_NODELET_MODEL_OBJECT3D_H
