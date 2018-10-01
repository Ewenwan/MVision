/*
 */

#ifndef OBJECT_ANALYTICS_NODELET_MODEL_OBJECT_UTILS_H
#define OBJECT_ANALYTICS_NODELET_MODEL_OBJECT_UTILS_H

#define PCL_NO_PRECOMPILE
#include <vector>
#include <utility>

#include <opencv2/core/types.hpp>

#include <pcl/point_types.h>

#include <object_msgs/ObjectsInBoxes.h>
#include <object_analytics_msgs/ObjectInBox3D.h>
#include <object_analytics_msgs/ObjectsInBoxes3D.h>

#include "object_analytics_nodelet/model/object2d.h"
#include "object_analytics_nodelet/model/object3d.h"

// PCL新定义点类型   3d点坐标 + 2d的像素坐标值 3d-2d点对
struct PointXYZPixel
{
  PCL_ADD_POINT4D;// 
  uint32_t pixel_x;// 像素值
  uint32_t pixel_y;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;  // NOLINT
// 注册点类型
POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZPixel,                // xyz + pixel x, y as fields
                                  (float, x, x)                 // field x
                                  (float, y, y)                 // field y
                                  (float, z, z)                 // field z
                                  (uint32_t, pixel_x, pixel_x)  // field pixel x
                                  (uint32_t, pixel_y, pixel_y)  // field pixel y
                                  )

namespace object_analytics_nodelet
{
using object_msgs::ObjectsInBoxes;// 2d物体边框数组 ---- roi数组
using object_analytics_msgs::ObjectsInBoxes3D;// 3d变空数组(2d roi+ point3d_max min)
using object_analytics_nodelet::model::Object2D;// ROI边框 + 物体名
using object_analytics_nodelet::model::Object3D;// ROI边框 + point3d_max min
using Relation = std::pair<Object2D, Object3D>;// 2d框 和 3d点云团 配对pair关系
using RelationVector = std::vector<Relation>;// 配对关系 数组
using Object2DVector = std::vector<Object2D>;// 2d框数组
using Object3DVector = std::vector<Object3D>;// 3d点云团数组

namespace model
{
/** @class ObjectUtils
 * Utilities for handling 2d and 3d objects.
 */
class ObjectUtils
{
public:
  /**
   * Convert 2d object of ObjectInBox format into Object2D and push into vector.
   *
   * @param[in]  objects_in_boxes2d List of 2d detection result
   * @param[out] objects2d          List of 2d wrapper
   */
  static void fill2DObjects(const ObjectsInBoxes::ConstPtr& objects_in_boxes2d, Object2DVector& objects2d);

  /**
   * Convert 3d object of ObjectInBox format into Object3D and push into vector.
   *
   * @param[in]  objects_in_boxes3d List of 3d segmentation result
   * @param[out] objects3d          List of 3d wrapper
   */
  static void fill3DObjects(const ObjectsInBoxes3D::ConstPtr& objects_in_boxes3d, Object3DVector& objects3d);

  /**
   * Find the relationships between 2d object and 3d object according to who share the maximum intersetction area.
   *
   * @param[in]  objects2d          List of 3d segmentation result
   * @param[in]  objects3d          List of 3d wrapper
   * @param[out] relations          Pair list of 2d and 3d object which are represet the same object
   */
  static void findMaxIntersectionRelationships(const Object2DVector& objects2d, Object3DVector& objects3d,
                                               RelationVector& relations);

  /**
   * In points of given point cloud, find the minimum and maximum point in x filed.
   *
   * @param[in]  point_cloud        Point cloud
   * @param[out] x_min              Minimum point in x
   * @param[out] x_max              Maximum point in x
   */
  static void getMinMaxPointsInX(const pcl::PointCloud<PointXYZPixel>::ConstPtr& point_cloud, PointXYZPixel& x_min,
                                 PointXYZPixel& x_max);

  /**
   * In points of given point cloud, find the minimum and maximum point in y filed.
   *
   * @param[in]  point_cloud        Point cloud
   * @param[out] y_min              Minimum point in y
   * @param[out] y_max              Maximum point in y
   */
  static void getMinMaxPointsInY(const pcl::PointCloud<PointXYZPixel>::ConstPtr& point_cloud, PointXYZPixel& y_min,
                                 PointXYZPixel& y_max);

  /**
   * In points of given point cloud, find the minimum and maximum point in z filed.
   *
   * @param[in]  point_cloud        Point cloud
   * @param[out] z_min              Minimum point in z
   * @param[out] z_max              Maximum point in z
   */
  static void getMinMaxPointsInZ(const pcl::PointCloud<PointXYZPixel>::ConstPtr& point_cloud, PointXYZPixel& z_min,
                                 PointXYZPixel& z_max);

  /**
   * In points of given point cloud, find projected ROI in terms of pixel.
   *
   * @param[in]  point_cloud        Point cloud
   * @param[out] roi                Projected ROI
   */
  static void getProjectedROI(const pcl::PointCloud<PointXYZPixel>::ConstPtr& point_cloud,
                              sensor_msgs::RegionOfInterest& roi);

  /**
   * @brief Calculate the match rate of two rectangles.
   *
   * The matching algorithm is based on ROI overlapping rate and ROI center deviation,
   * the rect with the most overlapping and the least center deviation will be returned.
   *
   * @param[in] r1                  One of the two rectangles.
   * @param[in] r2                  The other one of the two rectangles.
   */
  static double getMatch(const cv::Rect2d& r1, const cv::Rect2d& r2);

  /**
   * @brief Extract the indices of a given point cloud as a new point cloud
   *             XYZRGBA 点+颜色 点云  拷贝到 XYZ+像素点坐标 点云
   * @param[in] original            The input point cloud dataset.
   * @param[in] indices             The vector of indices representing the points to be copied from original.
   * @param[out] dest               The resultant output point cloud dataset.
   */
  static void copyPointCloud(const PointCloudT::ConstPtr& original, const std::vector<int>& indices,
                             pcl::PointCloud<PointXYZPixel>::Ptr& dest);
};
}  // namespace model
}  // namespace object_analytics_nodelet
#endif  // OBJECT_ANALYTICS_NODELET_MODEL_OBJECT_UTILS_H
