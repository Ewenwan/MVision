/*
 * 各种模型数据 结构
 */
#define PCL_NO_PRECOMPILE
#include <vector>

#include <ros/assert.h>// 断言
#include <ros/console.h>// 命令行

#include <pcl/common/io.h>//输入输出

#include "object_analytics_nodelet/model/object_utils.h"

namespace object_analytics_nodelet
{
namespace model
{
 
 // 返回2d物体 数组 
void ObjectUtils::fill2DObjects(const ObjectsInBoxes::ConstPtr& objects_in_boxes2d, Object2DVector& objects2d)
{
  for (auto item : objects_in_boxes2d->objects_vector)//2d物体roi 边框数组
  {
    Object2D object2d(item);
    objects2d.push_back(object2d);
  }
}
 // 返回3d物体 数组 
void ObjectUtils::fill3DObjects(const ObjectsInBoxes3D::ConstPtr& objects_in_boxes3d, Object3DVector& objects3d)
{
  for (auto item : objects_in_boxes3d->objects_in_boxes)// ObjectInBox3D 数组
  {
    Object3D object3d(item);
    objects3d.push_back(object3d);
  }
}

// 2d物体 和 3d物体 关系 ==============================================
// 遍历 每一个2d物体
//     遍历   每一个3d物体
//        计算 2d物体边框 和3d物体投影2d边框的相似度  两边框的匹配相似度 match = IOU * distance /  AvgSize
//        记录和 该2d边框最相似的 3d物体id
void ObjectUtils::findMaxIntersectionRelationships(const Object2DVector& objects2d, Object3DVector& objects3d,
                                                   RelationVector& relations)
{
  for (auto obj2d : objects2d)// 每一个2d物体
  {
    Object3DVector::iterator max_it = objects3d.begin();// 3d物体id
    double max = 0;

    auto obj2d_roi = obj2d.getRoi();// 2d物体 2d 边框roi
    const cv::Rect2d rect2d(obj2d_roi.x_offset, obj2d_roi.y_offset, obj2d_roi.width, obj2d_roi.height);// 边框
    
    for (Object3DVector::iterator it = max_it; it != objects3d.end(); ++it)// 每一个3d物体
    {
      auto obj3d_roi = it->getRoi();// 3d物体 roi 3d点云投影到2d平面后的2d边框
      const cv::Rect2d rect3d(obj3d_roi.x_offset, obj3d_roi.y_offset, obj3d_roi.width, obj3d_roi.height);// 3d点云投影到2d平面后的2d边框
      auto area = getMatch(rect2d, rect3d);// 两边框 的 匹配相似度   IOU * distance /  AvgSize

      if (area < max)
      {
        continue;
      }

      max = area;
      max_it = it;  // 为每一个 2d边框 寻找 一个 匹配度最高的 3d物体=======================================
    }

    if (max <= 0)
    {
      ROS_DEBUG_STREAM("Cannot find correlated 3D object for " << obj2d);
      continue;
    }

    relations.push_back(Relation(obj2d, *max_it));// 3d物体 和 2d物体 配对关系
    objects3d.erase(max_it);// 删除已经匹配的3d点云物体
  }
}

// 点云集合 x值最大值与最小值 =====================
void ObjectUtils::getMinMaxPointsInX(const pcl::PointCloud<PointXYZPixel>::ConstPtr& point_cloud, PointXYZPixel& x_min,
                                     PointXYZPixel& x_max)
{
  auto cmp_x = [](PointXYZPixel const& l, PointXYZPixel const& r) { return l.x < r.x; };// 按x值域大小 的 比较函数
  auto minmax_x = std::minmax_element(point_cloud->begin(), point_cloud->end(), cmp_x);//std库 获取最大最小值
  x_min = *(minmax_x.first);       // 最小值
  x_max = *(minmax_x.second);// 最大值
}
// 点云集合 y值最大值与最小值 =====================
void ObjectUtils::getMinMaxPointsInY(const pcl::PointCloud<PointXYZPixel>::ConstPtr& point_cloud, PointXYZPixel& y_min,
                                     PointXYZPixel& y_max)
{
  auto cmp_y = [](PointXYZPixel const& l, PointXYZPixel const& r) { return l.y < r.y; };// 按y值域大小 的 比较函数
  auto minmax_y = std::minmax_element(point_cloud->begin(), point_cloud->end(), cmp_y);
  y_min = *(minmax_y.first);
  y_max = *(minmax_y.second);
}
// 点云集合 z值最大值与最小值 =====================
void ObjectUtils::getMinMaxPointsInZ(const pcl::PointCloud<PointXYZPixel>::ConstPtr& point_cloud, PointXYZPixel& z_min,
                                     PointXYZPixel& z_max)
{
  auto cmp_z = [](PointXYZPixel const& l, PointXYZPixel const& r) { return l.z < r.z; };
  auto minmax_z = std::minmax_element(point_cloud->begin(), point_cloud->end(), cmp_z);
  z_min = *(minmax_z.first);
  z_max = *(minmax_z.second);
}

//  2d 像素点集 获取 对应的roi边框 min_x, min_y, max_x-min_x, max_y-min_y================
//  从 3d+2d点云团里获取 2droi边框       ======
// PointXYZPixel = 3d点 + 2d 像素点坐标======
// 计算 3d物体点云集合 投影在 相机平面上的 2d框 ROI=====
void ObjectUtils::getProjectedROI(const pcl::PointCloud<PointXYZPixel>::ConstPtr& point_cloud,
                                  sensor_msgs::RegionOfInterest& roi)
{
  auto cmp_x = [](PointXYZPixel const& l, PointXYZPixel const& r) { return l.pixel_x < r.pixel_x; };
  
  auto minmax_x = std::minmax_element(point_cloud->begin(), point_cloud->end(), cmp_x);// 点云对应像素点坐标， pixel_x的最大最小值
  
  roi.x_offset = minmax_x.first->pixel_x;       // x_offset 框的左上角点, x坐标最小值
  auto max_x = minmax_x.second->pixel_x;// x坐标最大值
  ROS_ASSERT(roi.x_offset >= 0);
  ROS_ASSERT(max_x >= roi.x_offset);
  roi.width = max_x - roi.x_offset;// 2d框 宽度   max_x - min_x

  auto cmp_y = [](PointXYZPixel const& l, PointXYZPixel const& r) { return l.pixel_y < r.pixel_y; };
  auto minmax_y = std::minmax_element(point_cloud->begin(), point_cloud->end(), cmp_y);// 点云对应像素点坐标， pixel_y的最大最小值
  roi.y_offset = minmax_y.first->pixel_y; // y_offset 框的左上角点, y坐标最小值
  auto max_y = minmax_y.second->pixel_y;// y坐标最大值
  ROS_ASSERT(roi.y_offset >= 0);
  ROS_ASSERT(max_y >= roi.y_offset);
  roi.height = max_y - roi.y_offset;                //  2d框 高度   max_y - min_x
}

// 两边框 的 匹配相似度   IOU * distance /  AvgSize===============
double ObjectUtils::getMatch(const cv::Rect2d& r1, const cv::Rect2d& r2)
{
  cv::Rect2i ir1(r1), ir2(r2);
  /* calculate center of rectangle #1  边框中心点 */
  cv::Point2i c1(ir1.x + (ir1.width >> 1), ir1.y + (ir1.height >> 1));// 边框 中心点1
  /* calculate center of rectangle #2  边框中心点 */
  cv::Point2i c2(ir2.x + (ir2.width >> 1), ir2.y + (ir2.height >> 1));// 边框 中心点2

  double a1 = ir1.area(), a2 = ir2.area(), a0 = (ir1 & ir2).area();// opencv 的 矩形支持 &并集 运算符
  /* calculate the overlap rate*/
  double overlap = a0 / (a1 + a2 - a0);// IOU 交并比
  /* calculate the deviation between centers #1 and #2*/
  double deviate = sqrt(powf((c1.x - c2.x), 2) + powf((c1.y - c2.y), 2));// 边框中心点 距离 距离近相似
  /* calculate the length of diagonal for the rectangle in average size*/
  // 使用 平均尺寸  进行匹配度 加权 =====================================================
  double len_diag = sqrt(powf(((ir1.width + ir2.width) >> 1), 2) + powf(((ir1.height + ir2.height) >> 1), 2));

  /* calculate the match rate. The more overlap, the more matching. Contrary, the more deviation, the less matching*/
  return overlap * len_diag / deviate;
}

// XYZRGBA 点+颜色 点云  拷贝到 XYZ+像素点坐标 点云
void ObjectUtils::copyPointCloud(const PointCloudT::ConstPtr& original, const std::vector<int>& indices,
                                 pcl::PointCloud<PointXYZPixel>::Ptr& dest)
{
  pcl::copyPointCloud(*original, indices, *dest);// 拷贝 3d点坐标
  uint32_t width = original->width;// 相当于图像宽度
  for (uint32_t i = 0; i < indices.size(); i++)
  {
    dest->points[i].pixel_x = indices[i] % width;// 列坐标
    dest->points[i].pixel_y = indices[i] / width;// 行坐标
  }
}

}  // namespace model
}  // namespace object_analytics_nodelet
