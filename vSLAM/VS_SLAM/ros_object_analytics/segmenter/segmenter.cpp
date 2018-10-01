/*
 3d 点云分割 算法
 步骤1:  点云滤波
 步骤2:  平面模型 Ransac 分割平面
 步骤3:  对平面上的点云，进行 欧氏距离聚类分割点云成不同点云集合

 */
#define PCL_NO_PRECOMPILE
#include <memory>
#include <vector>

#include <pcl_conversions/pcl_conversions.h>

#include <object_analytics_msgs/ObjectsInBoxes3D.h>
#include <object_analytics_msgs/ObjectInBox3D.h>

#include "object_analytics_nodelet/segmenter/organized_multi_plane_segmenter.h"
#include "object_analytics_nodelet/segmenter/segmenter.h"

namespace object_analytics_nodelet
{
namespace segmenter
{
using pcl::fromROSMsg;
using pcl::Label;
using pcl::Normal;// 点云平面法线
using pcl::PointIndices;// 索引
using pcl::IndicesPtr;
using pcl::copyPointCloud;
using object_analytics_nodelet::model::Object3D;

Segmenter::Segmenter(std::unique_ptr<AlgorithmProvider> provider) : provider_(std::move(provider))
{
}

void Segmenter::segment(const sensor_msgs::PointCloud2::ConstPtr& points, boost::shared_ptr<ObjectsInBoxes3D>& msg)
{
  PointCloudT::Ptr pointcloud(new PointCloudT);
  getPclPointCloud(points, *pointcloud);// 转换成 pcl点云类型

  std::vector<Object3D> objects;// 3d物体 数组
  doSegment(pointcloud, objects);// 分割点云得到多个 点云集合物体

  composeResult(objects, msg);
}

void Segmenter::getPclPointCloud(const sensor_msgs::PointCloud2::ConstPtr& points, PointCloudT& pcl_cloud)
{
  fromROSMsg<PointT>(*points, pcl_cloud);
}

void Segmenter::doSegment(const PointCloudT::ConstPtr& cloud, std::vector<Object3D>& objects)
{
  std::vector<PointIndices> cluster_indices;
  PointCloudT::Ptr cloud_segment(new PointCloudT);
  std::shared_ptr<Algorithm> seg = provider_->get();
  seg->segment(cloud, cloud_segment, cluster_indices);

  for (auto& indices : cluster_indices)
  {
    try
    {
      Object3D object3d(cloud_segment, indices.indices);
      objects.push_back(object3d);
    }
    catch (std::exception& e)
    {
      ROS_ERROR_STREAM(e.what());
    }
  }

  ROS_DEBUG_STREAM("get " << objects.size() << " objects from segmentation");
}

void Segmenter::composeResult(const std::vector<Object3D>& objects, boost::shared_ptr<ObjectsInBoxes3D>& msg)
{
  for (auto& obj : objects)
  {
    object_analytics_msgs::ObjectInBox3D oib3;
    oib3.min = obj.getMin();
    oib3.max = obj.getMax();
    oib3.roi = obj.getRoi();
    msg->objects_in_boxes.push_back(oib3);
  }

  ROS_DEBUG_STREAM("segmenter publish message with " << objects.size() << " objects");
}
}  // namespace segmenter
}  // namespace object_analytics_nodelet
