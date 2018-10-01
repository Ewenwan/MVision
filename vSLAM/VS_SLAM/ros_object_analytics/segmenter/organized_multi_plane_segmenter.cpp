/*
 */

#include <vector>

#include <pcl/common/time.h>
#include <pcl/filters/impl/conditional_removal.hpp>
#include <pcl/filters/impl/filter.hpp>
#include <pcl/search/impl/organized.hpp>
#include <pcl/segmentation/impl/organized_connected_component_segmentation.hpp>

#include "object_analytics_nodelet/const.h"
#include "object_analytics_nodelet/segmenter/organized_multi_plane_segmenter.h"

namespace object_analytics_nodelet
{
namespace segmenter
{
using pcl::Label;
using pcl::Normal;
using pcl::PointCloud;
using pcl::PointIndices;
using pcl::PlanarRegion;

OrganizedMultiPlaneSegmenter::OrganizedMultiPlaneSegmenter(ros::NodeHandle& nh)
  : plane_comparator_(new pcl::PlaneCoefficientComparator<PointT, Normal>)
  , euclidean_comparator_(new pcl::EuclideanPlaneCoefficientComparator<PointT, Normal>)
  , rgb_comparator_(new pcl::RGBPlaneCoefficientComparator<PointT, Normal>)
  , edge_aware_comparator_(new pcl::EdgeAwarePlaneComparator<PointT, Normal>)
  , euclidean_cluster_comparator_(new pcl::EuclideanClusterComparator<PointT, Normal, Label>)
  , conf_(nh, "OrganizedMultiPlaneSegmenter")
{
}

void OrganizedMultiPlaneSegmenter::segment(const PointCloudT::ConstPtr& cloud, PointCloudT::Ptr& cloud_segment,
                                           std::vector<PointIndices>& cluster_indices)
{
  double start = pcl::getTime();
  ROS_DEBUG_STREAM("Total original point size = " << cloud->size());

  pcl::copyPointCloud(*cloud, *cloud_segment);  // cloud_segment is same as cloud for this algorithm
  applyConfig();

  
  // 估计点云法线=================================================
  PointCloud<Normal>::Ptr normal_cloud(new PointCloud<Normal>);
  estimateNormal(cloud, normal_cloud);

  std::vector<PlanarRegion<PointT>, Eigen::aligned_allocator<PlanarRegion<PointT>>> regions;
  PointCloud<Label>::Ptr labels(new PointCloud<Label>);
  std::vector<PointIndices> label_indices;
  
  // 分割平面
  segmentPlanes(cloud, normal_cloud, regions, labels, label_indices);
  
  // 分割点云
  segmentObjects(cloud, regions, labels, label_indices, cluster_indices);

  double end = pcl::getTime();
  ROS_DEBUG_STREAM("Segmentation : " << double(end - start));
}

void OrganizedMultiPlaneSegmenter::estimateNormal(const PointCloudT::ConstPtr& cloud,
                                                  PointCloud<Normal>::Ptr& normal_cloud)
{
  double start = pcl::getTime();

  normal_estimation_.setInputCloud(cloud);
  normal_estimation_.compute(*normal_cloud);

  float* distance_map = normal_estimation_.getDistanceMap();
  boost::shared_ptr<pcl::EdgeAwarePlaneComparator<PointT, Normal>> eapc =
      boost::dynamic_pointer_cast<pcl::EdgeAwarePlaneComparator<PointT, Normal>>(edge_aware_comparator_);
  eapc->setDistanceMap(distance_map);
  eapc->setDistanceThreshold(0.01f, false);

  double end = pcl::getTime();
  ROS_DEBUG_STREAM("Calc normal : " << double(end - start));
}

// 分割物体=======================
void OrganizedMultiPlaneSegmenter::segmentPlanes(
    const PointCloudT::ConstPtr& cloud, const pcl::PointCloud<Normal>::Ptr& normal_cloud,
    std::vector<PlanarRegion<PointT>, Eigen::aligned_allocator<PlanarRegion<PointT>>>& regions,
    pcl::PointCloud<Label>::Ptr labels, std::vector<PointIndices>& label_indices)
{
  double start = pcl::getTime();

  std::vector<pcl::ModelCoefficients> model_coefficients;// 平面模型系数
  std::vector<PointIndices> inlier_indices;
  std::vector<PointIndices> boundary_indices;
  plane_segmentation_.setInputNormals(normal_cloud);// 点云法线
  plane_segmentation_.setInputCloud(cloud);                   // 需要分割的点云
  plane_segmentation_.segmentAndRefine(regions, model_coefficients, inlier_indices, labels, label_indices,
                                       boundary_indices);// 进行分割并 精细调整

  double end = pcl::getTime();
  ROS_DEBUG_STREAM("Plane detection : " << double(end - start));
}

// 分割物体==================================
void OrganizedMultiPlaneSegmenter::segmentObjects(
    const PointCloudT::ConstPtr& cloud,
    std::vector<PlanarRegion<PointT>, Eigen::aligned_allocator<PlanarRegion<PointT>>>& regions,
    PointCloud<Label>::Ptr labels, std::vector<PointIndices>& label_indices, std::vector<PointIndices>& cluster_indices)
{
  double start = pcl::getTime();

  std::vector<bool> plane_labels;
  plane_labels.resize(label_indices.size(), false);
  for (size_t i = 0; i < label_indices.size(); i++)
  {
    if (label_indices[i].indices.size() > plane_minimum_points_)
    {
      plane_labels[i] = true;
    }
  }

  euclidean_cluster_comparator_->setInputCloud(cloud);
  euclidean_cluster_comparator_->setLabels(labels);
  euclidean_cluster_comparator_->setExcludeLabels(plane_labels);

  PointCloud<Label> euclidean_labels;
  pcl::OrganizedConnectedComponentSegmentation<PointT, Label> euclidean_segmentation(euclidean_cluster_comparator_);
  euclidean_segmentation.setInputCloud(cloud);
  euclidean_segmentation.segment(euclidean_labels, cluster_indices);

  auto func = [this](PointIndices indices) { return indices.indices.size() < this->object_minimum_points_; };
  cluster_indices.erase(std::remove_if(cluster_indices.begin(), cluster_indices.end(), func), cluster_indices.end());

  double end = pcl::getTime();
  ROS_DEBUG_STREAM("Cluster : " << double(end - start));
}

void OrganizedMultiPlaneSegmenter::applyConfig()
{
  OrganizedMultiPlaneSegmentationConfig conf = conf_.getConfig();

  plane_minimum_points_ = static_cast<size_t>(conf.plane_minimum_points);
  object_minimum_points_ = static_cast<size_t>(conf.object_minimum_points);

  normal_estimation_.setNormalEstimationMethod(normal_estimation_.SIMPLE_3D_GRADIENT);
  normal_estimation_.setNormalEstimationMethod(normal_estimation_.COVARIANCE_MATRIX);
  normal_estimation_.setMaxDepthChangeFactor(conf.normal_max_depth_change);
  normal_estimation_.setNormalSmoothingSize(conf.normal_smooth_size);

  euclidean_cluster_comparator_->setDistanceThreshold(conf.euclidean_distance_threshold, false);

  plane_segmentation_.setMinInliers(conf.min_plane_inliers);
  plane_segmentation_.setAngularThreshold(pcl::deg2rad(conf.normal_angle_threshold));
  plane_segmentation_.setDistanceThreshold(conf.normal_distance_threshold);

  if (conf.comparator == kPlaneCoefficientComparator)
  {
    plane_segmentation_.setComparator(plane_comparator_);
  }
  else if (conf.comparator == kEuclideanPlaneCoefficientComparator)
  {
    plane_segmentation_.setComparator(euclidean_comparator_);
  }
  else if (conf.comparator == kRGBPlaneCoefficientComparator)
  {
    plane_segmentation_.setComparator(rgb_comparator_);
  }
  else if (conf.comparator == kEdgeAwarePlaneComaprator)
  {
    plane_segmentation_.setComparator(edge_aware_comparator_);
  }
}
}  // namespace segmenter
}  // namespace object_analytics_nodelet
