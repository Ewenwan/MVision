/*
 * Copyright (c) 2017 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef OBJECT_ANALYTICS_NODELET_SEGMENTER_ORGANIZED_MULTI_PLANE_SEGMENTER_H
#define OBJECT_ANALYTICS_NODELET_SEGMENTER_ORGANIZED_MULTI_PLANE_SEGMENTER_H

#define PCL_NO_PRECOMPILE

#include <vector>

#include <pcl/features/impl/integral_image_normal.hpp>
#include <pcl/features/impl/normal_3d.hpp>
#include <pcl/filters/impl/extract_indices.hpp>
#include <pcl/filters/impl/voxel_grid.hpp>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/search/impl/kdtree.hpp>
#include <pcl/segmentation/impl/extract_clusters.hpp>
#include <pcl/segmentation/impl/organized_multi_plane_segmentation.hpp>
#include <pcl/segmentation/plane_coefficient_comparator.h>
#include <pcl/segmentation/euclidean_plane_coefficient_comparator.h>
#include <pcl/segmentation/rgb_plane_coefficient_comparator.h>
#include <pcl/segmentation/edge_aware_plane_comparator.h>
#include <pcl/segmentation/euclidean_cluster_comparator.h>

#include "object_analytics_nodelet/OrganizedMultiPlaneSegmentationConfig.h"
#include "object_analytics_nodelet/segmenter/algorithm_config.h"
#include "object_analytics_nodelet/segmenter/algorithm.h"

namespace object_analytics_nodelet
{
namespace segmenter
{
using PointT = pcl::PointXYZRGBA;
using PointCloudT = pcl::PointCloud<PointT>;
using object_analytics_nodelet::segmenter::AlgorithmConfig;
using object_analytics_nodelet::OrganizedMultiPlaneSegmentationConfig;

/** @class OrganizedMultiPlaneSegmenter
 * SegmentAlgorithm implementation using organized multi plane segmentaion algorithm.
 */
class OrganizedMultiPlaneSegmenter : public Algorithm
{
public:
  /**
   * Constructor
   *
   * @param[in]   nh              Ros NodeHandle
   */
  explicit OrganizedMultiPlaneSegmenter(ros::NodeHandle& nh);

  /** Default destructor */
  ~OrganizedMultiPlaneSegmenter() = default;

  /**
   * Segment given point cloud into individuals which could be tell from each other in 3d spaces. Due to performance
   * reason the passed out cloud has been downsampled.
   *
   * @param[in]   cloud           Ponit cloud to segment
   * @param[out]  cloud_segment   Point cloud contains all individuals
   * @param[out]  cluster_indices Indices vector, each indidcates an individual in cloud_segment
   */
  void segment(const PointCloudT::ConstPtr& cloud, PointCloudT::Ptr& cloud_segment,
               std::vector<pcl::PointIndices>& cluster_indices);

private:
  void estimateNormal(const PointCloudT::ConstPtr& cloud, pcl::PointCloud<pcl::Normal>::Ptr& cloud_normal);
  void
  segmentPlanes(const PointCloudT::ConstPtr& cloud, const pcl::PointCloud<pcl::Normal>::Ptr& normal_cloud,
                std::vector<pcl::PlanarRegion<PointT>, Eigen::aligned_allocator<pcl::PlanarRegion<PointT>>>& regions,
                pcl::PointCloud<pcl::Label>::Ptr labels, std::vector<pcl::PointIndices>& label_indices);
  void
  segmentObjects(const PointCloudT::ConstPtr& cloud,
                 std::vector<pcl::PlanarRegion<PointT>, Eigen::aligned_allocator<pcl::PlanarRegion<PointT>>>& regions,
                 pcl::PointCloud<pcl::Label>::Ptr labels, std::vector<pcl::PointIndices>& label_indices,
                 std::vector<pcl::PointIndices>& cluster_indices);

  void applyConfig();

  pcl::OrganizedMultiPlaneSegmentation<PointT, pcl::Normal, pcl::Label> plane_segmentation_;// 多平面分割
  pcl::IntegralImageNormalEstimation<PointT, pcl::Normal> normal_estimation_;// 积分图像 点云 法线 估计

  pcl::PlaneCoefficientComparator<PointT, pcl::Normal>::Ptr plane_comparator_;// 平面系数
  pcl::EuclideanPlaneCoefficientComparator<PointT, pcl::Normal>::Ptr euclidean_comparator_; // 欧氏距离平面系数
  pcl::RGBPlaneCoefficientComparator<PointT, pcl::Normal>::Ptr rgb_comparator_;// 颜色分割 平面系数
  pcl::EdgeAwarePlaneComparator<PointT, pcl::Normal>::Ptr edge_aware_comparator_;// 平面边缘
  pcl::EuclideanClusterComparator<PointT, pcl::Normal, pcl::Label>::Ptr euclidean_cluster_comparator_;// 欧氏距离聚类物体分割

  AlgorithmConfig<OrganizedMultiPlaneSegmentationConfig> conf_;

  size_t plane_minimum_points_;
  size_t object_minimum_points_;

  enum Comparator
  {
    kPlaneCoefficientComparator,
    kEuclideanPlaneCoefficientComparator,
    kRGBPlaneCoefficientComparator,
    kEdgeAwarePlaneComaprator
  };
};
}  // namespace segmenter
}  // namespace object_analytics_nodelet
#endif  // OBJECT_ANALYTICS_NODELET_SEGMENTER_ORGANIZED_MULTI_PLANE_SEGMENTER_H
