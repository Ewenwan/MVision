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

#ifndef OBJECT_ANALYTICS_NODELET_SEGMENTER_ALGORITHM_H
#define OBJECT_ANALYTICS_NODELET_SEGMENTER_ALGORITHM_H

#define PCL_NO_PRECOMPILE
#include <vector>
#include <pcl/point_types.h>
#include <pcl/common/projection_matrix.h>

namespace object_analytics_nodelet
{
namespace segmenter
{
/** @class Alorithm
 * Interface. Define segmentaion alrogirthm Interface.
 */
class Algorithm
{
public:
  /** Default destructor */
  virtual ~Algorithm() = default;

  /**
   * Segment given point cloud into individuals which could be tell from each other in 3d spaces. Due to performance
   * reason the passed out cloud has been downsampled.
   *
   * @param[in]   cloud           Ponit cloud to segment
   * @param[out]  cloud_segment   Point cloud contains all individuals
   * @param[out]  cluster_indices Indices vector, each indidcates an individual in cloud_segment
   */
  virtual void segment(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr& cloud,
                       pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloud_segment,
                       std::vector<pcl::PointIndices>& cluster_indices) = 0;
};
}  // namespace segmenter
}  // namespace object_analytics_nodelet
#endif  // OBJECT_ANALYTICS_NODELET_SEGMENTER_ALGORITHM_H
