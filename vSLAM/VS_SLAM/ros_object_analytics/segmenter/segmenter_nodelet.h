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

#ifndef OBJECT_ANALYTICS_NODELET_SEGMENTER_SEGMENTER_NODELET_H
#define OBJECT_ANALYTICS_NODELET_SEGMENTER_SEGMENTER_NODELET_H

#include <nodelet/nodelet.h>

#include "object_analytics_nodelet/segmenter/segmenter.h"

namespace object_analytics_nodelet
{
namespace segmenter
{
/** @class SegmenterNodelet
 * Segmenter nodelet, segmenter implementation holder.
 */
class SegmenterNodelet : public nodelet::Nodelet
{
public:
  /** Default desctructor */
  ~SegmenterNodelet() = default;

private:
  /** Inherit from Nodelet class. Initialize Segmenter instance. */
  virtual void onInit();

  /**
   * @brief PointCloud2 callback
   *
   * @param[in] points PointCloud2 message from sensor.
   */
  void cbSegment(const sensor_msgs::PointCloud2::ConstPtr& points);

  ros::Subscriber sub_;
  ros::Publisher pub_;

  std::unique_ptr<Segmenter> impl_;
};
}  // namespace segmenter
}  // namespace object_analytics_nodelet
#endif  // OBJECT_ANALYTICS_NODELET_SEGMENTER_SEGMENTER_NODELET_H
