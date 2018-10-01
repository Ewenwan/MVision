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

#include <pluginlib/class_list_macros.h>
#include "object_analytics_nodelet/const.h"
#include "object_analytics_nodelet/segmenter/segmenter_nodelet.h"
#include "object_analytics_nodelet/segmenter/algorithm_provider_impl.h"

namespace object_analytics_nodelet
{
namespace segmenter
{
using object_analytics_nodelet::segmenter::AlgorithmProvider;
using object_analytics_nodelet::segmenter::AlgorithmProviderImpl;

void SegmenterNodelet::onInit()
{
  ros::NodeHandle nh = getNodeHandle();
  sub_ = nh.subscribe(Const::kTopicPC2, 1, &SegmenterNodelet::cbSegment, this);
  pub_ = nh.advertise<object_analytics_msgs::ObjectsInBoxes3D>(Const::kTopicSegmentation, 1);

  try
  {
    impl_.reset(new Segmenter(std::unique_ptr<AlgorithmProvider>(new AlgorithmProviderImpl(nh))));
  }
  catch (const std::runtime_error& e)
  {
    ROS_ERROR_STREAM("exception caught while starting segmenter nodelet, " << e.what());
    ros::shutdown();
  }
}

void SegmenterNodelet::cbSegment(const sensor_msgs::PointCloud2::ConstPtr& points)
{
  if (pub_.getNumSubscribers() == 0)
  {
    ROS_DEBUG_STREAM("No subscriber is listening on me, just skip");
    return;
  }

  boost::shared_ptr<ObjectsInBoxes3D> msg = boost::make_shared<ObjectsInBoxes3D>();
  msg->header = points->header;

  impl_->segment(points, msg);

  pub_.publish(msg);
}

}  // namespace segmenter
}  // namespace object_analytics_nodelet
PLUGINLIB_EXPORT_CLASS(object_analytics_nodelet::segmenter::SegmenterNodelet, nodelet::Nodelet)
