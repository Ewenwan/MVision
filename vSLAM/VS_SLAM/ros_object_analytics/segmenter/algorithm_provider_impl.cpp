/*
 * 
 */

#include <list>
#include <string>

#include <ros/ros.h>

#include "object_analytics_nodelet/segmenter/organized_multi_plane_segmenter.h"
#include "object_analytics_nodelet/segmenter/algorithm_provider_impl.h"

using object_analytics_nodelet::segmenter::OrganizedMultiPlaneSegmenter;

namespace object_analytics_nodelet
{
namespace segmenter
{
const std::string AlgorithmProviderImpl::DEFAULT = "OrganizedMultiPlaneSegmentation";

AlgorithmProviderImpl::AlgorithmProviderImpl(ros::NodeHandle& nh)
{
  conf_srv_ = boost::make_shared<dynamic_reconfigure::Server<SegmentationAlgorithmsConfig>>(nh);
  conf_srv_->getConfigDefault(conf_);

  conf_srv_->setCallback(boost::bind(&AlgorithmProviderImpl::cbConfig, this, _1, _2));

  algorithms_["OrganizedMultiPlaneSegmentation"] =
      std::static_pointer_cast<Algorithm>(std::make_shared<OrganizedMultiPlaneSegmenter>(nh));
}

std::shared_ptr<Algorithm> AlgorithmProviderImpl::get()
{
  std::string name = conf_.algorithm;
  try
  {
    std::shared_ptr<Algorithm> algo = algorithms_.at(name);
    return algo;
  }
  catch (const std::out_of_range& e)
  {
    ROS_WARN_STREAM("Algorithm named " << name << " doesn't exist, use " << AlgorithmProviderImpl::DEFAULT << " instea"
                                                                                                              "d");
    return algorithms_.at(DEFAULT);
  }
}

void AlgorithmProviderImpl::cbConfig(SegmentationAlgorithmsConfig& config, uint32_t level)
{
  conf_ = config;
}
}  // namespace segmenter
}  // namespace object_analytics_nodelet
