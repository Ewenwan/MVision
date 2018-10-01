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

#ifndef OBJECT_ANALYTICS_NODELET_SEGMENTER_ALGORITHM_CONFIG_H
#define OBJECT_ANALYTICS_NODELET_SEGMENTER_ALGORITHM_CONFIG_H

#include <string>

#include <boost/make_unique.hpp>       // 老
//#include <boost/smart_ptr/make_unique.hpp>

#include <ros/ros.h>
#include <dynamic_reconfigure/server.h>

namespace object_analytics_nodelet
{
namespace segmenter
{
/** @class AlorithmConfig
 *
 * Template class for encapsulating config related operations.
 */
template <class ConfigType>
class AlgorithmConfig
{
public:
  /**
   * Constructor of AlgorithmConfig
   *
   * @param[in] nh Ros NodeHandle
   * @param[in] name Name of the ros NodeHandle
   */
  explicit AlgorithmConfig(ros::NodeHandle& nh, const std::string& name)
  {
    ros::NodeHandle pnh = ros::NodeHandle(nh, name);
    conf_srv_ = boost::make_shared<dynamic_reconfigure::Server<ConfigType>>(pnh);
    conf_srv_->getConfigDefault(conf_);

    conf_srv_->setCallback(boost::bind(&AlgorithmConfig<ConfigType>::cbConfig, this, _1, _2));
  }

  /**
   * Get ConfigType instance
   *
   * @return ConfigType isntance
   */
  ConfigType getConfig()
  {
    return conf_;
  }

private:
  /**
   * Callback method. Called everytime ConfigType is changed, member variable conf_ will be updated accordingly.
   *
   * @param[in] config  Changed ConfigType instance
   * @param[in] level   Not used
   */
  void cbConfig(ConfigType& config, uint32_t level)
  {
    conf_ = config;
  }

  boost::shared_ptr<dynamic_reconfigure::Server<ConfigType>> conf_srv_;
  ConfigType conf_;
};
}  // namespace segmenter
}  // namespace object_analytics_nodelet
#endif  // OBJECT_ANALYTICS_NODELET_SEGMENTER_ALGORITHM_CONFIG_H
