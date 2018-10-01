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

#ifndef OBJECT_ANALYTICS_NODELET_SEGMENTER_ALGORITHM_PROVIDER_H
#define OBJECT_ANALYTICS_NODELET_SEGMENTER_ALGORITHM_PROVIDER_H

#include "object_analytics_nodelet/segmenter/algorithm.h"

namespace object_analytics_nodelet
{
namespace segmenter
{
using object_analytics_nodelet::segmenter::Algorithm;

/** @class AlorithmProvider
 * Segmentation algorithm factory class
 */
class AlgorithmProvider
{
public:
  /**
   * Get current selected algorithm instance
   *
   * @return Pointer to current slected algorithm instance
   */
  virtual std::shared_ptr<Algorithm> get() = 0;

  /**
   * Default virtual destructor
   */
  virtual ~AlgorithmProvider()
  {
  }
};
}  // namespace segmenter
}  // namespace object_analytics_nodelet
#endif  // OBJECT_ANALYTICS_NODELET_SEGMENTER_ALGORITHM_PROVIDER_H
