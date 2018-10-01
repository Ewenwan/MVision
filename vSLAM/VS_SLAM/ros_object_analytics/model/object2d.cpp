/*
 * Copyright (c) 2017 Intel Corporation
 */

#include <opencv2/core/types.hpp>
#include <ros/console.h>

#include "object_analytics_nodelet/model/object2d.h"

namespace object_analytics_nodelet
{
namespace model
{// 2d边框 + 物体名 ========== 
Object2D::Object2D(const object_msgs::ObjectInBox& oib) : roi_(oib.roi), object_(oib.object)
{
}

std::ostream& operator<<(std::ostream& os, const Object2D& obj)
{
  os << "Object2D[" << obj.object_.object_name;
  os << ", @(" << obj.roi_.x_offset << ", " << obj.roi_.y_offset << ")";
  os << ", width=" << obj.roi_.width << ", height=" << obj.roi_.height << "]";
  return os;
}
}  // namespace model
}  // namespace object_analytics_nodelet
