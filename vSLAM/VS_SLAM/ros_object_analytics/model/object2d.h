/*
 * Copyright (c) 2017 Intel Corporation
 */

#ifndef OBJECT_ANALYTICS_NODELET_MODEL_OBJECT2D_H
#define OBJECT_ANALYTICS_NODELET_MODEL_OBJECT2D_H

#include <object_msgs/ObjectInBox.h>

namespace object_analytics_nodelet //命名空间 
{
namespace model //命名空间 
{
/** @class Object2D
 * @brief Wrapper of object_msgs::ObjectInBox.
 *
 * Constructed from 2d detection result, represents one object_msgs::ObjectInBox instance.
 */
class Object2D // Object2D类
{
public:
  /**
   * Constructor
   *
   * @param[in] object_in_box   Object in box which comes from 2d detection result.
   */
  explicit Object2D(const object_msgs::ObjectInBox& object_in_box);

  /** Default destructor */
  ~Object2D() = default;

  /**
   * Get the region of interest in sensor_msgs::RegionOfInterest type of underlying object in image space.
   *
   * @return Underlying region of interest in image space
   */
  inline sensor_msgs::RegionOfInterest getRoi() const
  {
    return roi_;// 物体边框 
  }

  /**
   * Get the underlying object_msgs::Object.
   *
   * @return The underlying object_msgs::Object
   */
  inline object_msgs::Object getObject() const
  {
    return object_;// 物体名
  }

  /**
   * Overload operator << to dump information of underlying information.
   *
   * @param[in,out] os    Standard output stream
   * @param[in]     obj   Object to be dumped
   *
   * @return Standard output stream
   */
  friend std::ostream& operator<<(std::ostream& os, const Object2D& obj);

private:
  const sensor_msgs::RegionOfInterest roi_;// 物体边框
  const object_msgs::Object object_;// 物体名称 + 概率
};

using Object2DPtr = std::shared_ptr<Object2D>;// 2d物体类 指针
using Object2DConstPtr = std::shared_ptr<Object2D const>;// 2d物体类 常指针
}  // namespace model
}  // namespace object_analytics_nodelet
#endif  // OBJECT_ANALYTICS_NODELET_MODEL_OBJECT2D_H
