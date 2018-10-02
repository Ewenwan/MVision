// 检测对象头文件

#ifndef OBJECT_ANALYTICS_NODELET_DETECTOR_DETECTOR_H
#define OBJECT_ANALYTICS_NODELET_DETECTOR_DETECTOR_H

#include "./ncnn/include/net.h"  // ncnn 头文件
#include <opencv2/core/core.hpp>
#include <vector>

// 检测结果类====
struct Object
{
    cv::Rect_<float> rect;// 边框
    std::string object_name;// 物体类别名
    float prob;// 置信度
};


namespace object_analytics_nodelet
{
namespace detector
{
class Detector
{
public:
  /** Default constructor */
  Detector();// 网络初始化

  /** Default destructor */
  ~Detector();// 析构函数
  
  void Run(const cv::Mat& bgr_img, std::vector<Object>& objects);
  
private:
   // ncnn::Net * det_net_mobile;  
   ncnn::Net * det_net_ptr;// 检测网络指针
   ncnn::Mat * net_in_ptr; // 网络输入指针
};
}
}

#endif
