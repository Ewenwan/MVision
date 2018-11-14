# 基础常用
## 读取、转换、存储图片
## 简单画图 线 圆 椭圆 多边形  随机画图
## 图像元素访问 查找表 阶梯化像素值  色域缩减
## 快速 RGB 变 BGR
```c
cv::Mat rgb;
rgb = cv::Mat(480, 640, CV_8UC3);// 0～255 RGB 数据

  // RGB 变 BGR===========================
  void flipColors() 
  {
#pragma omp parallel for
    for (unsigned i = 0; i < rgb.total() * 3; i += 3) std::swap(rgb.data[i + 0], rgb.data[i + 2]);
  }

```
