/*
 *物体网格模型  物体的 三维纹理 模型文件，
   包含：2d-3d点对 特征点 特征点描述子
   Model类  Model.h
    textured model实现 Model类(class), 　
 */

#ifndef MODEL_H_
#define MODEL_H_

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

class Model
{
public:
  Model();//默认构造函数
  virtual ~Model();// 虚析构函数

  // 物体表面的关键点 
  std::vector<cv::KeyPoint> get_keypoints() const { return list_keypoints_; }//
  // 物体表面上 对应的 图像上的 2维点
  std::vector<cv::Point2f> get_points2d_in() const { return list_points2d_in_; }
  // 不在物体表面 对应的 图像上的 2维点
  std::vector<cv::Point2f> get_points2d_out() const { return list_points2d_out_; }
  // 物体表面上 对应的 图像上的 2维点 对应的3维点
  std::vector<cv::Point3f> get_points3d() const { return list_points3d_in_; }//
  // 2维点对应的描述子
  cv::Mat get_descriptors() const { return descriptors_; }// 关键点对应的描述子
  // 描述子数量
  int get_numDescriptors() const { return descriptors_.rows; }//
  // 以上类函数直接在 头文件内实现

// 一下稍微复杂的函数在 类对象的.cpp文件中实现

  // 添加 2d-3d 点对
  void add_correspondence(const cv::Point2f &point2d, const cv::Point3f &point3d);
  // 添加不在物体表面 对应的 图像上的 2维点
  void add_outlier(const cv::Point2f &point2d);
  // 添加描述子
  void add_descriptor(const cv::Mat &descriptor);
  // 添加关键点
  void add_keypoint(const cv::KeyPoint &kp);

  // 保存 物体的 三维纹理 模型文件 *.yaml 
  void save(const std::string path);
  // 载入 物体的 三维纹理 模型文件 *.yaml 
  void load(const std::string path);// 来打开 YAML 格式的文件，并读取存储的 3D点和相应的描述子


private:
  // 当前匹配点对 2d-3d 数量
  int n_correspondences_;
  // 物体表面的关键点 
  std::vector<cv::KeyPoint> list_keypoints_;
  // 物体表面上 对应的 图像上的 2维点
  std::vector<cv::Point2f> list_points2d_in_;
  // 不在物体表面 对应的 图像上的 2维点
  std::vector<cv::Point2f> list_points2d_out_;
  // 物体表面上 对应的 图像上的 2维点 对应的三维点
  std::vector<cv::Point3f> list_points3d_in_;
  // 2维点对应的描述子
  cv::Mat descriptors_;
};

#endif /* OBJECTMODEL_H_ */
