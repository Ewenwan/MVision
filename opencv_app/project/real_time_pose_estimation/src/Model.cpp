/* 物体网格模型  物体的 三维纹理 模型文件，
   包含：2d-3d点对 特征点 特征点描述子
 * Model类 Model.cpp
  textured model实现 Model类(class), 　
 */

#include "Model.h"
#include "CsvWriter.h"

//默认构造函数
Model::Model() : list_points2d_in_(0), list_points2d_out_(0), list_points3d_in_(0)
{
  n_correspondences_ = 0;
}

Model::~Model()// 虚析构函数
{
  // TODO Auto-generated destructor stub
}

// 添加 2d-3d 点对
void Model::add_correspondence(const cv::Point2f &point2d, const cv::Point3f &point3d)
{
  list_points2d_in_.push_back(point2d);
  list_points3d_in_.push_back(point3d);
  n_correspondences_++;//点对计数 ++ 
}

// 添加不在物体表面 对应的 图像上的 2维点
void Model::add_outlier(const cv::Point2f &point2d)
{
  list_points2d_out_.push_back(point2d);
}

// 添加描述子
void Model::add_descriptor(const cv::Mat &descriptor)
{
  descriptors_.push_back(descriptor);
}

// 添加关键点
void Model::add_keypoint(const cv::KeyPoint &kp)
{
  list_keypoints_.push_back(kp);
}


// 保存 物体的 三维纹理 模型文件 *.yaml 
void Model::save(const std::string path)
{
  cv::Mat points3dmatrix = cv::Mat(list_points3d_in_);//2d点 vector 保存成 Mat
  cv::Mat points2dmatrix = cv::Mat(list_points2d_in_);//3d点 vector 保存成 Mat
  //cv::Mat keyPointmatrix = cv::Mat(list_keypoints_);

  cv::FileStorage storage(path, cv::FileStorage::WRITE);//文件保存
  storage << "points_3d" << points3dmatrix;
  storage << "points_2d" << points2dmatrix;
  storage << "keypoints" << list_keypoints_;
  storage << "descriptors" << descriptors_;

  storage.release();// 释放文件句柄 
}

// 载入 物体的 三维纹理 模型文件 *.yaml 
void Model::load(const std::string path)
{
  cv::Mat points3d_mat;// 3d点是按 Mat存储的

  cv::FileStorage storage(path, cv::FileStorage::READ);//打开读取文件
  storage["points_3d"] >> points3d_mat;
  storage["descriptors"] >> descriptors_;// 3d点 描述子就是按 vector存储的

  points3d_mat.copyTo(list_points3d_in_);// 2d点 Mat -> vector

  storage.release();// 释放文件句柄

}
