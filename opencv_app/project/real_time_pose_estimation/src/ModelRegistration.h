/*
 * ModelRegistration.h
  记录 2d-3d 匹配点对 模型配准类
 */

#ifndef MODELREGISTRATION_H_
#define MODELREGISTRATION_H_

#include <iostream>
#include <opencv2/core/core.hpp>

class ModelRegistration
{
// 公有
public:
	ModelRegistration();//默认构造函数
	virtual ~ModelRegistration();// 虚析构函数

        // 简单的函数 头文件直接实现
	void setNumMax(int n) { max_registrations_ = n; }// 配准的最大点对总数
	std::vector<cv::Point2f> get_points2d() const { return list_points2d_; }//2d点坐标 像素值坐标
	std::vector<cv::Point3f> get_points3d() const { return list_points3d_; }//3d点坐标 三维空间坐标
	int getNumMax() const { return max_registrations_; }//最大匹配点对数
	int getNumRegist() const { return n_registrations_; }//当前已经匹配的点对数

        // 新定义 类cpp文件 实现 稍微复制的 类 方法函数

        // 判断当前是否还需要 确定匹配点 
	bool is_registrable() const { return (n_registrations_ < max_registrations_); }

       // 配准点
	void registerPoint(const cv::Point2f &point2d, const cv::Point3f &point3d);
	void reset();// 重置
//私有
private:
	// 当前配准的点数 The current number of registered points 
	int n_registrations_;
	// 配准的最大点对总数 The total number of points to register
	int max_registrations_;
	// 需要配置的二维点 The list of 2D points to register the model
	std::vector<cv::Point2f> list_points2d_;
	// 需要配准的三维点 The list of 3D points to register the model 
	std::vector<cv::Point3f> list_points3d_;
};

#endif /* MODELREGISTRATION_H_ */
