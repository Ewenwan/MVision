/* 常用的一些函数
  如画图 画线 显示文字 显示 帧率 可信度 中心点
  浮点数 float 转换成  字符串string
  Utils.h
 *
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <iostream>

#include "PnPProblem.h"

// 显示问句
void drawQuestion(cv::Mat image, cv::Point3f point, cv::Scalar color);

// 显示文字
void drawText(cv::Mat image, std::string text, cv::Scalar color);

// 显示文字
void drawText2(cv::Mat image, std::string text, cv::Scalar color);

// 显示 帧率 frame ratio
void drawFPS(cv::Mat image, double fps, cv::Scalar color);

// 显示 可信度 Confidence
void drawConfidence(cv::Mat image, double confidence, cv::Scalar color);

// 显示中心点
void drawCounter(cv::Mat image, int n, int n_max, cv::Scalar color);

// 画2d-3d 匹配点
void drawPoints(cv::Mat image, std::vector<cv::Point2f> &list_points_2d, std::vector<cv::Point3f> &list_points_3d, cv::Scalar color);

// 画2d点
void draw2DPoints(cv::Mat image, std::vector<cv::Point2f> &list_points, cv::Scalar color);

// 画箭头
void drawArrow(cv::Mat image, cv::Point2i p, cv::Point2i q, cv::Scalar color, int arrowMagnitude = 9, int thickness=1, int line_type=8, int shift=0);

// 话三维坐标轴
void draw3DCoordinateAxes(cv::Mat image, const std::vector<cv::Point2f> &list_points2d);

// 画物体的 网格 mesh
void drawObjectMesh(cv::Mat image, const Mesh *mesh, PnPProblem *pnpProblem, cv::Scalar color);

// 计算给定平移向量之差的 大小
double get_translation_error(const cv::Mat &t_true, const cv::Mat &t);

// 计算两个给定矩阵之差 的 norm 范数 矩阵
double get_rotation_error(const cv::Mat &R_true, const cv::Mat &R);

//  旋转矩阵Matrix 转换成 欧拉角Euler angles
cv::Mat rot2euler(const cv::Mat & rotationMatrix);

// 欧拉角Euler angles 转换成  旋转矩阵 Rotation Matrix  罗德里格
cv::Mat euler2rot(const cv::Mat & euler);

// 字符串string 转换成 整形integer
int StringToInt ( const std::string &Text );

// 字符串string 转换成 float
float StringToFloat ( const std::string &Text );

// 浮点数 float 转换成  字符串string
std::string FloatToString ( float Number );

// 整形 integer 转换成  字符串string
std::string IntToString ( int Number );

#endif 
/* UTILS_H_ */
