/*
保存2d-3d点数据文件
*/
#ifndef CSVWRITER_H
#define	CSVWRITER_H

#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include "Utils.h"

using namespace std;
using namespace cv;

class CsvWriter {
public:
  // 默认构造函数
  CsvWriter(const string &path, const string &separator = " ");
  // 默认析构函数
  ~CsvWriter();
  void writeXYZ(const vector<Point3f> &list_points3d);//写3d点
  // 写　2d-3d 点对 　＋　特征点对应的描述子（二维特征描述）
  void writeUVXYZ(const vector<Point3f> &list_points3d, const vector<Point2f> &list_points2d, const Mat &descriptors);

private:
  ofstream _file;//输出文件流
  string _separator;//　分隔符
  bool _isFirstTerm;//
};

#endif
