/*
读取
CSV格式的ply文件类

*/
#ifndef CSVREADER_H
#define	CSVREADER_H

#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include "Utils.h"

using namespace std;
using namespace cv;

class CsvReader {
public:

// 默认构造函数  ' ' (empty space)
  CsvReader(const string &path, const char &separator = ' ');

// // 读取ply文件　　得到　顶点列表　　和三角形面　列表
  void readPLY(vector<Point3f> &list_vertex, vector<vector<int> > &list_triangles);

private:
  // 
  ifstream _file;
  // ' ' (empty space)
  char _separator;
};

#endif
