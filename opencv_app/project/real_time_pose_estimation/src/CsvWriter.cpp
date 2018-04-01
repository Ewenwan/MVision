/*
保存2d-3d点数据文件
*/
#include "CsvWriter.h"

// 默认构造函数
CsvWriter::CsvWriter(const string &path, const string &separator){
  _file.open(path.c_str(), ofstream::out);//打开文件　输出
  _isFirstTerm = true;
  _separator = separator;//　分割符
}
//　默认析构函数
CsvWriter::~CsvWriter() {
  _file.flush();//　清空文件缓存区
  _file.close();//　关闭文件
}
//写3d点
void CsvWriter::writeXYZ(const vector<Point3f> &list_points3d)
{
  string x, y, z;//　字符串　
  for(unsigned int i = 0; i < list_points3d.size(); ++i)
  {
    x = FloatToString(list_points3d[i].x);//转换成字符串
    y = FloatToString(list_points3d[i].y);
    z = FloatToString(list_points3d[i].z);
    // 输出到文件
    _file << x << _separator << y << _separator << z << std::endl;
  }

}
// 写　2d-3d 点对 　＋　特征点对应的描述子（二维特征描述）
void CsvWriter::writeUVXYZ(const vector<Point3f> &list_points3d, const vector<Point2f> &list_points2d, const Mat &descriptors)
{
  string u, v, x, y, z, descriptor_str;//　字符串　
  for(unsigned int i = 0; i < list_points3d.size(); ++i)
  {
    u = FloatToString(list_points2d[i].x);//转换成字符串
    v = FloatToString(list_points2d[i].y);
    x = FloatToString(list_points3d[i].x);
    y = FloatToString(list_points3d[i].y);
    z = FloatToString(list_points3d[i].z);

    _file << u << _separator << v << _separator << x << _separator << y << _separator << z;

    for(int j = 0; j < 32; ++j)//前32个　　描述子
    {
      descriptor_str = FloatToString(descriptors.at<float>(i,j));
      _file << _separator << descriptor_str;
    }
    _file << std::endl;
  }
}
