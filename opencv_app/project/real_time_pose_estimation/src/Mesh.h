/*
 * 网格数据Mesh类(class)  Mesh.h
 * 三角形类Triangle   射线Ray类    物体网格类 Mesh

 */

#ifndef MESH_H_
#define MESH_H_

#include <iostream>
#include <opencv2/core/core.hpp>


// --------------------------------------------------- //
//                 TRIANGLE CLASS 三角形类             //
// --------------------------------------------------- //

class Triangle {
public:
 // explicit 指明　不可以　隐式转换　初始化　必须　显示转换　初始化
  explicit Triangle(int id, cv::Point3f V0, cv::Point3f V1, cv::Point3f V2);
  virtual ~Triangle();//虚析构函数

  cv::Point3f getV0() const { return v0_; }// 三角形的一个点　三维空间点
  cv::Point3f getV1() const { return v1_; }// 三角形的一个点
  cv::Point3f getV2() const { return v2_; }// 三角形的一个点

private:
  // 三角形 id
  int id_;
  // 三角形的三个点　三维空间点
  cv::Point3f v0_, v1_, v2_;
};


// --------------------------------------------------- //
//                     RAY CLASS　线段类　　　　　　　　　//
// --------------------------------------------------- //

class Ray {
public:
 // explicit 指明　不可以　隐式转换　初始化　必须　显示转换　初始化
  explicit Ray(cv::Point3f P0, cv::Point3f P1);
  virtual ~Ray();//虚析构函数

  cv::Point3f getP0() { return p0_; }
  cv::Point3f getP1() { return p1_; }

private:
  // 两个三维空间点　组成一条线段　三个线段组成一个三角形面
  cv::Point3f p0_, p1_;
};


// --------------------------------------------------- //
//                OBJECT MESH CLASS 物体网格模型类      //
// --------------------------------------------------- //

class Mesh
{
public:

  Mesh();//　构造函数
  virtual ~Mesh();//虚析构函数
  // 三角形列表
  std::vector<std::vector<int> > getTrianglesList() const { return list_triangles_; }
  // 顶点
  cv::Point3f getVertex(int pos) const { return list_vertex_[pos]; }
  // 定点数量
  int getNumVertices() const { return num_vertexs_; }
  // 载入　网格模型文件
  void load(const std::string path_file);

private:
  // 网格模型ｉｄ
  int id_;
  // 顶点数量
  int num_vertexs_;
  // 三角形面数量　例如一个长方体　8个顶点　　6个面　12个三角形面
  int num_triangles_;
  // 顶点列表
  std::vector<cv::Point3f> list_vertex_;
  // 三角形面列表　存储　顶点的　索引
  std::vector<std::vector<int> > list_triangles_;// n*3 数组　对于一个长方体　n =12
};

#endif /* OBJECTMESH_H_ */
