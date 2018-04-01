/*
 二维图像创建 三维数据
 * ModelRegistration.cpp
射线与三角形

你可以使用这个代码创建自定义的textured 3D model.
当前代码只对平面的(或者说面试平的)物体有效,
如果需要得到一个形状复杂的模型需要使用复杂的软件创建。


当前代码的输入为一幅图像和图像对应的3D网格。
我们提供相机的内参，用来校正获取的图像作为算法的输入图像.
所有的文件需要通过绝对路径或者相对于工作目录的相对路径指定。
如果没有指定文件，代码将会尝试使用默认参数.


程序执行，首先从输入图像中提取 ORB特征描述子，
然后使用网格数据和 Möller–Trumbore intersection 算法 
来计算特征的3D坐标系。
最后，3D坐标点和特征描述子存在YAML格式文件的不同列表中，
每一行存储一个不同的点.

 */

#include "ModelRegistration.h"

// 类的定义和实现分离
// 类函数定义的地方（另一个文件） 需要前置类名

// 类默认构造函数
ModelRegistration::ModelRegistration()
{
  n_registrations_ = 0;
  max_registrations_ = 0;
}

// 类析构函数 
ModelRegistration::~ModelRegistration()
{
  // TODO Auto-generated destructor stub 自动生成析构函数  
}

// 加入配准点
void ModelRegistration::registerPoint(const cv::Point2f &point2d, const cv::Point3f &point3d)
 {
   // add correspondence at the end of the vector
    list_points2d_.push_back(point2d);
    list_points3d_.push_back(point3d);
    n_registrations_++;
 }

void ModelRegistration::reset()
{
  n_registrations_ = 0;
  max_registrations_ = 0;
  list_points2d_.clear();
  list_points3d_.clear();
}
