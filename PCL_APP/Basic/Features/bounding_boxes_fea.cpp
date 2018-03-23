/*
moment_of_inertia.cpp

momentofinertiaestimation类获取基于惯性偏心矩 描述子。
这个类还允许提取云的轴对齐和定向包围盒。
但请记住，不可能的最小提取OBB包围盒。

首先计算点云的协方差矩阵，提取其特征值和向量。
你可以认为所得的特征向量是标准化的，
总是形成右手坐标系（主要特征向量表示x轴，而较小的向量表示z轴）。
在下一个步骤中，迭代过程发生。每次迭代时旋转主特征向量。
旋转顺序总是相同的，并且在其他特征向量周围执行，
这提供了点云旋转的不变性。
此后，我们将把这个旋转主向量作为当前轴。

然后在当前轴上计算转动惯量 和 将点云投影到以旋转向量为法线的平面上
计算偏心距

*/
#include <pcl/features/moment_of_inertia_estimation.h>
#include <vector>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <boost/thread/thread.hpp>

int main (int argc, char** argv)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr (new pcl::PointCloud<pcl::PointXYZ> ());

  if (argc != 2){
    if (pcl::io::loadPCDFile ("../lamppost.pcd", *cloud_ptr) == -1)
        return (-1);
  }
  else if (pcl::io::loadPCDFile (argv[1], *cloud_ptr) == -1)
    return (-1);

  pcl::MomentOfInertiaEstimation <pcl::PointXYZ> feature_extractor;// 惯量 偏心距 特征提取
  feature_extractor.setInputCloud (cloud_ptr);//输入点云
  feature_extractor.compute ();//计算

  std::vector <float> moment_of_inertia;// 惯量
  std::vector <float> eccentricity;// 偏心距
  pcl::PointXYZ min_point_AABB;
  pcl::PointXYZ max_point_AABB;
  pcl::PointXYZ min_point_OBB;
  pcl::PointXYZ max_point_OBB;
  pcl::PointXYZ position_OBB;
  Eigen::Matrix3f rotational_matrix_OBB;
  float major_value, middle_value, minor_value;
  Eigen::Vector3f major_vector, middle_vector, minor_vector;
  Eigen::Vector3f mass_center;

  feature_extractor.getMomentOfInertia (moment_of_inertia);// 惯量
  feature_extractor.getEccentricity (eccentricity);// 偏心距
// 以八个点的坐标 给出 包围盒
  feature_extractor.getAABB (min_point_AABB, max_point_AABB);// AABB 包围盒坐标 八个点的坐标
// 以中心点 姿态 坐标轴范围 给出   包围盒
  feature_extractor.getOBB (min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);// 位置和方向姿态
  feature_extractor.getEigenValues (major_value, middle_value, minor_value);//特征值
  feature_extractor.getEigenVectors (major_vector, middle_vector, minor_vector);//特征向量
  feature_extractor.getMassCenter (mass_center);//点云中心点

// 可视化
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);//背景黑色
  viewer->addCoordinateSystem (1.0);//坐标系 尺寸
  viewer->initCameraParameters ();//初始化相机参数
  viewer->addPointCloud<pcl::PointXYZ> (cloud_ptr, "sample cloud");//输入点云可视化
  viewer->addCube (min_point_AABB.x, max_point_AABB.x, min_point_AABB.y, max_point_AABB.y, min_point_AABB.z, max_point_AABB.z, 1.0, 1.0, 0.0, "AABB");//包围盒

  Eigen::Vector3f position (position_OBB.x, position_OBB.y, position_OBB.z);//位置
  Eigen::Quaternionf quat (rotational_matrix_OBB);//方向姿态 四元数表示
  viewer->addCube (position, quat, max_point_OBB.x - min_point_OBB.x, max_point_OBB.y - min_point_OBB.y, max_point_OBB.z - min_point_OBB.z, "OBB");

  pcl::PointXYZ center (mass_center (0), mass_center (1), mass_center (2));
  pcl::PointXYZ x_axis (major_vector (0) + mass_center (0), major_vector (1) + mass_center (1), major_vector (2) + mass_center (2));
  pcl::PointXYZ y_axis (middle_vector (0) + mass_center (0), middle_vector (1) + mass_center (1), middle_vector (2) + mass_center (2));
  pcl::PointXYZ z_axis (minor_vector (0) + mass_center (0), minor_vector (1) + mass_center (1), minor_vector (2) + mass_center (2));
  viewer->addLine (center, x_axis, 1.0f, 0.0f, 0.0f, "major eigen vector");
  viewer->addLine (center, y_axis, 0.0f, 1.0f, 0.0f, "middle eigen vector");
  viewer->addLine (center, z_axis, 0.0f, 0.0f, 1.0f, "minor eigen vector");

  Eigen::Vector3f p1 (min_point_OBB.x, min_point_OBB.y, min_point_OBB.z);
  Eigen::Vector3f p2 (min_point_OBB.x, min_point_OBB.y, max_point_OBB.z);
  Eigen::Vector3f p3 (max_point_OBB.x, min_point_OBB.y, max_point_OBB.z);
  Eigen::Vector3f p4 (max_point_OBB.x, min_point_OBB.y, min_point_OBB.z);
  Eigen::Vector3f p5 (min_point_OBB.x, max_point_OBB.y, min_point_OBB.z);
  Eigen::Vector3f p6 (min_point_OBB.x, max_point_OBB.y, max_point_OBB.z);
  Eigen::Vector3f p7 (max_point_OBB.x, max_point_OBB.y, max_point_OBB.z);
  Eigen::Vector3f p8 (max_point_OBB.x, max_point_OBB.y, min_point_OBB.z);

  p1 = rotational_matrix_OBB * p1 + position;
  p2 = rotational_matrix_OBB * p2 + position;
  p3 = rotational_matrix_OBB * p3 + position;
  p4 = rotational_matrix_OBB * p4 + position;
  p5 = rotational_matrix_OBB * p5 + position;
  p6 = rotational_matrix_OBB * p6 + position;
  p7 = rotational_matrix_OBB * p7 + position;
  p8 = rotational_matrix_OBB * p8 + position;

  pcl::PointXYZ pt1 (p1 (0), p1 (1), p1 (2));
  pcl::PointXYZ pt2 (p2 (0), p2 (1), p2 (2));
  pcl::PointXYZ pt3 (p3 (0), p3 (1), p3 (2));
  pcl::PointXYZ pt4 (p4 (0), p4 (1), p4 (2));
  pcl::PointXYZ pt5 (p5 (0), p5 (1), p5 (2));
  pcl::PointXYZ pt6 (p6 (0), p6 (1), p6 (2));
  pcl::PointXYZ pt7 (p7 (0), p7 (1), p7 (2));
  pcl::PointXYZ pt8 (p8 (0), p8 (1), p8 (2));

  viewer->addLine (pt1, pt2, 1.0, 0.0, 0.0, "1 edge");
  viewer->addLine (pt1, pt4, 1.0, 0.0, 0.0, "2 edge");
  viewer->addLine (pt1, pt5, 1.0, 0.0, 0.0, "3 edge");
  viewer->addLine (pt5, pt6, 1.0, 0.0, 0.0, "4 edge");
  viewer->addLine (pt5, pt8, 1.0, 0.0, 0.0, "5 edge");
  viewer->addLine (pt2, pt6, 1.0, 0.0, 0.0, "6 edge");
  viewer->addLine (pt6, pt7, 1.0, 0.0, 0.0, "7 edge");
  viewer->addLine (pt7, pt8, 1.0, 0.0, 0.0, "8 edge");
  viewer->addLine (pt2, pt3, 1.0, 0.0, 0.0, "9 edge");
  viewer->addLine (pt4, pt8, 1.0, 0.0, 0.0, "10 edge");
  viewer->addLine (pt3, pt4, 1.0, 0.0, 0.0, "11 edge");
  viewer->addLine (pt3, pt7, 1.0, 0.0, 0.0, "12 edge");

  while(!viewer->wasStopped())
  {
    viewer->spinOnce (100);
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  }

  return (0);
}

