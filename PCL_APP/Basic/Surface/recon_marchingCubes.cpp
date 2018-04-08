/*
三维重构之移动立方体算法

*/
#include <pcl/point_types.h>//点的类型的头文件
#include <pcl/io/pcd_io.h>//点云文件IO（pcd文件和ply文件）
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>//kd树 快速搜索
#include <pcl/features/normal_3d.h>//　法线特征
#include <pcl/surface/marching_cubes_hoppe.h>// 移动立方体算法
#include <pcl/surface/marching_cubes_rbf.h>
#include <pcl/surface/gp3.h>// 贪婪投影三角化算法
#include <pcl/visualization/pcl_visualizer.h>//可视化
#include <boost/thread/thread.hpp>//多线程
#include <fstream>// std
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <string>

int main (int argc, char** argv)
{
    if(argc<2){ 
	  std::cout << "need ply/pcd file. " << std::endl;
          return -1;
	}
    // 确定文件格式
    char tmpStr[100];
    strcpy(tmpStr,argv[1]);
    char* pext = strrchr(tmpStr, '.');
    std::string extply("ply");
    std::string extpcd("pcd");
    if(pext){
        *pext='\0';
        pext++;
    }
    std::string ext(pext);
    //如果不支持文件格式，退出程序
    if (!((ext == extply)||(ext == extpcd))){
        std::cout << "文件格式不支持!" << std::endl;
        std::cout << "支持文件格式：*.pcd和*.ply！" << std::endl;
        return(-1);
    }

    //根据文件格式选择输入方式
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>) ; //创建点云对象指针，用于存储输入
    if (ext == extply){
        if (pcl::io::loadPLYFile(argv[1] , *cloud) == -1){
            PCL_ERROR("Could not read ply file!\n") ;
            return -1;
        }
    }
    else{
        if (pcl::io::loadPCDFile(argv[1] , *cloud) == -1){
            PCL_ERROR("Could not read pcd file!\n") ;
            return -1;
        }
    }

  // 估计法向量 x,y,x + 法向量　+　曲率
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
  pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud(cloud);
  n.setInputCloud(cloud);
  n.setSearchMethod(tree);
  n.setKSearch(20);//20个邻居
  n.compute (*normals); //计算法线，结果存储在normals中
  //* normals 不能同时包含点的法向量和表面的曲率

  //将点云和法线放到一起
  pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal>);
  pcl::concatenateFields (*cloud, *normals, *cloud_with_normals);
  //* cloud_with_normals = cloud + normals


  //创建搜索树
  pcl::search::KdTree<pcl::PointNormal>::Ptr tree2 (new pcl::search::KdTree<pcl::PointNormal>);
  tree2->setInputCloud (cloud_with_normals);

  //初始化 移动立方体算法 MarchingCubes对象，并设置参数
    pcl::MarchingCubes<pcl::PointNormal> *mc;
    mc = new pcl::MarchingCubesHoppe<pcl::PointNormal> ();
    /*
  if (hoppe_or_rbf == 0)
    mc = new pcl::MarchingCubesHoppe<pcl::PointNormal> ();
  else
  {
    mc = new pcl::MarchingCubesRBF<pcl::PointNormal> ();
    (reinterpret_cast<pcl::MarchingCubesRBF<pcl::PointNormal>*> (mc))->setOffSurfaceDisplacement (off_surface_displacement);
  }
    */

    //创建多变形网格，用于存储结果
  pcl::PolygonMesh mesh;

  //设置MarchingCubes对象的参数
  mc->setIsoLevel (0.0f);
  mc->setGridResolution (50, 50, 50);
  mc->setPercentageExtendGrid (0.0f);

  //设置搜索方法
  mc->setInputCloud (cloud_with_normals);

  //执行重构，结果保存在mesh中
  mc->reconstruct (mesh);

  //保存网格图
  pcl::io::savePLYFile("result2.ply", mesh);

  // 显示结果图
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0); //设置背景
  viewer->addPolygonMesh(mesh,"my"); //设置显示的网格
  //viewer->addCoordinateSystem (1.0); //设置坐标系
  viewer->initCameraParameters ();
  while (!viewer->wasStopped ()){
    viewer->spinOnce (100);
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  }

  return (0);
}
