/*

使用积分图计算一个有序点云的法线，注意该方法只适用于有序点云!!

点云表面法线特征
直接从点云数据集中近似推断表面法线。
下载桌子点云数据
https://raw.github.com/PointCloudLibrary/data/master/tutorials/table_scene_lms400.pcd

表面法线是几何体表面的重要属性，在很多领域都有大量应用，
例如：在进行光照渲染时产生符合可视习惯的效果时需要表面法线信息才能正常进行，
对于一个已知的几何体表面，根据垂直于点表面的矢量，因此推断表面某一点的法线方向通常比较简单。
积分图像估计 点云表面法线

*/
#include <iostream>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>//点云文件pcd 读写
#include <pcl/visualization/cloud_viewer.h>//点云可视化
//#include <pcl/features/normal_3d.h>//法线特征
#include <pcl/features/integral_image_normal.h>

using namespace std;
// 别名
typedef pcl::PointCloud<pcl::PointXYZ>  Cloud;

typedef pcl::PointXYZ PointType;

int
 main (int argc, char** argv)
{
	// 定义　点云对象　指针
	Cloud::Ptr cloud_ptr (new Cloud);

	// 读取点云文件　填充点云对象
	pcl::PCDReader reader;
	reader.read( "../../Filtering/table_scene_lms400.pcd", *cloud_ptr);//非有序点云不可用
	if(cloud_ptr==NULL) { cout << "pcd file read err" << endl; return -1;}
	cout << "PointCLoud size() " << cloud_ptr->width * cloud_ptr->height
	<< " data points ( " << pcl::getFieldsList (*cloud_ptr) << "." << endl;

	// 估计的法线 normals
	pcl::PointCloud<pcl::Normal>::Ptr normals_ptr (new pcl::PointCloud<pcl::Normal>);
	pcl::PointCloud<pcl::Normal>& normals = *normals_ptr;
        // 积分图像方法 估计点云表面法线 
	pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
/*
预测方法
enum NormalEstimationMethod
{
  COVARIANCE_MATRIX, 从最近邻的协方差矩阵创建了9个积分图去计算一个点的法线
  AVERAGE_3D_GRADIENT, 创建了6个积分图去计算3D梯度里面竖直和水平方向的光滑部分，同时利用两个梯度的卷积来计算法线。
  AVERAGE_DEPTH_CHANGE 造了一个单一的积分图，从平均深度的变化中来计算法线。
};
*/ 
	ne.setNormalEstimationMethod (ne.AVERAGE_3D_GRADIENT);
	ne.setMaxDepthChangeFactor(0.02f);
	ne.setNormalSmoothingSize(10.0f);
	ne.setInputCloud(cloud_ptr);
	ne.compute(normals);

	// 点云+法线 可视化
	pcl::visualization::PCLVisualizer viewer("pcd　viewer");// 显示窗口的名字
	viewer.setBackgroundColor(0.0, 0.0, 0.0);//背景黑色
	//viewer.setBackgroundColor (1, 1, 1);//白色
	viewer.addCoordinateSystem (1.0f, "global");//坐标系
	viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud_ptr, normals_ptr);
	//pcl::visualization::PointCloudColorHandlerCustom<PointType> cloud_color_handler (cloud_ptr, 1, 0, 0);//红色
	//viewer.addPointCloud (cloud_ptr, cloud_color_handler, "original point cloud");

	while (!viewer.wasStopped())
	{
	viewer.spinOnce ();
	pcl_sleep(0.05);
	}

	return (0);
}
