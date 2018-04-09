/*
圆柱体分割　依据法线信息分割
先分割平面　得到平面上的点云
在平面上的点云中　分割圆柱体点云
实现圆柱体模型的分割：
采用随机采样一致性估计从带有噪声的点云中提取一个圆柱体模型。
*/
#include <pcl/ModelCoefficients.h>//模型系数
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>//按索引提取点云
#include <pcl/filters/passthrough.h>//　直通滤波器
#include <pcl/features/normal_3d.h>//　法线特征
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>//随机采用分割
#include <pcl/visualization/pcl_visualizer.h> // 可视化

typedef pcl::PointXYZ PointT;
int main (int argc, char** argv)
{
  // 所需要的对象　All the objects needed
  pcl::PCDReader reader;//PCD文件读取对象
  pcl::PassThrough<PointT> pass;//直通滤波对象
  pcl::NormalEstimation<PointT, pcl::Normal> ne;//法线估计对象
  pcl::SACSegmentationFromNormals<PointT, pcl::Normal> seg;//依据法线　分割对象
  pcl::PCDWriter writer;//PCD文件写对象
  pcl::ExtractIndices<PointT> extract;//点　提取对象
  pcl::ExtractIndices<pcl::Normal> extract_normals;    ///点法线特征　提取对象
  pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT> ());

  // 数据对象　Datasets
  pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
  pcl::PointCloud<PointT>::Ptr cloud_filtered (new pcl::PointCloud<PointT>);
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);//法线特征
  pcl::PointCloud<PointT>::Ptr cloud_filtered2 (new pcl::PointCloud<PointT>);
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals2 (new pcl::PointCloud<pcl::Normal>);//法线特征
  pcl::ModelCoefficients::Ptr coefficients_plane (new pcl::ModelCoefficients), coefficients_cylinder (new pcl::ModelCoefficients);//模型系数
  pcl::PointIndices::Ptr inliers_plane (new pcl::PointIndices), inliers_cylinder (new pcl::PointIndices);//内点索引

  // 读取桌面场景点云　Read in the cloud data
  reader.read ("../../surface/table_scene_mug_stereo_textured.pcd", *cloud);
  std::cerr << "PointCloud has: " << cloud->points.size () << " data points." << std::endl;

  // 直通滤波，将Z轴不在（0，1.5）范围的点过滤掉，将剩余的点存储到cloud_filtered对象中
  pass.setInputCloud (cloud);
  pass.setFilterFieldName ("z");// Z轴
  pass.setFilterLimits (0, 1.5);//　范围
  pass.filter (*cloud_filtered);
  std::cerr << "PointCloud after filtering has: " << cloud_filtered->points.size () << " data points." << std::endl;

  // 过滤后的点云进行法线估计，为后续进行基于法线的分割准备数据
  ne.setSearchMethod (tree);
  ne.setInputCloud (cloud_filtered);
  ne.setKSearch (50);
  ne.compute (*cloud_normals);//计算法线特征

  // Create the segmentation object for the planar model and set all the parameters
  // 同时优化模型系数
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_NORMAL_PLANE);//平面模型
  seg.setNormalDistanceWeight (0.1);  // 法线信息权重
  seg.setMethodType (pcl::SAC_RANSAC);//随机采样一致性算法
  seg.setMaxIterations (100);         //最大迭代次数
  seg.setDistanceThreshold (0.03);    //设置内点到模型的距离允许最大值
  seg.setInputCloud (cloud_filtered); //输入点云
  seg.setInputNormals (cloud_normals);//输入法线特征
  //获取平面模型的系数和处在平面的内点
  seg.segment (*inliers_plane, *coefficients_plane);//分割　得到内点索引　和模型系数
  std::cerr << "Plane coefficients: " << *coefficients_plane << std::endl;

  // 从点云中抽取分割　处在平面上的点集　内点
  extract.setInputCloud (cloud_filtered);
  extract.setIndices (inliers_plane);
  extract.setNegative (false);
  // 存储分割得到的平面上的点到点云文件
  pcl::PointCloud<PointT>::Ptr cloud_plane (new pcl::PointCloud<PointT> ());
  extract.filter (*cloud_plane);//平面上的点云
  std::cerr << "PointCloud representing the planar component: " << cloud_plane->points.size () << " data points." << std::endl;
  writer.write ("table_scene_mug_stereo_textured_plane.pcd", *cloud_plane, false);

  // 提取　得到平面上的点云　（外点）　以及其法线特征
  extract.setNegative (true);//除去内点
  extract.filter (*cloud_filtered2);//得到外点
  extract_normals.setNegative (true);
  extract_normals.setInputCloud (cloud_normals);
  extract_normals.setIndices (inliers_plane);
  extract_normals.filter (*cloud_normals2);//提取外点对应的法线特征

  // 在平面上的点云　分割　圆柱体　
  seg.setOptimizeCoefficients (true);   //设置对估计模型优化
  seg.setModelType (pcl::SACMODEL_CYLINDER);//设置分割模型为圆柱形
  seg.setMethodType (pcl::SAC_RANSAC);      //参数估计方法　随机采样一致性算法
  seg.setNormalDistanceWeight (0.1);        //设置表面法线权重系数
  seg.setMaxIterations (10000);             //设置迭代的最大次数10000
  seg.setDistanceThreshold (0.05);          //设置内点到模型的距离允许最大值
  seg.setRadiusLimits (0, 0.1);             //设置估计出的圆柱模型的半径的范围
  seg.setInputCloud (cloud_filtered2);      //输入点云
  seg.setInputNormals (cloud_normals2);     //输入点云对应的法线特征

  // 获取符合圆柱体模型的内点　和　对应的系数
  seg.segment (*inliers_cylinder, *coefficients_cylinder);
  std::cerr << "Cylinder coefficients: " << *coefficients_cylinder << std::endl;

  // 写文件到
  extract.setInputCloud (cloud_filtered2);
  extract.setIndices (inliers_cylinder);
  extract.setNegative (false);//得到圆柱体点云
  pcl::PointCloud<PointT>::Ptr cloud_cylinder (new pcl::PointCloud<PointT> ());
  extract.filter (*cloud_cylinder);
  if (cloud_cylinder->points.empty ()) 
    std::cerr << "Can't find the cylindrical component." << std::endl;
  else
  {
      std::cerr << "PointCloud representing the cylindrical component: " << cloud_cylinder->points.size () << " data points." << std::endl;
      writer.write ("table_scene_mug_stereo_textured_cylinder.pcd", *cloud_cylinder, false);
  }

// 3D点云显示 绿色
  pcl::visualization::PCLVisualizer viewer ("3D Viewer");
  viewer.setBackgroundColor (255, 255, 255);//背景颜色　白色
  //viewer.addCoordinateSystem (1.0);
  viewer.initCameraParameters ();
//平面上的点云　红色
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_plane_handler(cloud_plane, 255, 0, 0);
  viewer.addPointCloud (cloud_plane, cloud_plane_handler, "plan point");//添加点云
  viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "plan point");
//  圆柱体模型的内点　绿色
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_cylinder_handler(cloud_cylinder, 0, 255, 0);
  viewer.addPointCloud (cloud_cylinder, cloud_cylinder_handler, "cylinder point");//添加点云
  viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cylinder point");

    while (!viewer.wasStopped()){
        viewer.spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }

  return (0);
}

