/*
Range Images范围图像，意思是通过点云获得特定角度的观察图像.
以特定角度投影点云到二维平面上，根据距离设定像素值，显示图像。

在3D视窗中以点云形式进行可视化（深度图像来自于点云），
另一种是将深度值映射为颜色，从而以彩色图像方式可视化深度图像， 

深度图 到 点云
点云 范围图像 到 深度图

怎样可视化深度图像

本小节讲解如何可视化深度图像的两种方法，
在3D视窗中以点云形式进行可视化（深度图像来源于点云），
另一种是，将深度值映射为颜色，从而以彩色图像方式可视化深度图像。

学习如何从点云和给定的传感器位置来创建深度图像，
下面的程序，首先是生成一个矩形点云，然后基于该点云创建深度图像。
在3D视窗中以点云形式进行可视化（深度图像来自于点云），另一种是将深度值映射为颜色，从而以彩色图像方式可视化深度图像， 

*/
#include <iostream>
#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/range_image/range_image.h>             //关于深度图像的头文件
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/range_image_visualizer.h>//深度图可视化的头文件
#include <pcl/visualization/pcl_visualizer.h>        //PCL可视化的头文件
#include <pcl/console/parse.h>
 
typedef pcl::PointXYZ PointType;
//参数
float angular_resolution_x = 0.5f,//angular_resolution为模拟的深度传感器的角度分辨率，即深度图像中一个像素对应的角度大小
      angular_resolution_y = angular_resolution_x;
pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::CAMERA_FRAME;//深度图像遵循坐标系统
bool live_update = false;
//命令帮助提示
void 
printUsage (const char* progName)
{
  std::cout << "\n\nUsage: "<<progName<<" [options] <scene.pcd>\n\n"
            << "Options:\n"
            << "-------------------------------------------\n"
            << "-rx <float>  angular resolution in degrees (default "<<angular_resolution_x<<")\n"
            << "-ry <float>  angular resolution in degrees (default "<<angular_resolution_y<<")\n"
            << "-c <int>     coordinate frame (default "<< (int)coordinate_frame<<")\n"
            << "-l           live update - update the range image according to the selected view in the 3D viewer.\n"
            << "-h           this help\n"
            << "\n\n";
}

void 
setViewerPose (pcl::visualization::PCLVisualizer& viewer, const Eigen::Affine3f& viewer_pose)
{
  Eigen::Vector3f pos_vector = viewer_pose * Eigen::Vector3f(0, 0, 0);
  Eigen::Vector3f look_at_vector = viewer_pose.rotation () * Eigen::Vector3f(0, 0, 1) + pos_vector;
  Eigen::Vector3f up_vector = viewer_pose.rotation () * Eigen::Vector3f(0, -1, 0);
  viewer.setCameraPosition (pos_vector[0], pos_vector[1], pos_vector[2],
                            look_at_vector[0], look_at_vector[1], look_at_vector[2],
                            up_vector[0], up_vector[1], up_vector[2]);
}

//主函数
int 
main (int argc, char** argv)
{
  //输入命令分析
  if (pcl::console::find_argument (argc, argv, "-h") >= 0)
  {
    printUsage (argv[0]);
    return 0;
  }
  if (pcl::console::find_argument (argc, argv, "-l") >= 0)
  {
    live_update = true;
    std::cout << "Live update is on.\n";
  }
  if (pcl::console::parse (argc, argv, "-rx", angular_resolution_x) >= 0)
    std::cout << "Setting angular resolution in x-direction to "<<angular_resolution_x<<"deg.\n";
  if (pcl::console::parse (argc, argv, "-ry", angular_resolution_y) >= 0)
    std::cout << "Setting angular resolution in y-direction to "<<angular_resolution_y<<"deg.\n";
  int tmp_coordinate_frame;
  if (pcl::console::parse (argc, argv, "-c", tmp_coordinate_frame) >= 0)
  {
    coordinate_frame = pcl::RangeImage::CoordinateFrame (tmp_coordinate_frame);
    std::cout << "Using coordinate frame "<< (int)coordinate_frame<<".\n";
  }
  angular_resolution_x = pcl::deg2rad (angular_resolution_x);
  angular_resolution_y = pcl::deg2rad (angular_resolution_y);
  
  //读取点云PCD文件  如果没有输入PCD文件就生成一个点云
  pcl::PointCloud<PointType>::Ptr point_cloud_ptr (new pcl::PointCloud<PointType>);
  pcl::PointCloud<PointType>& point_cloud = *point_cloud_ptr;
  Eigen::Affine3f scene_sensor_pose (Eigen::Affine3f::Identity ());   //申明传感器的位置是一个4*4的仿射变换
  std::vector<int> pcd_filename_indices = pcl::console::parse_file_extension_argument (argc, argv, "pcd");
  if (!pcd_filename_indices.empty ())
  {
    std::string filename = argv[pcd_filename_indices[0]];
    if (pcl::io::loadPCDFile (filename, point_cloud) == -1)
    {
      std::cout << "Was not able to open file \""<<filename<<"\".\n";
      printUsage (argv[0]);
      return 0;
    }
   //给传感器的位姿赋值  就是获取点云的传感器的的平移与旋转的向量
    scene_sensor_pose = Eigen::Affine3f (Eigen::Translation3f (point_cloud.sensor_origin_[0],
                                                             point_cloud.sensor_origin_[1],
                                                             point_cloud.sensor_origin_[2])) *
                        Eigen::Affine3f (point_cloud.sensor_orientation_);
  }
  else
  {  //如果没有给点云，则我们要自己生成点云
    std::cout << "\nNo *.pcd file given => Genarating example point cloud.\n\n";
    for (float x=-0.5f; x<=0.5f; x+=0.01f)
    {
      for (float y=-0.5f; y<=0.5f; y+=0.01f)
      {
        PointType point;  point.x = x;  point.y = y;  point.z = 2.0f - y;
        point_cloud.points.push_back (point);
      }
    }
    point_cloud.width = (int) point_cloud.points.size ();  point_cloud.height = 1;
  }
  
  // -----从创建的点云中获取深度图--//
  //设置基本参数
  float noise_level = 0.0;
  float min_range = 0.0f;
  int border_size = 1;
  boost::shared_ptr<pcl::RangeImage> range_image_ptr(new pcl::RangeImage);
  pcl::RangeImage& range_image = *range_image_ptr;  
/*
 关于range_image.createFromPointCloud（）参数的解释 （涉及的角度都为弧度为单位） ：
   point_cloud为创建深度图像所需要的点云
  angular_resolution_x深度传感器X方向的角度分辨率
  angular_resolution_y深度传感器Y方向的角度分辨率
   pcl::deg2rad (360.0f)深度传感器的水平最大采样角度
   pcl::deg2rad (180.0f)垂直最大采样角度
   scene_sensor_pose设置的模拟传感器的位姿是一个仿射变换矩阵，默认为4*4的单位矩阵变换
   coordinate_frame定义按照那种坐标系统的习惯  默认为CAMERA_FRAME
   noise_level  获取深度图像深度时，邻近点对查询点距离值的影响水平
   min_range 设置最小的获取距离，小于最小的获取距离的位置为传感器的盲区
   border_size  设置获取深度图像边缘的宽度 默认为0 
*/ 
  range_image.createFromPointCloud (point_cloud, angular_resolution_x, angular_resolution_y,pcl::deg2rad (360.0f), pcl::deg2rad (180.0f),scene_sensor_pose, coordinate_frame, noise_level, min_range, border_size);
  
  //可视化点云
  pcl::visualization::PCLVisualizer viewer ("3D Viewer");
  viewer.setBackgroundColor (1, 1, 1);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointWithRange> range_image_color_handler (range_image_ptr, 0, 0, 0);
  viewer.addPointCloud (range_image_ptr, range_image_color_handler, "range image");
  viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "range image");
  //viewer.addCoordinateSystem (1.0f, "global");
  //PointCloudColorHandlerCustom<PointType> point_cloud_color_handler (point_cloud_ptr, 150, 150, 150);
  //viewer.addPointCloud (point_cloud_ptr, point_cloud_color_handler, "original point cloud");
  viewer.initCameraParameters ();
  //range_image.getTransformationToWorldSystem ()的作用是获取从深度图像坐标系统（应该就是传感器的坐标）转换为世界坐标系统的转换矩阵
  setViewerPose(viewer, range_image.getTransformationToWorldSystem ());  //设置视点的位置
  
  //可视化深度图
  pcl::visualization::RangeImageVisualizer range_image_widget ("Range image");
  range_image_widget.showRangeImage (range_image);
  
  while (!viewer.wasStopped ())
  {
    range_image_widget.spinOnce ();
    viewer.spinOnce ();
    pcl_sleep (0.01);
    
    if (live_update)
    {
      //如果选择的是——l的参数说明就是要根据自己选择的视点来创建深度图。
     // live update - update the range image according to the selected view in the 3D viewer.
      scene_sensor_pose = viewer.getViewerPose();
      range_image.createFromPointCloud (point_cloud, angular_resolution_x, angular_resolution_y,
                                        pcl::deg2rad (360.0f), pcl::deg2rad (180.0f),
                                        scene_sensor_pose, pcl::RangeImage::LASER_FRAME, noise_level, min_range, border_size);
      range_image_widget.showRangeImage (range_image);
    }
  }
}
