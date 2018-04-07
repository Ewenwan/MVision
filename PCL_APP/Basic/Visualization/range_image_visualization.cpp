/*可视化深度图像
在3D视窗中以点云形式进行可视化（深度图像来自于点云），
另一种是将深度值映射为颜色，从而以彩色图像方式可视化深度图像， 

*/

#include <iostream>

#include <boost/thread/thread.hpp>

#include <pcl/common/common_headers.h>
#include <pcl/range_image/range_image.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/range_image_visualizer.h>// 范围图像可视化
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>

typedef pcl::PointXYZ PointType;

// --------------------
// -----参数-----
// --------------------
float angular_resolution_x = 0.5f,
      angular_resolution_y = angular_resolution_x;
pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::CAMERA_FRAME;
bool live_update = false;

// --------------
// -----帮助信息-----
// --------------
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

/*
 设置视角位置
*/
void 
setViewerPose (pcl::visualization::PCLVisualizer& viewer, const Eigen::Affine3f& viewer_pose)   //设置视角位置
{
  Eigen::Vector3f pos_vector = viewer_pose * Eigen::Vector3f(0, 0, 0);   //eigen
  Eigen::Vector3f look_at_vector = viewer_pose.rotation () * Eigen::Vector3f(0, 0, 1) + pos_vector;
  Eigen::Vector3f up_vector = viewer_pose.rotation () * Eigen::Vector3f(0, -1, 0);
  viewer.setCameraPosition (pos_vector[0], pos_vector[1], pos_vector[2],
                            look_at_vector[0], look_at_vector[1], look_at_vector[2],
                            up_vector[0], up_vector[1], up_vector[2]);
}

// --------------
// -----Main-----
// --------------
int 
main (int argc, char** argv)
{
  // --------------------------------------
  // -----命令行参数解析-----
  // --------------------------------------
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
  
  // ------------------------------------------------------------------
  // -----读取pcd文件-----
  // ------------------------------------------------------------------
  pcl::PointCloud<PointType>::Ptr point_cloud_ptr (new pcl::PointCloud<PointType>);
  pcl::PointCloud<PointType>& point_cloud = *point_cloud_ptr;
  Eigen::Affine3f scene_sensor_pose (Eigen::Affine3f::Identity ());
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
    scene_sensor_pose = Eigen::Affine3f (Eigen::Translation3f (point_cloud.sensor_origin_[0],
                                                             point_cloud.sensor_origin_[1],
                                                             point_cloud.sensor_origin_[2])) *
                        Eigen::Affine3f (point_cloud.sensor_orientation_);
  }
  else//生成 点云
  {
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
  
  // -----------------------------------------------
  // -----从点云中生成范围图像 （深度图像 ） ----
  // -----------------------------------------------
  float noise_level = 0.0;
  float min_range = 0.0f;
  int border_size = 1;
  boost::shared_ptr<pcl::RangeImage> range_image_ptr(new pcl::RangeImage);
  pcl::RangeImage& range_image = *range_image_ptr;   
  range_image.createFromPointCloud (point_cloud, angular_resolution_x, angular_resolution_y,
                                    pcl::deg2rad (360.0f), pcl::deg2rad (180.0f),
                                    scene_sensor_pose, coordinate_frame, noise_level, min_range, border_size);
  
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  /*****************************************************************************************
   创建3D视窗对象，将背景颜色设置为白色，添加黑色的，点云大小为1的深度图像（点云），并使用Main函数
    上面定义的setViewerPose函数设置深度图像的视点参数，被注释的部分用于添加爱坐标系，并对原始点云进行可视化
   *****************************************************************************************/
  pcl::visualization::PCLVisualizer viewer ("3D Viewer");     //定义初始化可视化对象
  viewer.setBackgroundColor (1, 1, 1);                         //背景设置为白色
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointWithRange> range_image_color_handler (range_image_ptr, 0, 0, 0); //设置自定义颜色
  viewer.addPointCloud (range_image_ptr, range_image_color_handler, "range image");
  viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "range image");
  //viewer.addCoordinateSystem (1.0f, "global");
  //PointCloudColorHandlerCustom<PointType> point_cloud_color_handler (point_cloud_ptr, 150, 150, 150);
  //viewer.addPointCloud (point_cloud_ptr, point_cloud_color_handler, "original point cloud");
  viewer.initCameraParameters ();
  setViewerPose(viewer, range_image.getTransformationToWorldSystem ());
  
  // --------------------------
  // -----Show range image-----
  // --------------------------
   //用以图像的方式可视化深度图像，图像的颜色取决于深度值
  pcl::visualization::RangeImageVisualizer range_image_widget ("Range image");
  range_image_widget.showRangeImage (range_image);      //图像可视化方式显示深度图像
  
  //--------------------
  // -----Main loop-----
  //--------------------
  while (!viewer.wasStopped ())   //启动主循环以保证可视化代码的有效性，直到可视化窗口关闭
  {
    range_image_widget.spinOnce ();   //用于处理深度图像可视化类的当前事件
    viewer.spinOnce ();              //用于处理3D窗口当前的事件此外还可以随时更新2D深度图像，以响应可视化窗口中的当前视角，这通过命令行-1来激活
    pcl_sleep (0.01);
    
  //首先从窗口中得到当前的观察位置，然后创建对应视角的深度图像，并在图像显示插件中显示
    if (live_update)   
    {
      scene_sensor_pose = viewer.getViewerPose();
      range_image.createFromPointCloud (point_cloud, angular_resolution_x, angular_resolution_y,
                                        pcl::deg2rad (360.0f), pcl::deg2rad (180.0f),
                                        scene_sensor_pose, pcl::RangeImage::LASER_FRAME, noise_level, min_range, border_size);
      range_image_widget.showRangeImage (range_image);
    }
  }
}
