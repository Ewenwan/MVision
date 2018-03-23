/* 
NARF　
从深度图像(RangeImage)中提取NARF关键点  pcl::NarfKeypoint   
 然后计算narf特征 pcl::NarfDescriptor

1. 边缘提取
对点云而言，场景的边缘代表前景物体和背景物体的分界线。
所以，点云的边缘又分为三种：

前景边缘，背景边缘，阴影边缘。
就是点a 和点b 如果在 rangImage 上是相邻的，然而在三维距离上却很远，那么多半这里就有边缘。

在提取关键点时，
边缘应该作为一个重要的参考依据。
但一定不是唯一的依据。
对于某个物体来说关键点应该是表达了某些特征的点，而不仅仅是边缘点。
所以在设计关键点提取算法时，需要考虑到以下一些因素：
边缘和曲面结构都要考虑进去；
关键点要能重复；
关键点最好落在比较稳定的区域，方便提取法线。

图像的Harris角点算子将图像的关键点定义为角点。
角点也就是物体边缘的交点，
harris算子利用角点在两个方向的灰度协方差矩阵响应都很大，来定义角点。
既然关键点在二维图像中已经被成功定义且使用了，
看来在三维点云中可以沿用二维图像的定义
不过今天要讲的是另外一种思路，简单粗暴，
直接把三维的点云投射成二维的图像不就好了。
这种投射方法叫做range_image.

*/
#include <iostream>//标准输入输出流
#include <boost/thread/thread.hpp>
#include <pcl/range_image/range_image.h>// RangeImage 深度图像
#include <pcl/io/pcd_io.h>//PCL的PCD格式文件的输入输出头文件
#include <pcl/visualization/range_image_visualizer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/range_image_border_extractor.h>
#include <pcl/keypoints/narf_keypoint.h>// narf关键点检测
#include <pcl/features/narf_descriptor.h>// narf特征
#include <pcl/console/parse.h>//解析 命令行 参数

//定义别名
typedef pcl::PointXYZ PointType;
// --------------------
// -----参数　Parameters-----
// --------------------
//参数 全局变量
float angular_resolution = 0.5f;//角坐标分辨率
float support_size = 0.2f;//感兴趣点的尺寸（球面的直径）
pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::CAMERA_FRAME;//坐标框架：相机框架（而不是激光框架）
bool setUnseenToMaxRange = false;//是否将所有不可见的点 看作 最大距离
bool rotation_invariant = true;// 
// --------------
// -----打印帮助信息　Help-----
// --------------
//当用户输入命令行参数-h，打印帮助信息
void 
printUsage (const char* progName)
{
  std::cout << "\n\n用法　Usage: "<<progName<<" [options] <scene.pcd>\n\n"
            << "Options:\n"
            << "-------------------------------------------\n"
            << "-r <float>   角度　angular resolution in degrees (default "<<angular_resolution<<")\n"
            << "-c <int>     坐标系　coordinate frame (default "<< (int)coordinate_frame<<")\n"
            << "-m           Treat all unseen points as maximum range readings\n"
            << "-s <float>   support size for the interest points (diameter of the used sphere - "
            <<                                                     "default "<<support_size<<")\n"
            << "-o <0/1>     switch rotational invariant version of the feature on/off"
            << "-h           this help\n"
            << "\n\n";
}

//void 
//setViewerPose (pcl::visualization::PCLVisualizer& viewer, const Eigen::Affine3f& viewer_pose)
//{
  //Eigen::Vector3f pos_vector = viewer_pose * Eigen::Vector3f (0, 0, 0);
  //Eigen::Vector3f look_at_vector = viewer_pose.rotation () * Eigen::Vector3f (0, 0, 1) + pos_vector;
  //Eigen::Vector3f up_vector = viewer_pose.rotation () * Eigen::Vector3f (0, -1, 0);
  //viewer.setCameraPosition (pos_vector[0], pos_vector[1], pos_vector[2],
                            //look_at_vector[0], look_at_vector[1], look_at_vector[2],
                            //up_vector[0], up_vector[1], up_vector[2]);
//}

// --------------
// -----Main-----
// --------------
int 
main (int argc, char** argv)
{
  // --------------------------------------
  // ----- 解析 命令行 参数 Parse Command Line Arguments-----
  // --------------------------------------
  if (pcl::console::find_argument (argc, argv, "-h") >= 0)//help参数
  {
    printUsage (argv[0]);//程序名
    return 0;
  }
  if (pcl::console::find_argument (argc, argv, "-m") >= 0)
  {
    setUnseenToMaxRange = true;//将所有不可见的点 看作 最大距离
    cout << "Setting unseen values in range image to maximum range readings.\n";
  }
  if (pcl::console::parse (argc, argv, "-o", rotation_invariant) >= 0)
    cout << "Switching rotation invariant feature version "<< (rotation_invariant ? "on" : "off")<<".\n";
  int tmp_coordinate_frame;//坐标框架：相机框架（而不是激光框架）
  if (pcl::console::parse (argc, argv, "-c", tmp_coordinate_frame) >= 0)
  {
    coordinate_frame = pcl::RangeImage::CoordinateFrame (tmp_coordinate_frame);
    cout << "Using coordinate frame "<< (int)coordinate_frame<<".\n";
  }
  // 感兴趣点的尺寸（球面的直径）
  if (pcl::console::parse (argc, argv, "-s", support_size) >= 0)
    cout << "Setting support size to "<<support_size<<".\n";
  // 角坐标分辨率
  if (pcl::console::parse (argc, argv, "-r", angular_resolution) >= 0)
    cout << "Setting angular resolution to "<<angular_resolution<<"deg.\n";
  angular_resolution = pcl::deg2rad (angular_resolution);
  
  // ------------------------------------------------------------------
  // -----Read pcd file or create example point cloud if not given-----
  // ------------------------------------------------------------------
  //读取pcd文件；如果没有指定文件，就创建样本点
  pcl::PointCloud<PointType>::Ptr point_cloud_ptr(new pcl::PointCloud<PointType>);//点云对象指针
  pcl::PointCloud<PointType>& point_cloud = *point_cloud_ptr;//引用　上面点云的别名　常亮指针
  pcl::PointCloud<pcl::PointWithViewpoint> far_ranges;//带视角的点云
  Eigen::Affine3f scene_sensor_pose (Eigen::Affine3f::Identity ());//仿射变换
  //检查参数中是否有pcd格式文件名，返回参数向量中的索引号
  std::vector<int> pcd_filename_indices = pcl::console::parse_file_extension_argument (argc, argv, "pcd");
  if (!pcd_filename_indices.empty())
  {
    std::string filename = argv[pcd_filename_indices[0]];
    if (pcl::io::loadPCDFile (filename, point_cloud) == -1)//如果指定了pcd文件，读取pcd文件
    {
      std::cerr << "Was not able to open file \""<<filename<<"\".\n";
      printUsage (argv[0]);
      return 0;
    }
    //设置传感器的姿势
    scene_sensor_pose = Eigen::Affine3f (Eigen::Translation3f (point_cloud.sensor_origin_[0],
                                                               point_cloud.sensor_origin_[1],
                                                               point_cloud.sensor_origin_[2])) *
                        Eigen::Affine3f (point_cloud.sensor_orientation_);
    //读取远距离文件?
    std::string far_ranges_filename = pcl::getFilenameWithoutExtension (filename)+"_far_ranges.pcd";
    if (pcl::io::loadPCDFile (far_ranges_filename.c_str (), far_ranges) == -1)
      std::cout << "Far ranges file \""<<far_ranges_filename<<"\" does not exists.\n";
  }
  else//没有指定pcd文件，生成点云，并填充它
  {
    setUnseenToMaxRange = true;//将所有不可见的点 看作 最大距离
    cout << "\nNo *.pcd file given => Genarating example point cloud.\n\n";
    for (float x=-0.5f; x<=0.5f; x+=0.01f)
    {
      for (float y=-0.5f; y<=0.5f; y+=0.01f)
      {
        PointType point;  point.x = x;  point.y = y;  point.z = 2.0f - y;
        point_cloud.points.push_back (point);//设置点云中点的坐标
      }
    }
    point_cloud.width = (int) point_cloud.points.size ();  
    point_cloud.height = 1;
  }
  
  // -----------------------------------------------
  // -----Create RangeImage from the PointCloud-----
  // -----------------------------------------------
  // ======从点云数据，创建深度图像=====================
  // 直接把三维的点云投射成二维的图像
  float noise_level = 0.0;
//noise level表示的是容差率，因为1°X1°的空间内很可能不止一个点，
//noise level = 0则表示去最近点的距离作为像素值，如果=0.05则表示在最近点及其后5cm范围内求个平均距离
//minRange表示深度最小值，如果=0则表示取1°X1°的空间内最远点，近的都忽略
  float min_range = 0.0f;
//bordersieze表示图像周边点 
  int border_size = 1;
  boost::shared_ptr<pcl::RangeImage> range_image_ptr (new pcl::RangeImage);//创建RangeImage对象（智能指针）
  pcl::RangeImage& range_image = *range_image_ptr; //RangeImage的引用  
  //从点云创建深度图像
//rangeImage也是PCL的基本数据结构
//pcl::RangeImage rangeImage;
// 球坐标系
//角分辨率
//float angularResolution = (float) (  1.0f * (M_PI/180.0f));  //   1.0 degree in radians　弧度
//phi可以取360°
//  float maxAngleWidth     = (float) (360.0f * (M_PI/180.0f));  // 360.0 degree in radians
//a取180°
//  float maxAngleHeight    = (float) (180.0f * (M_PI/180.0f));  // 180.0 degree in radians
//半圆扫一圈就是整个图像了
  range_image.createFromPointCloud (point_cloud, angular_resolution, pcl::deg2rad (360.0f), pcl::deg2rad (180.0f),
                                   scene_sensor_pose, coordinate_frame, noise_level, min_range, border_size);
  range_image.integrateFarRanges (far_ranges);//整合远距离点云
  if (setUnseenToMaxRange)
    range_image.setUnseenToMaxRange ();
  
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  // 3D点云显示
  pcl::visualization::PCLVisualizer viewer ("3D Viewer");
  viewer.setBackgroundColor (1, 1, 1);//背景颜色　白色
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointWithRange> range_image_color_handler (range_image_ptr, 0, 0, 0);
  viewer.addPointCloud (range_image_ptr, range_image_color_handler, "range image");//添加点云
  viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "range image");
  //viewer.addCoordinateSystem (1.0f, "global");
  //PointCloudColorHandlerCustom<PointType> point_cloud_color_handler (point_cloud_ptr, 150, 150, 150);
  //viewer.addPointCloud (point_cloud_ptr, point_cloud_color_handler, "original point cloud");
  viewer.initCameraParameters ();
  //setViewerPose (viewer, range_image.getTransformationToWorldSystem ());
  
  // --------------------------
  // -----Show range image-----
  // --------------------------
  //显示深度图像（平面图）
  pcl::visualization::RangeImageVisualizer range_image_widget ("Range image");
  range_image_widget.showRangeImage (range_image);
  
  // --------------------------------
  // -----Extract NARF keypoints-----
  // --------------------------------
  // ==================提取NARF关键点=======================
  pcl::RangeImageBorderExtractor range_image_border_extractor;//创建深度图像的边界提取器，用于提取NARF关键点
  pcl::NarfKeypoint narf_keypoint_detector (&range_image_border_extractor);//创建NARF对象
  narf_keypoint_detector.setRangeImage (&range_image);//设置点云对应的深度图
  narf_keypoint_detector.getParameters ().support_size = support_size;// 感兴趣点的尺寸（球面的直径）
  //narf_keypoint_detector.getParameters ().add_points_on_straight_edges = true;
  //narf_keypoint_detector.getParameters ().distance_for_additional_points = 0.5;
  
  pcl::PointCloud<int> keypoint_indices;//用于存储关键点的索引 PointCloud<int>
  narf_keypoint_detector.compute (keypoint_indices);//计算NARF关键
  std::cout << "Found找到关键点： "<<keypoint_indices.points.size ()<<" key points.\n";

  // ----------------------------------------------
  // -----Show keypoints in range image widget-----
  // ----------------------------------------------
 //在range_image_widget中显示关键点
  //for (size_t i=0; i<keypoint_indices.points.size (); ++i)
    //range_image_widget.markPoint (keypoint_indices.points[i]%range_image.width,
                                  //keypoint_indices.points[i]/range_image.width);
  
  // -------------------------------------
  // -----Show keypoints in 3D viewer-----
  // -------------------------------------
  //在3D图形窗口中显示关键点
  pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_ptr (new pcl::PointCloud<pcl::PointXYZ>);//创建关键点指针
  pcl::PointCloud<pcl::PointXYZ>& keypoints = *keypoints_ptr;//引用
  keypoints.points.resize (keypoint_indices.points.size ());//初始化大小
  for (size_t i=0; i<keypoint_indices.points.size (); ++i)//按照索引获得　关键点
    keypoints.points[i].getVector3fMap () = range_image.points[keypoint_indices.points[i]].getVector3fMap ();

  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> keypoints_color_handler (keypoints_ptr, 255, 0, 0);//红色
  viewer.addPointCloud<pcl::PointXYZ> (keypoints_ptr, keypoints_color_handler, "keypoints");//添加显示关键点
  viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "keypoints");
  //渲染属性，可视化工具，3维数据， 其中PCL_VISUALIZER_POINT_SIZE表示设置点的大小为7

   // ------------------------------------------------------
  //========================提取 NARF 特征 ====================
  // ------------------------------------------------------
  std::vector<int> keypoint_indices2;//用于存储关键点的索引 vector<int>  
  keypoint_indices2.resize (keypoint_indices.points.size ());
  for (unsigned int i=0; i<keypoint_indices.size (); ++i) // This step is necessary to get the right vector type
    keypoint_indices2[i] = keypoint_indices.points[i];//narf关键点 索引
  pcl::NarfDescriptor narf_descriptor (&range_image, &keypoint_indices2);//narf特征描述子
  narf_descriptor.getParameters().support_size = support_size;
  narf_descriptor.getParameters().rotation_invariant = rotation_invariant;
  pcl::PointCloud<pcl::Narf36> narf_descriptors;
  narf_descriptor.compute (narf_descriptors);
  cout << "Extracted "<<narf_descriptors.size ()<<" descriptors for "
                      <<keypoint_indices.points.size ()<< " keypoints.\n";
 
  //--------------------
  // -----Main loop-----
  //--------------------
  while (!viewer.wasStopped ())
  {
    range_image_widget.spinOnce ();  // process GUI events　　 处理 GUI事件
    viewer.spinOnce ();
    pcl_sleep(0.01);
  }
}


