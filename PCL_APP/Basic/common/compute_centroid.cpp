/*
计算点云重心

 点云的重心是一个点坐标，计算出云中所有点的平均值。
你可以说它是“质量中心”，它对于某些算法有多种用途。
如果你想计算一个聚集的物体的实际重心，
记住，传感器没有检索到从相机中相反的一面，
就像被前面板遮挡的背面，或者里面的。
只有面对相机表面的一部分。

*/
#include <pcl/io/pcd_io.h>
#include <pcl/common/centroid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <iostream>

int
main(int argc, char** argv)
{
    // 创建点云的对象
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);

     
    // 读取点云
    if (pcl::io::loadPCDFile<pcl::PointXYZ>("../../Filtering/table_scene_lms400.pcd", *cloud_ptr) != 0)
    {
        return -1;
    }

// 3D点云显示
  pcl::visualization::PCLVisualizer viewer ("3D Viewer");
  viewer.setBackgroundColor (1, 1, 1);//背景颜色　白色
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(cloud_ptr, 1.0, 1.0, 1.0);
  viewer.addPointCloud (cloud_ptr, color_handler, "raw point");//添加点云
  viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "raw point");
  //viewer.addCoordinateSystem (1.0);
  viewer.initCameraParameters ();

    // 创建存储点云重心的对象
    Eigen::Vector4f centroid;//齐次表示 
    pcl::compute3DCentroid(*cloud_ptr, centroid);
    std::cout << "The XYZ coordinates of the centroid are: ("
              << centroid[0] << ", "
              << centroid[1] << ", "
              << centroid[2] << ")." << std::endl;

   pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cet_ptr(new pcl::PointCloud<pcl::PointXYZ>);
 //随机创建点云并打印出来
   cloud_cet_ptr->width  = 1;
   cloud_cet_ptr->height = 1;
   cloud_cet_ptr->points.resize (cloud_cet_ptr->width * cloud_cet_ptr->height);

   for (size_t i = 0; i < cloud_cet_ptr->points.size (); ++i)
   {
    cloud_cet_ptr->points[i].x = centroid[0];
    cloud_cet_ptr->points[i].y = centroid[1];
    cloud_cet_ptr->points[i].z = centroid[2];
   }

  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_cet_handler(cloud_cet_ptr, 255, 0, 0);
  viewer.addPointCloud (cloud_cet_ptr, color_cet_handler, "cent point");//添加点云
  viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 20, "cent point");
  //--------------------
  // -----Main loop-----
  //--------------------
  while (!viewer.wasStopped ())
  {
    viewer.spinOnce (10);
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  }

}
