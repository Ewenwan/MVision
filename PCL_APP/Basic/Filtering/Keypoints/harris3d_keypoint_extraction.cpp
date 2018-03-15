/* 
3DHarris　 方块体内点数量变化确定角点
　　在2DHarris里，我们使用了 图像梯度构成的 协方差矩阵。 
    图像梯度。。。嗯。。。。每个像素点都有一个梯度，
    在一阶信息量的情况下描述了两个相邻像素的关系。显然这个思想可以轻易的移植到点云上来。

想象一下，如果在 点云中存在一点p
　　1、在p上建立一个局部坐标系：z方向是法线方向，x,y方向和z垂直。
　　2、在p上建立一个小正方体，不要太大，大概像材料力学分析应力那种就行
　　3、假设点云的密度是相同的，点云是一层蒙皮，不是实心的。
　　a、如果小正方体沿z方向移动，那小正方体里的点云数量应该不变
　　b、如果小正方体位于边缘上，则沿边缘移动，点云数量几乎不变，沿垂直边缘方向移动，点云数量改
　　c、如果小正方体位于角点上，则有两个方向都会大幅改变点云数量

　　如果由法向量x,y,z构成协方差矩阵，那么它应该是一个对称矩阵。
而且特征向量有一个方向是法线方向，另外两个方向和法线垂直。
　　那么直接用协方差矩阵替换掉图像里的M矩阵，就得到了点云的Harris算法。
　　其中，半径r可以用来控制角点的规模
　　r小，则对应的角点越尖锐（对噪声更敏感）
　　r大，则可能在平缓的区域也检测出角点

./harris3d ../../Filtering/build/table_scene_lms400_inliers.pcd 
*/
#include <iostream>//标准输入输出流
#include <boost/thread/thread.hpp>
#include <pcl/io/pcd_io.h>//PCL的PCD格式文件的输入输出头文件
#include <pcl/visualization/pcl_visualizer.h>
//#include <pcl/keypoints/narf_keypoint.h>//关键点检测
#include <pcl/keypoints/harris_3d.h>

#include <cstdlib>
#include <vector>
using namespace std;

//定义别名
typedef pcl::PointXYZ PointType;

// --------------
// -----Main-----
// --------------
int 
main (int argc, char** argv)
{

  // ------------------------------------------------------------------
  // -----Read pcd file or create example point cloud if not given-----
  // ------------------------------------------------------------------
  //读取pcd文件；如果没有指定文件，就创建样本点
  pcl::PointCloud<PointType>::Ptr point_cloud_ptr(new pcl::PointCloud<PointType>);//点云对象指针
  pcl::PointCloud<PointType>& point_cloud = *point_cloud_ptr;//引用　上面点云的别名　常亮指针
  //检查参数中是否有pcd格式文件名，返回参数向量中的索引号
  if (argc>=2)//第二个参数为ｐｃｄ文件名
  {
      pcl::io::loadPCDFile (argv[1], point_cloud);
      cout << "load pcd file : "<< argv[1]<< endl;
      cout << "point_cloud has :"<< point_cloud.points.size()<<" n points."<<endl;
  }
  else//没有指定pcd文件，生成点云，并填充它
  {
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
  
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  // 3D点云显示
  //pcl::visualization::PCLVisualizer viewer ("3D Viewer");//可视化对象
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer);//可视化对象指针
  viewer->setBackgroundColor (1, 1, 1);//背景颜色　白色
  viewer->addPointCloud(point_cloud_ptr);//指针

  // --------------------------------
  // -----Extract Harri keypoints-----
  // --------------------------------
  // 提取Harri关键点
  pcl::HarrisKeypoint3D<pcl::PointXYZ,pcl::PointXYZI,pcl::Normal> harris;
  harris.setInputCloud(point_cloud_ptr);//设置输入点云 指针
  cout<<"input successful"<<endl;
  harris.setNonMaxSupression(true);
  harris.setRadius(0.02f);//　块体半径
  harris.setThreshold(0.01f);//数量阈值
  cout<<"parameter set successful"<<endl;

  //新建的点云必须初始化，清零，否则指针会越界
  //注意Harris的输出点云必须是有强度(I)信息的 pcl::PointXYZI，因为评估值保存在I分量里
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_out_ptr (new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>& cloud_out = *cloud_out_ptr;
  cloud_out.height=1;
  cloud_out.width =100;
  cloud_out.resize(cloud_out.height * cloud_out.width);    
  cloud_out.clear();
  cout << "extracting... "<< endl;
  // 计算特征点
  harris.compute(cloud_out);
  // 关键点
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_harris_ptr (new pcl::PointCloud<pcl::PointXYZ>);//指针
  pcl::PointCloud<pcl::PointXYZ>& cloud_harris = *cloud_harris_ptr;//引用
  cloud_harris.height=1;
  cloud_harris.width =100;
  cloud_harris.resize(cloud_out.height * cloud_out.width);
  cloud_harris.clear();//清空
  int size = cloud_out.size();
  cout << "extraction : " << size << " n keypoints."<< endl;
  pcl::PointXYZ point;
  //可视化结果不支持XYZI格式点云，所有又要导回XYZ格式。。。。
  for (int i = 0; i<size; ++i)
  {    
        point.x = cloud_out.at(i).x;
        point.y = cloud_out.at(i).y;
        point.z = cloud_out.at(i).z;
        cloud_harris.push_back(point);
  }
  // -------------------------------------
  // -----Show keypoints in 3D viewer-----
  // -------------------------------------
  //在3D图形窗口中显示关键点
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> harris_color_handler (cloud_harris_ptr, 0, 255, 0);//第一个参数类型为　指针
  viewer->addPointCloud(cloud_harris_ptr, harris_color_handler,"harris");//第一个参数类型为　指针
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "harris");

  //--------------------
  // -----Main loop-----
  //--------------------
  while (!viewer->wasStopped ())
  {
    viewer->spinOnce ();
    pcl_sleep(0.1);
  }
}


