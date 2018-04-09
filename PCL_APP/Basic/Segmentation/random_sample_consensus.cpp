/*
在没有任何参数的情况下，三维窗口显示创建的原始点云（含有局内点和局外点），
如图所示，很明显这是一个带有噪声的菱形平面，
噪声点是立方体，自己要是我们在产生点云是生成的是随机数生在（0，1）范围内。
./random_sample_consensus

./random_sample_consensus -f

./random_sample_consensus -sf

*/
#include <iostream>
#include <pcl/console/parse.h>
#include <pcl/filters/extract_indices.h>          // 由索引提取点云
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/ransac.h>          // 采样一致性
#include <pcl/sample_consensus/sac_model_plane.h> // 平面模型
#include <pcl/sample_consensus/sac_model_sphere.h>// 球模型
#include <pcl/visualization/pcl_visualizer.h>     // 可视化
#include <boost/thread/thread.hpp>

/*
输入点云
返回一个可视化的对象
*/
boost::shared_ptr<pcl::visualization::PCLVisualizer>
simpleVis (pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
  // --------------------------------------------
  // -----打开3维可视化窗口 加入点云----
  // --------------------------------------------
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);//背景颜色 黑se
  viewer->addPointCloud<pcl::PointXYZ> (cloud, "sample cloud");//添加点云
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");//点云对象大小
  //viewer->addCoordinateSystem (1.0, "global");//添加坐标系
  viewer->initCameraParameters ();//初始化相机参数
  return (viewer);
}

/******************************************************************************
 对点云进行初始化，并对其中一个点云填充点云数据作为处理前的的原始点云，
 其中大部分点云数据是基于设定的圆球和平面模型计算
  而得到的坐标值作为局内点，有1/5的点云数据是被随机放置的组委局外点。
 ******************************************************************************/
int
main(int argc, char** argv)
{
  // 初始化点云对象
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);  //存储源点云
  pcl::PointCloud<pcl::PointXYZ>::Ptr final1 (new pcl::PointCloud<pcl::PointXYZ>);  //存储提取的局内点

  // 填充点云数据
  cloud->width    = 500;                //填充点云数目
  cloud->height   = 1;                  //无序点云
  cloud->is_dense = false;
  cloud->points.resize (cloud->width * cloud->height);
  for (size_t i = 0; i < cloud->points.size (); ++i)
  {
    if (pcl::console::find_argument (argc, argv, "-s") >= 0 || pcl::console::find_argument (argc, argv, "-sf") >= 0)
    {
//根据命令行参数用 x^2 + y^2 + Z^2 = 1 设置一部分点云数据，此时点云组成 1/4 个球体 作为内点
      cloud->points[i].x = 1024 * rand () / (RAND_MAX + 1.0);
      cloud->points[i].y = 1024 * rand () / (RAND_MAX + 1.0);
      if (i % 5 == 0)
        cloud->points[i].z = 1024 * rand () / (RAND_MAX + 1.0);   //此处对应的点为局外点
      else if(i % 2 == 0)//正值
        cloud->points[i].z =  sqrt( 1 - (cloud->points[i].x * cloud->points[i].x)
                                      - (cloud->points[i].y * cloud->points[i].y));
      else//负值
        cloud->points[i].z =  - sqrt( 1 - (cloud->points[i].x * cloud->points[i].x)
                                        - (cloud->points[i].y * cloud->points[i].y));
    }
    else
    { //用x+y+z=1设置一部分点云数据，此时 用点云组成的菱形平面作为内点
      cloud->points[i].x = 1024 * rand () / (RAND_MAX + 1.0);
      cloud->points[i].y = 1024 * rand () / (RAND_MAX + 1.0);
      if( i % 2 == 0)
        cloud->points[i].z = 1024 * rand () / (RAND_MAX + 1.0);   //对应的局外点
      else
        cloud->points[i].z = -1 * (cloud->points[i].x + cloud->points[i].y);
    }
  }

  std::vector<int> inliers;  //存储局内点集合的点的索引的向量

  //创建随机采样一致性对象
  pcl::SampleConsensusModelSphere<pcl::PointXYZ>::Ptr
    model_s(new pcl::SampleConsensusModelSphere<pcl::PointXYZ> (cloud));   //针对球模型的对象
  pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr
    model_p (new pcl::SampleConsensusModelPlane<pcl::PointXYZ> (cloud));   //针对平面模型的对象
  if(pcl::console::find_argument (argc, argv, "-f") >= 0)
  {  //根据命令行参数，来随机估算对应平面模型，并存储估计的局内点
    pcl::RandomSampleConsensus<pcl::PointXYZ> ransac (model_p);
    ransac.setDistanceThreshold (.01);    //与平面距离小于0.01 的点称为局内点考虑
    ransac.computeModel();                //执行随机参数估计
    ransac.getInliers(inliers);           //存储估计所得的局内点
  }
  else if (pcl::console::find_argument (argc, argv, "-sf") >= 0 )
  { 
   //根据命令行参数  来随机估算对应的圆球模型，存储估计的内点
    pcl::RandomSampleConsensus<pcl::PointXYZ> ransac (model_s);
    ransac.setDistanceThreshold (.01);
    ransac.computeModel();
    ransac.getInliers(inliers);
  }

  // 复制估算模型的所有的局内点到final中
  pcl::copyPointCloud<pcl::PointXYZ>(*cloud, inliers, *final1);

  // 创建可视化对象并加入原始点云或者所有的局内点
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
  if (pcl::console::find_argument (argc, argv, "-f") >= 0 || pcl::console::find_argument (argc, argv, "-sf") >= 0)
    viewer = simpleVis(final1);
  else
    viewer = simpleVis(cloud);
  while (!viewer->wasStopped ())
  {
    viewer->spinOnce (100);
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  }
  return 0;
 }
