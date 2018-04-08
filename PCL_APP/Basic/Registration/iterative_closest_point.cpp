/*
使用ICP迭代最近点算法，程序随机生成一个点与作为源点云，
并将其沿x轴平移后作为目标点云，
然后利用ICP估计源点云到目标点云的
刚体变换矩阵，中间对所有信息都打印出来。


迭代最近点算法（Iterative CLosest Point简称ICP算法）:
ICP算法对待拼接的2片点云，首先根据一定的准则确立对应点集P与Q，
其中对应点对的个数，然后通过最小二乘法迭代计算最优的坐标变换，
即旋转矩阵R和平移矢量t，使得误差函数最小，

迭代最近点 Iterative Closest Point （ICP）
ICP算法本质上是基于最小二乘法的最优配准方法。
该算法重复进行选择对应关系点对，计算最优刚体变换这一过程，
直到满足正确配准的收敛精度要求。
算法的输入：参考点云和目标点云，停止迭代的标准。
算法的输出：旋转和平移矩阵，即转换矩阵。

ICP处理流程分为四个主要的步骤：

	1. 对原始点云数据进行采样(关键点 keypoints(NARF, SIFT 、FAST、均匀采样 UniformSampling)、
	   特征描述符　descriptions，NARF、 FPFH、BRIEF 、SIFT、ORB )
	2. 确定初始对应点集(匹配 matching )
	3. 去除错误对应点对(随机采样一致性估计 RANSAC )
	4. 坐标变换的求解

Feature based registration
    1. SIFT 关键点 (pcl::SIFT…something)
    2. FPFH 特征描述符  (pcl::FPFHEstimation)  
    3. 估计对应关系 (pcl::CorrespondenceEstimation)
    4. 错误对应关系的去除( pcl::CorrespondenceRejectionXXX )  
    5. 坐标变换的求解

使用点匹配时，使用点的XYZ的坐标作为特征值，针对有序点云和无序点云数据的不同的处理策略：
	1. 穷举配准（brute force matching）;
	2. kd树最近邻查询（FLANN）;
	3. 在有序点云数据的图像空间中查找;
	4. 在无序点云数据的索引空间中查找.
特征描述符匹配：
	1. 穷举配准（brute force matching）;
	2. kd树最近邻查询（FLANN）。


错误对应关系的去除（correspondence rejection）:

	由于噪声的影响，通常并不是所有估计的对应关系都是正确的，
	由于错误的对应关系对于最终的刚体变换矩阵的估算会产生负面的影响，
	所以必须去除它们，可以采用随机采样一致性估计，或者其他方法剔除错误的对应关系，
	最终只使用一定比例的对应关系，这样既能提高变换矩阵的估计京都也可以提高配准点的速度。

变换矩阵的估算（transormation estimation）的步骤如下:

	1. 在对应关系的基础上评估一些错误的度量标准
	2. 在摄像机位姿（运动估算）和最小化错误度量标准下估算一个
	　　刚体变换(  rigid  transformation )
	3. 优化点的结构 (SVD奇异值分解 运动估计;使用Levenberg-Marquardt 优化 运动估计;)
	4. 使用刚体变换把源旋转/平移到与目标所在的同一坐标系下，用所有点，点的一个子集或者关键点运算一个内部的ICP循环
	5. 进行迭代，直到符合收敛性判断标准为止。

可以有试验结果看得出变换后的点云只是在x轴的值增加了固定的值0.7,
然后由这目标点云与源点云计算出它的旋转与平移，明显可以看出最后一行的x值为0.7
同时，我们可以自己更改程序，来观察不同的实验结果。

对于两幅图像通过ICP求它的变换：
刚开始，如果直接通过通过kinect 得到数据运行会出现如下的错误，是因为该ICP 
算法不能处理含有NaNs的点云数据，所以需要通过移除这些点，才能作为ICP算法的输入



*/
#include <iostream>                 //标准输入输出头文件
#include <pcl/io/pcd_io.h>          //I/O操作头文件
#include <pcl/point_types.h>        //点类型定义头文件
#include <pcl/registration/icp.h>   //ICP配准类相关头文件
#include <sstream> 

int
 main (int argc, char** argv)
{ 
  //创建两个pcl::PointCloud<pcl::PointXYZ>共享指针，并初始化它们
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out (new pcl::PointCloud<pcl::PointXYZ>);

  float x_trans = 0.7;
  if(argc>=2) {
   std::istringstream xss(argv[1]);
   xss >> x_trans;

  }

  // 随机填充点云
  cloud_in->width    = 5;               //设置点云宽度
  cloud_in->height   = 1;               //设置点云为无序点
  cloud_in->is_dense = false;
  cloud_in->points.resize (cloud_in->width * cloud_in->height);
  for (size_t i = 0; i < cloud_in->points.size (); ++i)//循环随机填充
  {
    cloud_in->points[i].x = 1024 * rand () / (RAND_MAX + 1.0f);
    cloud_in->points[i].y = 1024 * rand () / (RAND_MAX + 1.0f);
    cloud_in->points[i].z = 1024 * rand () / (RAND_MAX + 1.0f);
  }
  std::cout << "Saved " << cloud_in->points.size () << " data points to input:"//打印处点云总数
      << std::endl;
// 打印坐标
  for (size_t i = 0; i < cloud_in->points.size (); ++i) std::cout << "    " << //打印处实际坐标
      cloud_in->points[i].x << " " << cloud_in->points[i].y << " " <<
      cloud_in->points[i].z << std::endl;
  *cloud_out = *cloud_in;
  std::cout << "size:" << cloud_out->points.size() << std::endl;
//实现一个简单的点云刚体变换，以构造目标点云
  for (size_t i = 0; i < cloud_in->points.size (); ++i)
    cloud_out->points[i].x = cloud_in->points[i].x + x_trans;//　x轴正方向偏移　0.7米
  std::cout << "Transformed " << cloud_in->points.size () << " data points:"
      << std::endl;
//打印构造出来的目标点云
  for (size_t i = 0; i < cloud_out->points.size (); ++i)   
      std::cout << "    " << cloud_out->points[i].x << " " <<
      cloud_out->points[i].y << " " << cloud_out->points[i].z << std::endl;

  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;//创建IterativeClosestPoint的对象
  icp.setInputCloud(cloud_in);                 //cloud_in设置为点云的源点
  icp.setInputTarget(cloud_out);               //cloud_out设置为与cloud_in对应的匹配目标
  pcl::PointCloud<pcl::PointXYZ> Final;        //存储经过配准变换点云后的点云
  icp.align(Final);  
    
//打印经过配准变换点云后的点云
std::cout << "Assiend " << Final.points.size () << " data points:"
      << std::endl;
  for (size_t i = 0; i < Final.points.size (); ++i)   
      std::cout << "    " << Final.points[i].x << " " <<
      			     Final.points[i].y << " " << 
			     Final.points[i].z << std::endl;                      
  //打印配准相关输入信息
  std::cout << "has converged:" << icp.hasConverged() << " score: " <<
  icp.getFitnessScore() << std::endl;
  std::cout << "Transformation: "<< "\n" << icp.getFinalTransformation() << std::endl;

 return (0);
}
