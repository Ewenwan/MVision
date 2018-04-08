/*
滑动最小二乘 表面平滑
重建算法
在测量较小的数据时会产生一些误差，这些误差所造成的不规则数据如果直接拿来曲面重建的话，
会使得重建的曲面不光滑或者有漏洞，可以采用对数据重采样来解决这样问题，
通过对周围的数据点进行高阶多项式插值来重建表面缺少的部分，
1）用最小二乘法对点云进行平滑处理
Moving Least Squares (MLS) surface reconstruction method 
滑动最小二乘 表面平滑　重建算法

虽说此类放在了Surface下面，但是通过反复的研究与使用，
我发现此类并不能输出拟合后的表面，不能生成Mesh或者Triangulations，
只是将点云进行了MLS的映射，使得输出的点云更加平滑。

因此，在我看来此类应该放在Filter下。
通过多次的实验与数据的处理，
我发现此类主要适用于点云的光顺处理，
当然输入的点云最好是滤过离群点之后的点集，
否则将会牺牲表面拟合精度的代价来获得输出点云。

Pcl::MovingLeastSquares<PointInT, PointOutT> mls
其中PointInT决定了输入点云类型，
PointOutT为点云输出类型（
当法线估计标志位设置为true时，输出向量必须加上normals，这一点我们将会在成员函数里介绍）。



*/
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h> //kd-tree搜索对象的类定义的头文件
#include <pcl/surface/mls.h>         //滑动 最小二乘法平滑处理类定义头文件

int
main (int argc, char** argv)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr (new pcl::PointCloud<pcl::PointXYZ> ());
  pcl::io::loadPCDFile ("../bun0.pcd", *cloud_ptr);

  // 创建 KD-Tree
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  // Output has the PointNormal type in order to store the normals calculated by MLS
  pcl::PointCloud<pcl::PointNormal> mls_points;
  // 定义最小二乘实现的对象mls
  pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal> mls;
  mls.setComputeNormals (true);  //设置在最小二乘计算中需要进行法线估计
  // Set parameters
  mls.setInputCloud (cloud_ptr);//


// 3D点云显示
// pcl::visualization::PCLVisualizer viewer ("3D Viewer");
/// viewer.setBackgroundColor (1, 1, 1);//背景颜色　白色
// pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(cloud_ptr, 1.0, 1.0, 1.0);
//  viewer.addPointCloud (cloud_ptr, color_handler, "raw point");//添加点云
//  viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "raw point");
  //viewer.addCoordinateSystem (1.0);
 // viewer.initCameraParameters ();

  // mls.setPolynomialOrder(3);
// MLS拟合曲线的阶数，这个阶数在构造函数里默认是2，
//但是参考文献给出最好选择3或者4，当然不难得出随着阶数的增加程序运行的时间也增加。

  //mls.setPolynomialFit (true);// 对于法线的估计是由多项式还是仅仅依靠切线 旧就版本
  mls.setPolynomialOrder(true);
  mls.setSearchMethod (tree);//　使用kdTree加速搜索
  mls.setSearchRadius (0.03);// 确定搜索的半径。
// 也就是说在这个半径里进行表面映射和曲面拟合。
// 从实验结果可知：半径越小拟合后曲面的失真度越小，反之有可能出现过拟合的现象。
// mls.setUpsamplingMethod(NONE);//上采样　增加密度较小区域的密度　　对于holes的填补却无能为力

//mls.setUpsamplingMethod(SAMPLE_LOCAL_PLANE);
// 需要设置半径　和　步数　mls.setUpsamplingRadius() 
// 此函数规定了点云增长的区域。可以这样理解：把整个点云按照此半径划分成若干个子点云，然后一一索引进行点云增长。
// mls.setUpsamlingStepSize(double size) 对于每个子点云处理时迭代的步长

//mls.setUpsamplingMethod(RANDOM_UNIFORM_DENSITY);// 它使得稀疏区域的密度增加，从而使得整个点云的密度均匀
// 需要设置密度　 mls.setPointDensity(int desired_num);//意为半径内点的个数。

//mls.setUpsamplingMethod(VOXEL_GRID_DILATION);// 体素格　上采样
// 填充空洞和平均化点云的密度。它需要调用的函数为：
// mls.setDilationVoxelSize(float voxel_size) 设定voxel的大小。
// mls.setDilationIterations(int iterations) 设置迭代的次数。

  // Reconstruct
  mls.process (mls_points);


  // Save output
  pcl::io::savePCDFile ("bun0-mls.pcd", mls_points);
}
