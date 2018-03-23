/*
点云表面法线特征
直接从点云数据集中近似推断表面法线。
下载桌子点云数据
https://raw.github.com/PointCloudLibrary/data/master/tutorials/table_scene_lms400.pcd

表面法线是几何体表面的重要属性，在很多领域都有大量应用，
例如：在进行光照渲染时产生符合可视习惯的效果时需要表面法线信息才能正常进行，
对于一个已知的几何体表面，根据垂直于点表面的矢量，因此推断表面某一点的法线方向通常比较简单。

确定表面一点法线的问题近似于估计表面的一个相切面法线的问题，
因此转换过来以后就变成一个最小二乘法平面拟合估计问题。

估计表面法线的解决方案就变成了分析一个协方差矩阵的特征矢量和特征值
（或者PCA—主成分分析），这个协方差矩阵从查询点的近邻元素中创建。
更具体地说，对于每一个点Pi,对应的协方差矩阵。

参考理解
http://geometryhub.net/notes/pointcloudnormal

PCA降维到 二维平面去法线
http://blog.codinglabs.org/articles/pca-tutorial.html


点云法线有什么用
点云渲染：法线信息可以用于光照渲染，有些地方也称着色（立体感）。
如下图所示，左边的点云没有法线信息，右边的点云有法线信息。
比如Phone光照模型里，
漫反射光照符合Lambert余弦定律：漫反射光强与N * L成正比，N为法线方向，L为点到光源的向量。
所以，在模型边缘处，N与L近似垂直，着色会比较暗。

点云的几何属性：法线可用于计算几何相关的信息，
广泛应用于点云注册（配准），点云重建，特征点检测等。
另外法线信息还可以用于区分薄板正反面。

前面说的是二维降到一维时的情况，假如我们有一堆散乱的三维点云,则可以这样计算法线：
1）对每一个点，取临近点，比如取最临近的50个点，当然会用到K-D树
2）对临近点做PCA降维，把它降到二维平面上,
可以想象得到这个平面一定是它的切平面(在切平面上才可以尽可能分散）
3）切平面的法线就是该点的法线了，而这样的法线有两个，
取哪个还需要考虑临近点的凸包方向


*/
#include <iostream>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>//点云文件pcd 读写
#include <pcl/visualization/cloud_viewer.h>//点云可视化
#include <pcl/visualization/pcl_visualizer.h>// 高级可视化点云类
#include <pcl/features/normal_3d.h>//法线特征
#include <pcl/kdtree/kdtree_flann.h>//搜索方法
#include <boost/thread/thread.hpp>
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
  reader.read( "../../Filtering/table_scene_lms400.pcd", *cloud_ptr);
  if(cloud_ptr==NULL) { cout << "pcd file read err" << endl; return -1;}
  cout << "PointCLoud size() " << cloud_ptr->width * cloud_ptr->height
       << " data points ( " << pcl::getFieldsList (*cloud_ptr) << "." << endl;

// 创建法线估计类====================================
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
//pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;//多核 计算法线模型 OpenMP
  ne.setInputCloud (cloud_ptr);
/*
 法线估计类NormalEstimation的实际计算调用程序内部执行以下操作：
对点云P中的每个点p
  1.得到p点的最近邻元素
  2.计算p点的表面法线n
  3.检查n的方向是否一致指向视点，如果不是则翻转

 在PCL内估计一点集对应的协方差矩阵，可以使用以下函数调用实现：
//定义每个表面小块的3x3协方差矩阵的存储对象
Eigen::Matrix3fcovariance_matrix;
//定义一个表面小块的质心坐标16-字节对齐存储对象
Eigen::Vector4fxyz_centroid;
//估计质心坐标
compute3DCentroid(cloud,xyz_centroid);
//计算3x3协方差矩阵
computeCovarianceMatrix(cloud,xyz_centroid,covariance_matrix);

*/

// 添加搜索算法 kdtree search  最近的几个点 估计平面 协方差矩阵PCA分解 求解法线
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
  ne.setSearchMethod (tree);

  // 输出点云 带有法线描述
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals_ptr (new pcl::PointCloud<pcl::Normal>);
  pcl::PointCloud<pcl::Normal>& cloud_normals = *cloud_normals_ptr;
  // Use all neighbors in a sphere of radius 3cm
  ne.setRadiusSearch (0.03);//半径内搜索临近点
  //ne.setKSearch(8);       //其二 指定临近点数量

  // 计算表面法线特征
  ne.compute (cloud_normals);

  // 点云+法线 可视化
  //pcl::visualization::PCLVisualizer viewer("pcd　viewer");// 显示窗口的名字
 boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_ptr (new pcl::visualization::PCLVisualizer ("3D Viewer"));  
//设置一个boost共享对象，并分配内存空间
  viewer_ptr->setBackgroundColor(0.0, 0.0, 0.0);//背景黑色
  //viewer.setBackgroundColor (1, 1, 1);//白色
  viewer_ptr->addCoordinateSystem (1.0f, "global");//坐标系
  // pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud); 
  //pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZ> rgb(cloud);
 // pcl::visualization::PointCloudColorHandlerRandom<PointType> cloud_color_handler(cloud_ptr);  
 //该句的意思是：对输入的点云着色，Random表示的是随机上色，以上是其他两种渲染色彩的方式.
  pcl::visualization::PointCloudColorHandlerCustom<PointType> cloud_color_handler (cloud_ptr, 255, 0, 0);//红色
  //viewer->addPointCloud<pcl::PointXYZRGB>(cloud_ptr,cloud_color_handler,"sample cloud");//PointXYZRGB 类型点
  viewer_ptr->addPointCloud<PointType>(cloud_ptr, cloud_color_handler, "original point cloud");//点云标签

  viewer_ptr->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud_ptr, cloud_normals_ptr, 5,0.02,"normal");//法线标签
//其中，参数5表示整个点云中每5各点显示一个法向量（若全部显示，可设置为1，  0.02表示法向量的长度，最后一个参数暂时还不知道 如何影响的）);
  viewer_ptr->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "original point cloud");
  //渲染属性，可视化工具，3维数据， 其中PCL_VISUALIZER_POINT_SIZE表示设置点的大小为3

  //viewer_ptr->addCoordinateSystem(1.0);//建立空间直角坐标系
  //viewer_ptr->setCameraPosition(0,0,200); //设置坐标原点
  viewer_ptr->initCameraParameters();//初始化相机参数

  while (!viewer_ptr->wasStopped())
    {
        viewer_ptr->spinOnce ();
        pcl_sleep(0.01);
    }

  return (0);
}
