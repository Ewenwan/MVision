/*
视点特征直方图VFH(Viewpoint Feature Histogram)描述子，
它是一种新的特征表示形式，应用在点云聚类识别和六自由度位姿估计问题。

视点特征直方图（或VFH）是源于FPFH描述子.
由于它的获取速度和识别力，我们决定利用FPFH强大的识别力，
但是为了使构造的特征保持缩放不变性的性质同时，
还要区分不同的位姿，计算时需要考虑加入视点变量。

我们做了以下两种计算来构造特征，以应用于目标识别问题和位姿估计：
1.扩展FPFH，使其利用整个点云对象来进行计算估计（如2图所示），
在计算FPFH时以物体中心点与物体表面其他所有点之间的点对作为计算单元。

2.添加视点方向与每个点估计法线之间额外的统计信息，为了达到这个目的，
我们的关键想法是在FPFH计算中将视点方向变量直接融入到相对法线角计算当中。

通过统计视点方向与每个法线之间角度的直方图来计算视点相关的特征分量。
注意：并不是每条法线的视角，因为法线的视角在尺度变换下具有可变性，
我们指的是平移视点到查询点后的视点方向和每条法线间的角度。
第二组特征分量就是前面PFH中讲述的三个角度，如PFH小节所述，
只是现在测量的是在中心点的视点方向和每条表面法线之间的角度。

因此新组合的特征被称为视点特征直方图（VFH）。
下图表体现的就是新特征的想法，包含了以下两部分：

1.一个视点方向相关的分量
2.一个包含扩展FPFH的描述表面形状的分量

对扩展的FPFH分量来说，默认的VFH的实现使用45个子区间进行统计，
而对于视点分量要使用128个子区间进行统计，这样VFH就由一共308个浮点数组成阵列。
在PCL中利用pcl::VFHSignature308的点类型来存储表示。P
FH/FPFH描述子和VFH之间的主要区别是：

对于一个已知的点云数据集，只一个单一的VFH描述子，
而合成的PFH/FPFH特征的数目和点云中的点数目相同。


*/

#include <pcl/point_types.h>
//#include <pcl/features/pfh.h>
//#include <pcl/features/fpfh.h>
#include <pcl/features/vfh.h>
#include <pcl/io/pcd_io.h>//点云文件pcd 读写
#include <pcl/features/normal_3d.h>//法线特征

#include <pcl/visualization/histogram_visualizer.h> //直方图的可视化
#include <pcl/visualization/pcl_plotter.h>// 直方图的可视化 方法2
// 可视化 https://segmentfault.com/a/1190000006685118

//using namespace std;
using std::cout;
using std::endl;
int main(int argc, char** argv)
{

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr (new pcl::PointCloud<pcl::PointXYZ>);
//======【1】 读取点云文件　填充点云对象======
  pcl::PCDReader reader;
  reader.read( "../../Filtering/table_scene_lms400.pcd", *cloud_ptr);
  if(cloud_ptr==NULL) { cout << "pcd file read err" << endl; return -1;}
  cout << "PointCLoud size() " << cloud_ptr->width * cloud_ptr->height
       << " data points ( " << pcl::getFieldsList (*cloud_ptr) << "." << endl;

// =====【2】计算法线========创建法线估计类====================================
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
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
  ne.setSearchMethod (tree);//设置近邻搜索算法 
  // 输出点云 带有法线描述
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals_ptr (new pcl::PointCloud<pcl::Normal>);
  pcl::PointCloud<pcl::Normal>& cloud_normals = *cloud_normals_ptr;
  // Use all neighbors in a sphere of radius 3cm
  ne.setRadiusSearch (0.03);//半价内搜索临近点 3cm
  // 计算表面法线特征
  ne.compute (cloud_normals);


//=======【3】创建VFH估计对象vfh，并把输入数据集cloud和法线normal传递给它================
  pcl::VFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::VFHSignature308> vfh;
  //pcl::PFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::PFHSignature125> pfh;// phf特征估计其器
  //pcl::FPFHEstimation<pcl::PointXYZ,pcl::Normal,pcl::FPFHSignature33> fpfh;// fphf特征估计其器
  // pcl::FPFHEstimationOMP<pcl::PointXYZ,pcl::Normal,pcl::FPFHSignature33> fpfh;//多核加速
  vfh.setInputCloud (cloud_ptr);
  vfh.setInputNormals (cloud_normals_ptr);
  //如果点云是PointNormal类型，则执行vfh.setInputNormals (cloud);
  //创建一个空的kd树对象，并把它传递给FPFH估计对象。
  //基于已知的输入数据集，建立kdtree
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree2 (new pcl::search::KdTree<pcl::PointXYZ> ());
  //pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr tree2 (new pcl::KdTreeFLANN<pcl::PointXYZ> ()); //-- older call for PCL 1.5-
  vfh.setSearchMethod (tree2);//设置近邻搜索算法 
  //输出数据集
  //pcl::PointCloud<pcl::PFHSignature125>::Ptr pfh_fe_ptr (new pcl::PointCloud<pcl::PFHSignature125> ());//phf特征
  //pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_fe_ptr (new pcl::PointCloud<pcl::FPFHSignature33>());//fphf特征
  pcl::PointCloud<pcl::VFHSignature308>::Ptr vfh_fe_ptr (new pcl::PointCloud<pcl::VFHSignature308> ());//vhf特征
  //使用半径在5厘米范围内的所有邻元素。
  //注意：此处使用的半径必须要大于估计表面法线时使用的半径!!!
  //fpfh.setRadiusSearch (0.05);
  //计算pfh特征值
  vfh.compute (*vfh_fe_ptr);


  cout << "phf feature size : " << vfh_fe_ptr->points.size() << endl;
  // 应该 等于 1
// ========直方图可视化=============================
  //pcl::visualization::PCLHistogramVisualizer view;//直方图可视化
  //view.setBackgroundColor(255,0,0);//背景红色
  //view.addFeatureHistogram<pcl::VFHSignature308> (*vfh_fe_ptr,"vfh"，0); 
  //view.spinOnce();  //循环的次数
  //view.spin();  //无限循环
  pcl::visualization::PCLPlotter plotter;
  plotter.addFeatureHistogram(*vfh_fe_ptr, 300); //设置的很坐标长度，该值越大，则显示的越细致
  plotter.plot();


  return 0;
}


