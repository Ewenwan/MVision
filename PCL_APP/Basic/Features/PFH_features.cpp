/*
计算法线---计算临近点对角度差值-----直方图--
点特征直方图(PFH)描述子
点特征直方图(Point Feature Histograms)
正如点特征表示法所示，表面法线和曲率估计是某个点周围的几何特征基本表示法。
虽然计算非常快速容易，但是无法获得太多信息，因为它们只使用很少的
几个参数值来近似表示一个点的k邻域的几何特征。然而大部分场景中包含许多特征点，
这些特征点有相同的或者非常相近的特征值，因此采用点特征表示法，
其直接结果就减少了全局的特征信息。

http://www.pclcn.org/study/shownews.php?lang=cn&id=101

通过参数化查询点与邻域点之间的空间差异，并形成一个多维直方图对点的k邻域几何属性进行描述。
直方图所在的高维超空间为特征表示提供了一个可度量的信息空间，
对点云对应曲面的6维姿态来说它具有不变性，
并且在不同的采样密度或邻域的噪音等级下具有鲁棒性。

是基于点与其k邻域之间的关系以及它们的估计法线，
简言之，它考虑估计法线方向之间所有的相互作用，
试图捕获最好的样本表面变化情况，以描述样本的几何特征。

Pq 用红色标注并放在圆球的中间位置，半径为r， 
(Pq)的所有k邻元素（即与点Pq的距离小于半径r的所有点）
全部互相连接在一个网络中。最终的PFH描述子通过计
算邻域内所有两点之间关系而得到的直方图，
因此存在一个O(nk^2) 的计算复杂性。


每一点对，原有12个参数，6个坐标值，6个坐标姿态（基于法线）
PHF计算没一点对的 相对坐标角度差值三个值和 坐标点之间的欧氏距离 d
从12个参数减少到4个参数

computePairFeatures (const Eigen::Vector4f &p1, const Eigen::Vector4f &n1,
const Eigen::Vector4f &p2, const Eigen::Vector4f &n2,
float &f1, float &f2, float &f3, float &f4);

为查询点创建最终的PFH表示，所有的四元组将会以某种统计的方式放进直方图中，
这个过程首先把每个特征值范围划分为b个子区间，并统计落在每个子区间的点数目，
因为四分之三的特征在上述中为法线之间的角度计量，
在三角化圆上可以将它们的参数值非常容易地归一到相同的区间内。

默认PFH的实现使用5个区间分类（例如：四个特征值中的每个都使用5个区间来统计），
其中不包括距离（在上文中已经解释过了——但是如果有需要的话，
也可以通过用户调用computePairFeatures方法来获得距离值），
这样就组成了一个125浮点数元素的特征向量（35），
其保存在一个pcl::PFHSignature125的点类型中。


*/

#include <pcl/point_types.h>
#include <pcl/features/pfh.h>
#include <pcl/io/pcd_io.h>//点云文件pcd 读写
#include <pcl/features/normal_3d.h>//法线特征

#include <pcl/visualization/histogram_visualizer.h> //直方图的可视化
#include <pcl/visualization/pcl_plotter.h>// 直方图的可视化 方法2

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


//=======【3】创建PFH估计对象pfh，并将输入点云数据集cloud和法线normals传递给它=================
  pcl::PFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::PFHSignature125> pfh;// phf特征估计其器
  pfh.setInputCloud (cloud_ptr);
  pfh.setInputNormals (cloud_normals_ptr);
 //如果点云是类型为PointNormal,则执行pfh.setInputNormals (cloud);
 //创建一个空的kd树表示法，并把它传递给PFH估计对象。
 //基于已给的输入数据集，建立kdtree
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree2 (new pcl::search::KdTree<pcl::PointXYZ> ());
  //pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr tree2 (new pcl::KdTreeFLANN<pcl::PointXYZ> ()); //-- older call for PCL 1.5-
  pfh.setSearchMethod (tree2);//设置近邻搜索算法 
  //输出数据集
  pcl::PointCloud<pcl::PFHSignature125>::Ptr pfh_fe_ptr (new pcl::PointCloud<pcl::PFHSignature125> ());//phf特征
 //使用半径在5厘米范围内的所有邻元素。
  //注意：此处使用的半径必须要大于估计表面法线时使用的半径!!!
  pfh.setRadiusSearch (0.05);
  //计算pfh特征值
  pfh.compute (*pfh_fe_ptr);


  cout << "phf feature size : " << pfh_fe_ptr->points.size() << endl;
  // 应该与input cloud->points.size ()有相同的大小，即每个点都有一个pfh特征向量


// ========直方图可视化=============================
  pcl::visualization::PCLHistogramVisualizer view;//直方图可视化
  view.setBackgroundColor(255,0,0);//背景红色
  view.addFeatureHistogram<pcl::PFHSignature125> (*pfh_fe_ptr,"pfh",100); 
  view.spinOnce();  //循环的次数
  //view.spin();  //无限循环
  // pcl::visualization::PCLPlotter plotter;
   //plotter.addFeatureHistogram(*pfh_fe_ptr, 300); //设置的很坐标长度，该值越大，则显示的越细致
   //plotter.plot();


  return 0;
}


