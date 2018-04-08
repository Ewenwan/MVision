/*无序点云的快速三角化  
得到点云文件的mesh网格文件ply 文件
使用贪婪投影三角化算法对有向点云进行三角化，
具体方法是：
（1）先将有向点云投影到某一局部二维坐标平面内
（2）在坐标平面内进行平面内的三角化
（3）根据平面内三位点的拓扑连接关系获得一个三角网格曲面模型.
贪婪投影三角化算法原理：
是处理一系列可以使网格“生长扩大”的点（边缘点）延伸这些点直到所有符合几何正确性和拓扑正确性的点都被连上，该算法可以用来处理来自一个或者多个扫描仪扫描到得到并且有多个连接处的散乱点云但是算法也是有很大的局限性，它更适用于采样点云来自表面连续光滑的曲面且点云的密度变化比较均匀的情况

*/
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_io.h>//视觉化工具函式库（VTK，Visualization Toolkit）　模型
#include <pcl/kdtree/kdtree_flann.h>// 搜索算法
#include <pcl/features/normal_3d.h> // 法线特征
#include <pcl/surface/gp3.h>        // 贪婪投影三角化算法

int
main (int argc, char** argv)
{
  // 将一个XYZ点类型的PCD文件打开并存储到对象中
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PCLPointCloud2 cloud_blob;
  pcl::io::loadPCDFile ("../bun0.pcd", cloud_blob);//cloud_blob格式打开
  pcl::fromPCLPointCloud2 (cloud_blob, *cloud_ptr);//在转换到PointXYZ格式

  // 法线估计 Normal estimation
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> nl;//法线估计对象
  pcl::PointCloud<pcl::Normal>::Ptr normals_ptr (new pcl::PointCloud<pcl::Normal>);//存储估计的法线
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);//定义kd树指针
  tree->setInputCloud (cloud_ptr);// 用cloud构建tree对象
  nl.setInputCloud (cloud_ptr);
  nl.setSearchMethod (tree);
  nl.setKSearch (20);//20个临近点
  nl.compute (*normals_ptr);//估计法线存储到其中
  //* normals should not contain the point normals + surface curvatures

  //  XYZ + normal 对象合体
  pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals_ptr (new pcl::PointCloud<pcl::PointNormal>);
  pcl::concatenateFields (*cloud_ptr, *normals_ptr, *cloud_with_normals_ptr);//连接字段
  //* cloud_with_normals = cloud + normals

  // 定义搜索树对象
  pcl::search::KdTree<pcl::PointNormal>::Ptr tree2 (new pcl::search::KdTree<pcl::PointNormal>);
  tree2->setInputCloud (cloud_with_normals_ptr);   // 点云构建搜索树

  // 贪婪投影三角化算法
  pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;   // 定义三角化对象
  pcl::PolygonMesh triangles;                // 存储最终三角化的网络模型
 
  // Set the maximum distance between connected points (maximum edge length)
  gp3.setSearchRadius (0.025);  // 设置连接点之间的最大距离，（即是三角形最大边长）

  // 设置各参数值
  gp3.setMu (2.5);  // 设置被样本点搜索其近邻点的最远距离为2.5，为了使用点云密度的变化
  gp3.setMaximumNearestNeighbors (100);// 设置样本点可搜索的邻域个数
  gp3.setMaximumSurfaceAngle(M_PI/4);  // 设置某点法线方向偏离样本点法线的最大角度45
  gp3.setMinimumAngle(M_PI/18);        // 设置三角化后得到的三角形内角的最小的角度为10
  gp3.setMaximumAngle(2*M_PI/3);       // 设置三角化后得到的三角形内角的最大角度为120
  gp3.setNormalConsistency(false);     // 设置该参数保证法线朝向一致

  // Get result
  gp3.setInputCloud (cloud_with_normals_ptr);// 设置输入点云为有向点云
  gp3.setSearchMethod (tree2);               // 设置搜索方式
  gp3.reconstruct (triangles);               // 重建提取三角化

  // 附加顶点信息
  std::vector<int> parts = gp3.getPartIDs();
  std::vector<int> states = gp3.getPointStates();

// 保存mesh文件
   pcl::io::saveVTKFile ("mesh.vtk", triangles);
 system("pcl_viewer mesh.vtk");
  // Finish
  return (0);
}
