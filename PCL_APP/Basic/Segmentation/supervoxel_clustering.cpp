/*超体聚类是一种图像的分割方法。
超体（supervoxel）是一种集合，集合的元素是“体”。
与体素滤波器中的体类似，其本质是一个个的小方块。
与大部分的分割手段不同，超体聚 类的目的并不是分割出某种特定物体，超
体是对点云实施过分割(over segmentation)，将场景点云化成很多小块，并研究每个小块之间的关系。
这种将更小单元合并的分割思路已经出现了有些年份了，在图像分割中，像 素聚类形成超像素，
以超像素关系来理解图像已经广为研究。本质上这种方法是对局部的一种总结，
纹理，材质，颜色类似的部分会被自动的分割成一块，有利于后 续识别工作。
比如对人的识别，如果能将头发，面部，四肢，躯干分开，则能更好的对各种姿态，性别的人进行识别。

点云和图像不一样，其不存在像素邻接关系。所以，超体聚类之前，
必须以八叉树对点云进行划分，获得不同点团之间的邻接关系。
与图像相似点云的邻接关系也有很多，如面邻接，线邻接，点邻接。

超体聚类实际上是一种特殊的区域生长算法，和无限制的生长不同，
超体聚类首先需要规律的布置区域生长“晶核”。晶核在空间中实际上是均匀分布的,
并指定晶核距离（Rseed)。再指定粒子距离(Rvoxel)。
再指定最小晶粒(MOV)，过小的晶粒需要融入最近的大晶粒。
[reference](http://www.cnblogs.com/li-yao7758258/p/6628175.html)

./supervoxel --NT

*/
#include <pcl/console/parse.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/segmentation/supervoxel_clustering.h>

//VTK include needed for drawing graph lines
#include <vtkPolyLine.h>

// 数据类型
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointNT> PointNCloudT;
typedef pcl::PointXYZL PointLT;
typedef pcl::PointCloud<PointLT> PointLCloudT;

//可视化
void addSupervoxelConnectionsToViewer (PointT &supervoxel_center,
                                       PointCloudT &adjacent_supervoxel_centers,
                                       std::string supervoxel_name,
                                       boost::shared_ptr<pcl::visualization::PCLVisualizer> & viewer);


int
main (int argc, char ** argv)
{
//解析命令行
 // if (argc < 2)
 // {
    pcl::console::print_error ("Syntax is: %s <pcd-file> \n "
                                "--NT Dsables the single cloud transform \n"
                                "-v <voxel resolution>\n-s <seed resolution>\n"
                                "-c <color weight> \n-z <spatial weight> \n"
                                "-n <normal_weight>\n", argv[0]);
  //  return (1);
  //}

  //打开点云
  PointCloudT::Ptr cloud = boost::shared_ptr <PointCloudT> (new PointCloudT ());
  pcl::console::print_highlight ("Loading point cloud...\n");
  if (pcl::io::loadPCDFile<PointT> ("../milk_cartoon_all_small_clorox.pcd", *cloud))
  {
    pcl::console::print_error ("Error loading cloud file!\n");
    return (1);
  }


  bool disable_transform = pcl::console::find_switch (argc, argv, "--NT");

  float voxel_resolution = 0.008f;  //分辨率
  bool voxel_res_specified = pcl::console::find_switch (argc, argv, "-v");
  if (voxel_res_specified)
    pcl::console::parse (argc, argv, "-v", voxel_resolution);

  float seed_resolution = 0.1f;
  bool seed_res_specified = pcl::console::find_switch (argc, argv, "-s");
  if (seed_res_specified)
    pcl::console::parse (argc, argv, "-s", seed_resolution);

  float color_importance = 0.2f;
  if (pcl::console::find_switch (argc, argv, "-c"))
    pcl::console::parse (argc, argv, "-c", color_importance);

  float spatial_importance = 0.4f;
  if (pcl::console::find_switch (argc, argv, "-z"))
    pcl::console::parse (argc, argv, "-z", spatial_importance);

  float normal_importance = 1.0f;
  if (pcl::console::find_switch (argc, argv, "-n"))
    pcl::console::parse (argc, argv, "-n", normal_importance);

//如何使用SupervoxelClustering函数
  pcl::SupervoxelClustering<PointT> super (voxel_resolution, seed_resolution);
  if (disable_transform)//如果设置的是参数--NT  就用默认的参数
  super.setUseSingleCameraTransform (false);
  super.setInputCloud (cloud);
  super.setColorImportance (color_importance); //0.2f
  super.setSpatialImportance (spatial_importance); //0.4f
  super.setNormalImportance (normal_importance); //1.0f

  std::map <uint32_t, pcl::Supervoxel<PointT>::Ptr > supervoxel_clusters;

  pcl::console::print_highlight ("Extracting supervoxels!\n");
  super.extract (supervoxel_clusters);
  pcl::console::print_info ("Found %d supervoxels\n", supervoxel_clusters.size ());

  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);

  PointCloudT::Ptr voxel_centroid_cloud = super.getVoxelCentroidCloud ();//获得体素中心的点云
  viewer->addPointCloud (voxel_centroid_cloud, "voxel centroids");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,2.0, "voxel centroids");     //渲染点云
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_OPACITY,0.95, "voxel centroids");

  PointLCloudT::Ptr labeled_voxel_cloud = super.getLabeledVoxelCloud ();
  viewer->addPointCloud (labeled_voxel_cloud, "labeled voxels");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_OPACITY,0.8, "labeled voxels");

  PointNCloudT::Ptr sv_normal_cloud = super.makeSupervoxelNormalCloud (supervoxel_clusters);

  //We have this disabled so graph is easy to see, uncomment to see supervoxel normals
  //viewer->addPointCloudNormals<PointNormal> (sv_normal_cloud,1,0.05f, "supervoxel_normals");

  pcl::console::print_highlight ("Getting supervoxel adjacency\n");

  std::multimap<uint32_t, uint32_t> supervoxel_adjacency;
  super.getSupervoxelAdjacency (supervoxel_adjacency);
  //To make a graph of the supervoxel adjacency, we need to iterate through the supervoxel adjacency multimap
  //为了使整个超体形成衣服图，我们需要遍历超体的每个临近的个体
  std::multimap<uint32_t,uint32_t>::iterator label_itr = supervoxel_adjacency.begin ();
  for ( ; label_itr != supervoxel_adjacency.end (); )
  {
    //First get the label
    uint32_t supervoxel_label = label_itr->first;
    //Now get the supervoxel corresponding to the label
    pcl::Supervoxel<PointT>::Ptr supervoxel = supervoxel_clusters.at (supervoxel_label);

    //Now we need to iterate through the adjacent supervoxels and make a point cloud of them
    PointCloudT adjacent_supervoxel_centers;
    std::multimap<uint32_t,uint32_t>::iterator adjacent_itr = supervoxel_adjacency.equal_range (supervoxel_label).first;
    for ( ; adjacent_itr!=supervoxel_adjacency.equal_range (supervoxel_label).second; ++adjacent_itr)
    {
      pcl::Supervoxel<PointT>::Ptr neighbor_supervoxel = supervoxel_clusters.at (adjacent_itr->second);
      adjacent_supervoxel_centers.push_back (neighbor_supervoxel->centroid_);
    }
    //Now we make a name for this polygon
    std::stringstream ss;
    ss << "supervoxel_" << supervoxel_label;
    //This function is shown below, but is beyond the scope of this tutorial - basically it just generates a "star" polygon mesh from the points given
//从给定的点云中生成一个星型的多边形，
    addSupervoxelConnectionsToViewer (supervoxel->centroid_, adjacent_supervoxel_centers, ss.str (), viewer);
    //Move iterator forward to next label
    label_itr = supervoxel_adjacency.upper_bound (supervoxel_label);
  }

  while (!viewer->wasStopped ())
  {
    viewer->spinOnce (100);
  }
  return (0);
}

//VTK可视化构成的聚类图
void
addSupervoxelConnectionsToViewer (PointT &supervoxel_center,
                                  PointCloudT &adjacent_supervoxel_centers,
                                  std::string supervoxel_name,
                                  boost::shared_ptr<pcl::visualization::PCLVisualizer> & viewer)
{
  vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New ();
  vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New ();
  vtkSmartPointer<vtkPolyLine> polyLine = vtkSmartPointer<vtkPolyLine>::New ();

  //Iterate through all adjacent points, and add a center point to adjacent point pair
  PointCloudT::iterator adjacent_itr = adjacent_supervoxel_centers.begin ();
  for ( ; adjacent_itr != adjacent_supervoxel_centers.end (); ++adjacent_itr)
  {
    points->InsertNextPoint (supervoxel_center.data);
    points->InsertNextPoint (adjacent_itr->data);
  }
  // Create a polydata to store everything in
  vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New ();
  // Add the points to the dataset
  polyData->SetPoints (points);
  polyLine->GetPointIds  ()->SetNumberOfIds(points->GetNumberOfPoints ());
  for(unsigned int i = 0; i < points->GetNumberOfPoints (); i++)
    polyLine->GetPointIds ()->SetId (i,i);
  cells->InsertNextCell (polyLine);
  // Add the lines to the dataset
  polyData->SetLines (cells);
  viewer->addModelFromPolyData (polyData,supervoxel_name);
}
