/*
 * 基于不同领域半径估计的　法线的差异类分割点云
[paper](https://arxiv.org/pdf/1209.1759v1.pdf)
步骤：
        1. 估计大领域半径 r_l 下的　法线    
        2. 估计small领域半径 r_ｓ 下的　法线 
        3. 法线的差异  det(n,r_l, r_s) = (n_l - n_s)/2
        4. 条件滤波器   曲率大于 ??
        5. 欧式聚类 法线的差异

 */
#include <string>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/organized.h>
#include <pcl/search/kdtree.h>//搜索　kd树
#include <pcl/features/normal_3d_omp.h>//计算点云法线曲率特征
#include <pcl/filters/conditional_removal.h>//条件滤波器
#include <pcl/segmentation/extract_clusters.h>//根据点云索引提取对应的点云
#include <pcl/features/don.h>//基于不同领域半径估计的　法线的差异类
#include <pcl/segmentation/supervoxel_clustering.h>
using namespace pcl;
using namespace std;

int
main (int argc, char *argv[])
{
  //small领域半径 The smallest scale to use in the DoN filter.
  double scale1;

  //大领域半径 The largest scale to use in the DoN filter.
  double scale2;

  ///The minimum DoN magnitude to threshold by
  double threshold;

  // 欧式聚类提取 segment scene into clusters with given distance tolerance using euclidean clustering
  double segradius;

  if (argc < 6)
  {
    cerr << "usage: " << argv[0] << " inputfile smallscale largescale threshold segradius" << endl;
    exit (EXIT_FAILURE);
  }

  /// the file to read from.
  string infile = argv[1];
  /// small scale
  istringstream (argv[2]) >> scale1;
  /// large scale
  istringstream (argv[3]) >> scale2;
  istringstream (argv[4]) >> threshold;   // 法线的差异  threshold for DoN magnitude
  istringstream (argv[5]) >> segradius;   //  欧式聚类 threshold for radius segmentation

  // Load cloud in blob format
  pcl::PCLPointCloud2 blob;
  pcl::io::loadPCDFile (infile.c_str (), blob);
  pcl::PointCloud<PointXYZRGB>::Ptr cloud (new pcl::PointCloud<PointXYZRGB>);
  pcl::fromPCLPointCloud2 (blob, *cloud);

  // Create a search tree, use KDTreee for non-organized data.
  pcl::search::Search<PointXYZRGB>::Ptr tree;
  if (cloud->isOrganized ())
  {
    tree.reset (new pcl::search::OrganizedNeighbor<PointXYZRGB> ());
  }
  else
  {
    tree.reset (new pcl::search::KdTree<PointXYZRGB> (false));
  }

  // Set the input pointcloud for the search tree
  tree->setInputCloud (cloud);

  if (scale1 >= scale2)
  {
    cerr << "Error: Large scale must be > small scale!" << endl;
    exit (EXIT_FAILURE);
  }

  //   Compute normals using both small and large scales at each point
  pcl::NormalEstimationOMP<PointXYZRGB, PointNormal> ne;
  ne.setInputCloud (cloud);
  ne.setSearchMethod (tree);

  /**
   * NOTE: setting viewpoint is very important, so that we can ensure
   * normals are all pointed in the same direction!
   */
  ne.setViewPoint (std::numeric_limits<float>::max (), std::numeric_limits<float>::max (), std::numeric_limits<float>::max ());
  
// 估计small领域半径 r_ｓ 下的　法线 calculate normals with the small scale
  cout << "Calculating normals for scale..." << scale1 << endl;
  pcl::PointCloud<PointNormal>::Ptr normals_small_scale (new pcl::PointCloud<PointNormal>);
  ne.setRadiusSearch (scale1);
  ne.compute (*normals_small_scale);

  // 估计大领域半径 r_l 下的　法线  calculate normals with the large scale
  cout << "Calculating normals for scale..." << scale2 << endl;
  pcl::PointCloud<PointNormal>::Ptr normals_large_scale (new pcl::PointCloud<PointNormal>);
  ne.setRadiusSearch (scale2);
  ne.compute (*normals_large_scale);

  // Create output cloud for DoN results
  PointCloud<PointNormal>::Ptr doncloud (new pcl::PointCloud<PointNormal>);
  copyPointCloud<PointXYZRGB, PointNormal>(*cloud, *doncloud);

  cout << "Calculating DoN... " << endl;
  // Create DoN operator
  pcl::DifferenceOfNormalsEstimation<PointXYZRGB, PointNormal, PointNormal> don;
  don.setInputCloud (cloud);
  don.setNormalScaleLarge (normals_large_scale);
  don.setNormalScaleSmall (normals_small_scale);

  if (!don.initCompute ())
  {
    std::cerr << "Error: Could not initialize DoN feature operator" << std::endl;
    exit (EXIT_FAILURE);
  }
  // Compute DoN
  don.computeFeature (*doncloud);

  // Save DoN features
  pcl::PCDWriter writer;
  writer.write<pcl::PointNormal> ("don.pcd", *doncloud, false); 

  // Filter by magnitude
  cout << "Filtering out DoN mag <= " << threshold << "..." << endl;
  //创建条件定义对象  曲率
  pcl::ConditionOr<PointNormal>::Ptr range_cond (new pcl::ConditionOr<PointNormal> () );
  range_cond->addComparison (pcl::FieldComparison<PointNormal>::ConstPtr (// 曲率 大于 
                               new pcl::FieldComparison<PointNormal> ("curvature", pcl::ComparisonOps::GT, threshold))
                             );
  // Build the filter
  // pcl::ConditionalRemoval<PointNormal> condrem(*range_cond);
  pcl::ConditionalRemoval<PointNormal> condrem;//创建条件滤波器
  condrem.setCondition (range_cond);    //并用条件定义对象初始化      
  condrem.setInputCloud (doncloud);
  pcl::PointCloud<PointNormal>::Ptr doncloud_filtered (new pcl::PointCloud<PointNormal>);
  // Apply filter
  condrem.filter (*doncloud_filtered);
  doncloud = doncloud_filtered;


  // Save filtered output
  std::cout << "Filtered Pointcloud: " << doncloud->points.size () << " data points." << std::endl;
  writer.write<pcl::PointNormal> ("don_filtered.pcd", *doncloud, false); 

  // Filter by magnitude
  cout << "Clustering using EuclideanClusterExtraction with tolerance <= " << segradius << "..." << endl;

  pcl::search::KdTree<PointNormal>::Ptr segtree (new pcl::search::KdTree<PointNormal>);
  segtree->setInputCloud (doncloud);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<PointNormal> ec;

  ec.setClusterTolerance (segradius);
  ec.setMinClusterSize (50);
  ec.setMaxClusterSize (100000);
  ec.setSearchMethod (segtree);
  ec.setInputCloud (doncloud);
  ec.extract (cluster_indices);

  int j = 0;
  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it, j++)
  {
    pcl::PointCloud<PointNormal>::Ptr cloud_cluster_don (new pcl::PointCloud<PointNormal>);
    for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
    {
      cloud_cluster_don->points.push_back (doncloud->points[*pit]);
    }

    cloud_cluster_don->width = int (cloud_cluster_don->points.size ());
    cloud_cluster_don->height = 1;
    cloud_cluster_don->is_dense = true;

    //Save cluster
    cout << "PointCloud representing the Cluster: " << cloud_cluster_don->points.size () << " data points." << std::endl;
    stringstream ss;
    ss << "don_cluster_" << j << ".pcd";
    writer.write<pcl::PointNormal> (ss.str (), *cloud_cluster_don, false);
  }
}
