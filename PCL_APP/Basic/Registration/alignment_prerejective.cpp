/* 
刚体的鲁棒位置估计
刚体　在场景点云中　匹配出位姿
对目标点云和　场景点云　体素格下采样　
对目标点云和　场景点云　提取法线特征
对目标点云和　场景点云　提取fpfh特征
SampleConsensusPrerejective随机采样一致性　配准　

Robust pose estimation of rigid objects


*/
#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/time.h>//　时间
#include <pcl/console/print.h>//　命令行　打印
#include <pcl/features/normal_3d_omp.h>//　发现特征
#include <pcl/features/fpfh_omp.h>//　快速点特征直方图特征
#include <pcl/filters/filter.h>//　滤波
#include <pcl/filters/voxel_grid.h>//　体素格滤波
#include <pcl/io/pcd_io.h>//io
#include <pcl/registration/icp.h>//　配准
#include <pcl/registration/sample_consensus_prerejective.h>//　随机采样一致性　投影
#include <pcl/segmentation/sac_segmentation.h>//　分割　随机采样性　分割
#include <pcl/visualization/pcl_visualizer.h>//　可视化

// Types
typedef pcl::PointNormal PointNT;// xyz点＋法线＋曲率
typedef pcl::PointCloud<PointNT> PointCloudT;//点云
typedef pcl::FPFHSignature33 FeatureT;//fphf特征
typedef pcl::FPFHEstimationOMP<PointNT,PointNT,FeatureT> FeatureEstimationT;//fphf特征估计　多核
typedef pcl::PointCloud<FeatureT> FeatureCloudT;//特征点云
typedef pcl::visualization::PointCloudColorHandlerCustom<PointNT> ColorHandlerT;//点云颜色控制器

//　刚体　在场景点云中　匹配出位姿 
// Align a rigid object to a scene with clutter and occlusions
int
main (int argc, char **argv)
{
  // 点云　Point clouds
  PointCloudT::Ptr object (new PointCloudT);//物体点云
  PointCloudT::Ptr object_aligned (new PointCloudT);//物体匹配点云
  PointCloudT::Ptr scene (new PointCloudT);//场景　点云
  FeatureCloudT::Ptr object_features (new FeatureCloudT);//物体点云fphf特征
  FeatureCloudT::Ptr scene_features (new FeatureCloudT);//　场景点云fphf特征
  
  // 命令行参数检测　Get input object and scene
  if (argc != 3)
  {
    pcl::console::print_error ("Syntax is: %s object.pcd scene.pcd\n", argv[0]);
    return (1);
  }
  
  // 载入物体点云　和　场景点云　object and scene
  pcl::console::print_highlight ("Loading point clouds...\n");
  if (pcl::io::loadPCDFile<PointNT> (argv[1], *object) < 0 ||
      pcl::io::loadPCDFile<PointNT> (argv[2], *scene) < 0)
  {
    pcl::console::print_error ("Error loading object/scene file!\n");
    return (1);
  }
  
  //　体素格　下采样降低数量
  pcl::console::print_highlight ("Downsampling...\n");
  pcl::VoxelGrid<PointNT> grid;//　体素格滤波
  const float leaf = 0.005f;   //　体素格大小 5cm
  grid.setLeafSize (leaf, leaf, leaf);
  grid.setInputCloud (object);//输入点云
  grid.filter (*object);//得到　下采样输出
  grid.setInputCloud (scene);
  grid.filter (*scene);
  
  // 估计场景点云的　法线　scene
  pcl::console::print_highlight ("Estimating scene normals...\n");
  pcl::NormalEstimationOMP<PointNT,PointNT> nest;//　多线程　法线特征估计
  nest.setRadiusSearch (0.01);//　搜索半径
  nest.setInputCloud (scene);//　输入场景点云
  nest.compute (*scene);//　计算法线
  
  nest.setInputCloud (object);//　输入物体点云
  nest.compute (*object);//　计算法线

  
  
  // 估计物体点云　和　场景点云　fphf特征　Estimate features
  pcl::console::print_highlight ("Estimating features...\n");
  FeatureEstimationT fest;//　fphf特征估计
  fest.setRadiusSearch (0.025);//　搜索半径
  fest.setInputCloud (object);
  fest.setInputNormals (object);
  fest.compute (*object_features);//　物体点云　fphf特征
  fest.setInputCloud (scene);
  fest.setInputNormals (scene);
  fest.compute (*scene_features);//　场景点云　fphf特征
  
  // SampleConsensusPrerejective随机采样一致性　配准　Perform alignment
  pcl::console::print_highlight ("Starting alignment...\n");
  pcl::SampleConsensusPrerejective<PointNT,PointNT,FeatureT> align;//配准
  align.setInputSource (object);//源点云
  align.setSourceFeatures (object_features);//源点云　fphf特征
  align.setInputTarget (scene);//目标点云
  align.setTargetFeatures (scene_features);//目标点云　　特征
  align.setMaximumIterations (50000); // 　RANSAC 　最大迭代次数
  align.setNumberOfSamples (3); // 采样点数　Number of points to sample for generating/prerejecting a pose
  align.setCorrespondenceRandomness (5);// 使用的特征数量　Number of nearest features to use
  align.setSimilarityThreshold (0.9f); // 相似性　阈值　Polygonal edge length similarity threshold
  align.setMaxCorrespondenceDistance (2.5f * leaf);// 内点　阈值　Inlier threshold
  align.setInlierFraction (0.25f); // Required inlier fraction for accepting a pose hypothesis
  {
    pcl::ScopeTime t("Alignment");
    align.align (*object_aligned);
  }
  
  if (align.hasConverged ())
  {
    // 打印结果　Print results
    printf ("\n");
    Eigen::Matrix4f transformation = align.getFinalTransformation ();
    pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (0,0), transformation (0,1), transformation (0,2));
    pcl::console::print_info ("R = | %6.3f %6.3f %6.3f | \n", transformation (1,0), transformation (1,1), transformation (1,2));
    pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (2,0), transformation (2,1), transformation (2,2));
    pcl::console::print_info ("\n");
    pcl::console::print_info ("t = < %0.3f, %0.3f, %0.3f >\n", transformation (0,3), transformation (1,3), transformation (2,3));
    pcl::console::print_info ("\n");
    pcl::console::print_info ("Inliers: %i/%i\n", align.getInliers ().size (), object->size ());
    
    // 显示配准　Show alignment
    pcl::visualization::PCLVisualizer visu("Alignment");
    visu.addPointCloud (scene, ColorHandlerT (scene, 0.0, 255.0, 0.0), "scene");
    visu.addPointCloud (object_aligned, ColorHandlerT (object_aligned, 0.0, 0.0, 255.0), "object_aligned");
    visu.spin ();
  }
  else
  {
    pcl::console::print_error ("Alignment failed!\n");
    return (1);
  }
  
  return (0);
}

