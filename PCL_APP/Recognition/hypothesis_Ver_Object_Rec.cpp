/*3D物体识别的假设检验 
如何做3D物体识别通过验证模型假设在聚类里面。在描述器匹配后，
这次我们将运行某个相关组算法在PCL里面为了聚类点对点相关性的集合，
决定假设物体在场景里面的实例。
在这个假定里面，全局假设验证算法将被用来减少错误的数量。


*/
#include <pcl/io/pcd_io.h>// 文件、设备读写
#include <pcl/point_cloud.h>//基础pcl点云类型
#include <pcl/correspondence.h>//分组算法 对应表示两个实体之间的匹配（例如，点，描述符等）。
#include <pcl/features/normal_3d_omp.h>//法向量特征 + 曲率特征
#include <pcl/features/shot_omp.h>//描述子 shot描述子 0～1
#include <pcl/features/board.h>
#include <pcl/filters/uniform_sampling.h>//均匀采样 滤波
#include <pcl/recognition/cg/hough_3d.h>//hough算子
#include <pcl/recognition/cg/geometric_consistency.h>//几何一致性
#include <pcl/recognition/hv/hv_go.h>//模型假设验证 Hypothesis Verification 
#include <pcl/registration/icp.h>//icp点云配准
#include <pcl/visualization/pcl_visualizer.h>//可视化
#include <pcl/kdtree/kdtree_flann.h>// kdtree 快速近邻搜索
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/common/transforms.h> //点云转换 转换矩阵
#include <pcl/console/parse.h>//命令行参数解析
// 别名
typedef pcl::PointXYZRGBA PointType;//PointXYZRGBA数据结构 点类型 位置和颜色
typedef pcl::Normal NormalType;     //法线类型　　法线＋曲率
typedef pcl::ReferenceFrame RFType; //参考帧
typedef pcl::SHOT352 DescriptorType;//SHOT特征的数据结构（32*11=352）
//点云风格　颜色　和大小　结构体
struct CloudStyle
{
    double r;
    double g;
    double b;
    double size;

    CloudStyle (double r,
                double g,
                double b,
                double size) :
        r (r),
        g (g),
        b (b),
        size (size)
    {
    }
};
CloudStyle style_white (255.0, 255.0, 255.0, 4.0);//白色
CloudStyle style_red (255.0, 0.0, 0.0, 3.0);
CloudStyle style_green (0.0, 255.0, 0.0, 5.0);
CloudStyle style_cyan (93.0, 200.0, 217.0, 4.0);
CloudStyle style_violet (255.0, 0.0, 255.0, 8.0);

std::string model_filename_;
std::string scene_filename_;

//算法参数 Algorithm params
bool show_keypoints_ (false);
bool use_hough_ (true);
float model_ss_ (0.02f);// 模型点云下采样滤波搜索半径
float scene_ss_ (0.02f);// 场景点云下采样滤波搜索半径 
float rf_rad_ (0.015f);
float descr_rad_ (0.02f);
float cg_size_ (0.01f);//聚类 霍夫空间设置每个bin的大小
float cg_thresh_ (5.0f);//聚类阈值
int icp_max_iter_ (5);//icp最大迭代次数
float icp_corr_distance_ (0.005f);//icp相关性距离
float hv_clutter_reg_ (5.0f);//模型假设验证 Hypothesis Verification 
float hv_inlier_th_ (0.005f);
float hv_occlusion_th_ (0.01f);
float hv_rad_clutter_ (0.03f);
float hv_regularizer_ (3.0f);
float hv_rad_normals_ (0.05);
bool hv_detect_clutter_ (true);

// 打印帮组信息 程序用法=========================================================
void
showHelp (char *filename)
{
  std::cout << std::endl;
  std::cout << "***************************************************************************" << std::endl;
  std::cout << "*                                                                         *" << std::endl;
  std::cout << "*          Global Hypothese Verification Tutorial - Usage Guide          *" << std::endl;
  std::cout << "*                                                                         *" << std::endl;
  std::cout << "***************************************************************************" << std::endl << std::endl;
  std::cout << "Usage: " << filename << " model_filename.pcd scene_filename.pcd [Options]" << std::endl << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "     -h:                          Show this help." << std::endl;
  std::cout << "     -k:                          Show keypoints." << std::endl;
  std::cout << "     --algorithm (Hough|GC):      Clustering algorithm used (default Hough)." << std::endl;
  std::cout << "     --model_ss val:              Model uniform sampling radius (default " << model_ss_ << ")" << std::endl;
  std::cout << "     --scene_ss val:              Scene uniform sampling radius (default " << scene_ss_ << ")" << std::endl;
  std::cout << "     --rf_rad val:                Reference frame radius (default " << rf_rad_ << ")" << std::endl;
  std::cout << "     --descr_rad val:             Descriptor radius (default " << descr_rad_ << ")" << std::endl;
  std::cout << "     --cg_size val:               Cluster size (default " << cg_size_ << ")" << std::endl;
  std::cout << "     --cg_thresh val:             Clustering threshold (default " << cg_thresh_ << ")" << std::endl << std::endl;
  std::cout << "     --icp_max_iter val:          ICP max iterations number (default " << icp_max_iter_ << ")" << std::endl;
  std::cout << "     --icp_corr_distance val:     ICP correspondence distance (default " << icp_corr_distance_ << ")" << std::endl << std::endl;
  std::cout << "     --hv_clutter_reg val:        Clutter Regularizer (default " << hv_clutter_reg_ << ")" << std::endl;
  std::cout << "     --hv_inlier_th val:          Inlier threshold (default " << hv_inlier_th_ << ")" << std::endl;
  std::cout << "     --hv_occlusion_th val:       Occlusion threshold (default " << hv_occlusion_th_ << ")" << std::endl;
  std::cout << "     --hv_rad_clutter val:        Clutter radius (default " << hv_rad_clutter_ << ")" << std::endl;
  std::cout << "     --hv_regularizer val:        Regularizer value (default " << hv_regularizer_ << ")" << std::endl;
  std::cout << "     --hv_rad_normals val:        Normals radius (default " << hv_rad_normals_ << ")" << std::endl;
  std::cout << "     --hv_detect_clutter val:     TRUE if clutter detect enabled (default " << hv_detect_clutter_ << ")" << std::endl << std::endl;
}

// 命令行参数解析==============================================================
void
parseCommandLine (int argc,
                  char *argv[])
{
  // 打印帮组信息Show help
  if (pcl::console::find_switch (argc, argv, "-h"))
  {
    showHelp (argv[0]);
    exit (0);
  }

  // 模型和场景文件  Model & scene filenames
  std::vector<int> filenames;
  filenames = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");
  if (filenames.size () != 2)
  {
    std::cout << "Filenames missing.\n";
    showHelp (argv[0]);
    exit (-1);
  }

  model_filename_ = argv[filenames[0]];
  scene_filename_ = argv[filenames[1]];

  // 程序 行为定义  Program behavior
  if (pcl::console::find_switch (argc, argv, "-k"))
  {
    show_keypoints_ = true;//显示关键点
  }
// 聚类算法 
  std::string used_algorithm;
  if (pcl::console::parse_argument (argc, argv, "--algorithm", used_algorithm) != -1)
  {
    if (used_algorithm.compare ("Hough") == 0)
    {
      use_hough_ = true;
    }
    else if (used_algorithm.compare ("GC") == 0)
    {
      use_hough_ = false;
    }
    else
    {
      std::cout << "Wrong algorithm name.\n";
      showHelp (argv[0]);
      exit (-1);
    }
  }

  //一般参数变量General parameters
  pcl::console::parse_argument (argc, argv, "--model_ss", model_ss_);
  pcl::console::parse_argument (argc, argv, "--scene_ss", scene_ss_);
  pcl::console::parse_argument (argc, argv, "--rf_rad", rf_rad_);
  pcl::console::parse_argument (argc, argv, "--descr_rad", descr_rad_);
  pcl::console::parse_argument (argc, argv, "--cg_size", cg_size_);
  pcl::console::parse_argument (argc, argv, "--cg_thresh", cg_thresh_);
  pcl::console::parse_argument (argc, argv, "--icp_max_iter", icp_max_iter_);
  pcl::console::parse_argument (argc, argv, "--icp_corr_distance", icp_corr_distance_);
  pcl::console::parse_argument (argc, argv, "--hv_clutter_reg", hv_clutter_reg_);
  pcl::console::parse_argument (argc, argv, "--hv_inlier_th", hv_inlier_th_);
  pcl::console::parse_argument (argc, argv, "--hv_occlusion_th", hv_occlusion_th_);
  pcl::console::parse_argument (argc, argv, "--hv_rad_clutter", hv_rad_clutter_);
  pcl::console::parse_argument (argc, argv, "--hv_regularizer", hv_regularizer_);
  pcl::console::parse_argument (argc, argv, "--hv_rad_normals", hv_rad_normals_);
  pcl::console::parse_argument (argc, argv, "--hv_detect_clutter", hv_detect_clutter_);
}

int
main (int   argc,
      char* argv[])
{
//======== 【1】命令行参数解析=================================
  parseCommandLine (argc, argv);
//======== 【2】新建必要的 指针变量============================
  pcl::PointCloud<PointType>::Ptr model (new pcl::PointCloud<PointType> ());//模型点云
  pcl::PointCloud<PointType>::Ptr model_keypoints (new pcl::PointCloud<PointType> ());//模型点云的关键点 点云
  pcl::PointCloud<PointType>::Ptr scene (new pcl::PointCloud<PointType> ());//场景点云
  pcl::PointCloud<PointType>::Ptr scene_keypoints (new pcl::PointCloud<PointType> ());//场景点云的 关键点 点云
  pcl::PointCloud<NormalType>::Ptr model_normals (new pcl::PointCloud<NormalType> ());//模型点云的 法线向量
  pcl::PointCloud<NormalType>::Ptr scene_normals (new pcl::PointCloud<NormalType> ());//场景点云的 法线向量
  pcl::PointCloud<DescriptorType>::Ptr model_descriptors (new pcl::PointCloud<DescriptorType> ());//模型点云 特征点的 特征描述子
  pcl::PointCloud<DescriptorType>::Ptr scene_descriptors (new pcl::PointCloud<DescriptorType> ());//场景点云 特征点的 特征描述子

//=======【3】载入点云=========================================
  if (pcl::io::loadPCDFile (model_filename_, *model) < 0)
  {
    std::cout << "Error loading model cloud." << std::endl;
    showHelp (argv[0]);
    return (-1);
  }
  if (pcl::io::loadPCDFile (scene_filename_, *scene) < 0)
  {
    std::cout << "Error loading scene cloud." << std::endl;
    showHelp (argv[0]);
    return (-1);
  }

//========【5】计算 法向量和曲率 ===========================================
  pcl::NormalEstimationOMP<PointType, NormalType> norm_est;//多核 计算法线模型 
  norm_est.setKSearch (10);//最近10个点 协方差矩阵PCA分解 计算 法线向量
  norm_est.setInputCloud (model);//模型点云
  norm_est.compute (*model_normals);//模型点云的法线向量

  norm_est.setInputCloud (scene);//场景点云
  norm_est.compute (*scene_normals);//场景点云的法线向量

//=======【6】下采样滤波使用均匀采样（可以试试体素格子下采样）得到关键点==========
  pcl::UniformSampling<PointType> uniform_sampling;//下采样滤波模型
  uniform_sampling.setInputCloud (model);//模型点云
  uniform_sampling.setRadiusSearch (model_ss_);//模型点云下采样滤波搜索半径
  uniform_sampling.filter (*model_keypoints);//下采样得到的关键点
  std::cout << "Model total points: " << model->size () << "; Selected Keypoints: " 
	    << model_keypoints->size () << std::endl;

  uniform_sampling.setInputCloud (scene);//场景点云
  uniform_sampling.setRadiusSearch (scene_ss_);//场景点云下采样滤波搜索半径
  uniform_sampling.filter (*scene_keypoints);//下采样得到的关键点
  std::cout << "Scene total points: " << scene->size () << "; Selected Keypoints: " 
	    << scene_keypoints->size () << std::endl;

//========【7】为keypoints关键点计算SHOT描述子Descriptor=================
  pcl::SHOTEstimationOMP<PointType, NormalType, DescriptorType> descr_est;//shot描述子
  descr_est.setRadiusSearch (descr_rad_);

  descr_est.setInputCloud (model_keypoints);//输入模型的关键点
  descr_est.setInputNormals (model_normals);//输入模型的法线
  descr_est.setSearchSurface (model);//输入的点云
  descr_est.compute (*model_descriptors);//模型点云描述子

  descr_est.setInputCloud (scene_keypoints);
  descr_est.setInputNormals (scene_normals);
  descr_est.setSearchSurface (scene);
  descr_est.compute (*scene_descriptors);//场景点云描述子

//========【8】按存储方法KDTree匹配两个点云（描述子向量匹配）点云分组得到匹配的组========
  pcl::CorrespondencesPtr model_scene_corrs (new pcl::Correspondences ());//最佳匹配点对组
  pcl::KdTreeFLANN<DescriptorType> match_search;//匹配搜索     设置配准的方法
  match_search.setInputCloud (model_descriptors);//模型点云描述子
  std::vector<int> model_good_keypoints_indices;
  std::vector<int> scene_good_keypoints_indices;
  // 为  场景点云 的每一个关键点 匹配模型点云一个 描述子最相似的 点 < 0.25f
  for (size_t i = 0; i < scene_descriptors->size (); ++i)
  {
    std::vector<int> neigh_indices (1);//设置最近邻点的索引
    std::vector<float> neigh_sqr_dists (1);//申明最近邻平方距离值
    if (!pcl_isfinite (scene_descriptors->at (i).descriptor[0]))//跳过NAN点
    {
      continue;
    }
// sscene_descriptors->at (i)是给定点云描述子, 1是临近点个数 ，neigh_indices临近点的索引  neigh_sqr_dists是与临近点的平方距离值
    int found_neighs = match_search.nearestKSearch (scene_descriptors->at (i), 1, neigh_indices, neigh_sqr_dists);
    if (found_neighs == 1 && neigh_sqr_dists[0] < 0.25f)//（描述子与临近的距离在一般在0到1之间）才添加匹配
 //在模型点云中 找 距离 场景点云点i shot描述子距离 <0.25 的点  
    {
      pcl::Correspondence corr (neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
      model_scene_corrs->push_back (corr);
      model_good_keypoints_indices.push_back (corr.index_query);//模型点云好的　关键点
      scene_good_keypoints_indices.push_back (corr.index_match);//场景点云好的管家的奶奶
    }
  }
  pcl::PointCloud<PointType>::Ptr model_good_kp (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr scene_good_kp (new pcl::PointCloud<PointType> ());
  pcl::copyPointCloud (*model_keypoints, model_good_keypoints_indices, *model_good_kp);
  pcl::copyPointCloud (*scene_keypoints, scene_good_keypoints_indices, *scene_good_kp);

  std::cout << "Correspondences found: " << model_scene_corrs->size () << std::endl;

//===========【9】实际的配准方法的实现 执行聚类================
  //变换矩阵 旋转矩阵与平移矩阵 
// 对eigen中的固定大小的类使用STL容器的时候，如果直接使用就会出错 需要使用 Eigen::aligned_allocator 对齐技术
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rototranslations;
  std::vector<pcl::Correspondences> clustered_corrs;//匹配点 相互连线的索引
// clustered_corrs[i][j].index_query 模型点 索引
// clustered_corrs[i][j].index_match 场景点 索引

  if (use_hough_)//  使用 Hough3D 3D霍夫 算法寻找匹配点
  {
    //=========计算参考帧的Hough（也就是关键点）=========
    pcl::PointCloud<RFType>::Ptr model_rf (new pcl::PointCloud<RFType> ());
    pcl::PointCloud<RFType>::Ptr scene_rf (new pcl::PointCloud<RFType> ());

    pcl::BOARDLocalReferenceFrameEstimation<PointType, NormalType, RFType> rf_est;
    rf_est.setFindHoles (true);
    rf_est.setRadiusSearch (rf_rad_);

    rf_est.setInputCloud (model_keypoints);
    rf_est.setInputNormals (model_normals);
    rf_est.setSearchSurface (model);
    rf_est.compute (*model_rf);

    rf_est.setInputCloud (scene_keypoints);
    rf_est.setInputNormals (scene_normals);
    rf_est.setSearchSurface (scene);
    rf_est.compute (*scene_rf);

//  聚类 聚类的方法 Clustering
//对输入点与的聚类，以区分不同的实例的场景中的模型
    pcl::Hough3DGrouping<PointType, PointType, RFType, RFType> clusterer;
    clusterer.setHoughBinSize (cg_size_);
    clusterer.setHoughThreshold (cg_thresh_);
    clusterer.setUseInterpolation (true);
    clusterer.setUseDistanceWeight (false);

    clusterer.setInputCloud (model_keypoints);
    clusterer.setInputRf (model_rf);
    clusterer.setSceneCloud (scene_keypoints);
    clusterer.setSceneRf (scene_rf);
    clusterer.setModelSceneCorrespondences (model_scene_corrs);

    clusterer.recognize (rototranslations, clustered_corrs);
  }
  else// 或者使用几何一致性性质 Using GeometricConsistency
  {
    pcl::GeometricConsistencyGrouping<PointType, PointType> gc_clusterer;
    gc_clusterer.setGCSize (cg_size_);//设置几何一致性的大小
    gc_clusterer.setGCThreshold (cg_thresh_);//阀值

    gc_clusterer.setInputCloud (model_keypoints);
    gc_clusterer.setSceneCloud (scene_keypoints);
    gc_clusterer.setModelSceneCorrespondences (model_scene_corrs);

    gc_clusterer.recognize (rototranslations, clustered_corrs);
  }

  /**
   * Stop if no instances
   */
  if (rototranslations.size () <= 0)
  {
    cout << "*** No instances found! ***" << endl;
    return (0);
  }
  else
  {
    cout << "Recognized Instances: " << rototranslations.size () << endl << endl;
  }

  /**
   * Generates clouds for each instances found 
   */
  std::vector<pcl::PointCloud<PointType>::ConstPtr> instances;

  for (size_t i = 0; i < rototranslations.size (); ++i)
  {
    pcl::PointCloud<PointType>::Ptr rotated_model (new pcl::PointCloud<PointType> ());
    pcl::transformPointCloud (*model, *rotated_model, rototranslations[i]);
    instances.push_back (rotated_model);
  }

//ICP 点云配准==========================================================
  std::vector<pcl::PointCloud<PointType>::ConstPtr> registered_instances;
  if (true)
  {
    cout << "--- ICP ---------" << endl;

    for (size_t i = 0; i < rototranslations.size (); ++i)
    {
      pcl::IterativeClosestPoint<PointType, PointType> icp;
      icp.setMaximumIterations (icp_max_iter_);
      icp.setMaxCorrespondenceDistance (icp_corr_distance_);
      icp.setInputTarget (scene);//场景
      icp.setInputSource (instances[i]);//场景中的实例
      pcl::PointCloud<PointType>::Ptr registered (new pcl::PointCloud<PointType>);
      icp.align (*registered);//匹配到的点云
      registered_instances.push_back (registered);
      cout << "Instance " << i << " ";
      if (icp.hasConverged ())
      {
        cout << "Aligned!" << endl;
      }
      else
      {
        cout << "Not Aligned!" << endl;
      }
    }

    cout << "-----------------" << endl << endl;
  }

// 模型假设验证  Hypothesis Verification=====================================

  cout << "--- Hypotheses Verification ---" << endl;
  std::vector<bool> hypotheses_mask;  // Mask Vector to identify positive hypotheses

  pcl::GlobalHypothesesVerification<PointType, PointType> GoHv;

  GoHv.setSceneCloud (scene);  // Scene Cloud
  GoHv.addModels (registered_instances, true);  //Models to verify

  GoHv.setInlierThreshold (hv_inlier_th_);
  GoHv.setOcclusionThreshold (hv_occlusion_th_);
  GoHv.setRegularizer (hv_regularizer_);
  GoHv.setRadiusClutter (hv_rad_clutter_);
  GoHv.setClutterRegularizer (hv_clutter_reg_);
  GoHv.setDetectClutter (hv_detect_clutter_);
  GoHv.setRadiusNormals (hv_rad_normals_);

  GoHv.verify ();
  GoHv.getMask (hypotheses_mask);  // i-element TRUE if hvModels[i] verifies hypotheses

  for (int i = 0; i < hypotheses_mask.size (); i++)
  {
    if (hypotheses_mask[i])
    {
      cout << "Instance " << i << " is GOOD! <---" << endl;
    }
    else
    {
      cout << "Instance " << i << " is bad!" << endl;
    }
  }
  cout << "-------------------------------" << endl;

//======可视化 Visualization====================================================
  pcl::visualization::PCLVisualizer viewer ("Hypotheses Verification");
  viewer.addPointCloud (scene, "scene_cloud");

  pcl::PointCloud<PointType>::Ptr off_scene_model (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr off_scene_model_keypoints (new pcl::PointCloud<PointType> ());

  pcl::PointCloud<PointType>::Ptr off_model_good_kp (new pcl::PointCloud<PointType> ());
  pcl::transformPointCloud (*model, *off_scene_model, Eigen::Vector3f (-1, 0, 0), Eigen::Quaternionf (1, 0, 0, 0));
  pcl::transformPointCloud (*model_keypoints, *off_scene_model_keypoints, Eigen::Vector3f (-1, 0, 0), Eigen::Quaternionf (1, 0, 0, 0));
  pcl::transformPointCloud (*model_good_kp, *off_model_good_kp, Eigen::Vector3f (-1, 0, 0), Eigen::Quaternionf (1, 0, 0, 0));

  if (show_keypoints_)
  {
    CloudStyle modelStyle = style_white;
    pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_color_handler (off_scene_model, modelStyle.r, modelStyle.g, modelStyle.b);
    viewer.addPointCloud (off_scene_model, off_scene_model_color_handler, "off_scene_model");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, modelStyle.size, "off_scene_model");
  }

  if (show_keypoints_)
  {
    CloudStyle goodKeypointStyle = style_violet;
    pcl::visualization::PointCloudColorHandlerCustom<PointType> model_good_keypoints_color_handler (off_model_good_kp, goodKeypointStyle.r, goodKeypointStyle.g,
                                                                                                    goodKeypointStyle.b);
    viewer.addPointCloud (off_model_good_kp, model_good_keypoints_color_handler, "model_good_keypoints");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, goodKeypointStyle.size, "model_good_keypoints");

    pcl::visualization::PointCloudColorHandlerCustom<PointType> scene_good_keypoints_color_handler (scene_good_kp, goodKeypointStyle.r, goodKeypointStyle.g,
                                                                                                    goodKeypointStyle.b);
    viewer.addPointCloud (scene_good_kp, scene_good_keypoints_color_handler, "scene_good_keypoints");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, goodKeypointStyle.size, "scene_good_keypoints");
  }

  for (size_t i = 0; i < instances.size (); ++i)
  {
    std::stringstream ss_instance;
    ss_instance << "instance_" << i;

    CloudStyle clusterStyle = style_red;
    pcl::visualization::PointCloudColorHandlerCustom<PointType> instance_color_handler (instances[i], clusterStyle.r, clusterStyle.g, clusterStyle.b);
    viewer.addPointCloud (instances[i], instance_color_handler, ss_instance.str ());
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, clusterStyle.size, ss_instance.str ());

    CloudStyle registeredStyles = hypotheses_mask[i] ? style_green : style_cyan;
    ss_instance << "_registered" << endl;
    pcl::visualization::PointCloudColorHandlerCustom<PointType> registered_instance_color_handler (registered_instances[i], registeredStyles.r,
                                                                                                   registeredStyles.g, registeredStyles.b);
    viewer.addPointCloud (registered_instances[i], registered_instance_color_handler, ss_instance.str ());
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, registeredStyles.size, ss_instance.str ());
  }

  while (!viewer.wasStopped ())
  {
    viewer.spinOnce ();
  }

  return (0);
}
