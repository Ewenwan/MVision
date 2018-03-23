/*
基于对应分组的三维物体识别
基于pcl_recognition模块的三维物体识别。
具体来说，它解释了如何使用对应分组算法，
以便将3D描述符匹配阶段之后获得的一组点到点对应集集合到当前场景中的模型实例中。
每个集群，代表一个可能的场景中的模型实例，
对应的分组算法输出的
变换矩阵识别当前的场景中，
模型的六自由度位姿估计 6DOF pose estimation 。
执行命令
./correspondence_grouping ../milk.pcd ../milk_cartoon_all_small_clorox.pcd -c -k

【5】计算法线向量  近邻邻域内 协方差矩阵PCA 降维到二维平面 计算法线向量
   PCA降维原理 http://blog.codinglabs.org/articles/pca-tutorial.html
   前面说的是二维降到一维时的情况，假如我们有一堆散乱的三维点云,则可以这样计算法线：
   1）对每一个点，取临近点，比如取最临近的50个点，当然会用到K-D树
   2）对临近点做PCA降维，把它降到二维平面上,可以想象得到这个平面一定是它的切平面(在切平面上才可以尽可能分散）
   3）切平面的法线就是该点的法线了，而这样的法线有两个，取哪个还需要考虑临近点的凸包方向
【6】下采样滤波使用均匀采样（可以试试体素格子下采样）得到关键点
【7】为keypoints关键点计算SHOT描述子
【8】按存储方法KDTree匹配两个点云（描述子向量匹配）点云分组得到匹配的组 描述 点对匹配关系
【9】参考帧霍夫聚类/集合一致性聚类得到 匹配点云cluster  平移矩阵和 匹配点对关系
【10】分组显示 平移矩阵 T 将模型点云按T变换后显示 以及显示 点对之间的连线

*/
#include <pcl/io/pcd_io.h>// 文件、设备读写
#include <pcl/point_cloud.h>//基础pcl点云类型
#include <pcl/correspondence.h>//分组算法 对应表示两个实体之间的匹配（例如，点，描述符等）。
// 特征
#include <pcl/features/normal_3d_omp.h>//法向量特征
#include <pcl/features/shot_omp.h> //描述子 shot描述子 0～1
// https://blog.csdn.net/bengyanluo1542/article/details/76061928?locationNum=9&fps=1
// (Signature of Histograms of OrienTations)方向直方图特征
#include <pcl/features/board.h>
// 滤波
#include <pcl/filters/uniform_sampling.h>//均匀采样 滤波
// 识别
#include <pcl/recognition/cg/hough_3d.h>//hough算子
#include <pcl/recognition/cg/geometric_consistency.h> //几何一致性
// 可视化
#include <pcl/visualization/pcl_visualizer.h>//可视化
// kdtree
#include <pcl/kdtree/kdtree_flann.h>// kdtree 快速近邻搜索
#include <pcl/kdtree/impl/kdtree_flann.hpp>
// 转换
#include <pcl/common/transforms.h>//点云转换 转换矩阵
// 命令行参数
#include <pcl/console/parse.h>//命令行参数解析
// 别名
/*
shot 特征描述
构造方法：以查询点p为中心构造半径为r 的球形区域，沿径向、方位、俯仰3个方向划分网格，
其中径向2次，方位8次（为简便图中径向只划分了4个），俯仰2次划分网格，
将球形区域划分成32个空间区域。
在每个空间区域计算计算落入该区域点的法线nv和中心点p法线np之间的夹角余弦cosθ=nv·np，
再根据计算的余弦值对落入每一个空间区域的点数进行直方图统计（划分11个），
对计算结果进行归一化，使得对点云密度具有鲁棒性，得到一个352维特征（32*11=352）。
（原论文：Unique Signatures of Histograms for Local Surface）
*/
typedef pcl::PointXYZRGBA PointType;//PointXYZRGBA数据结构 点类型 位置和颜色
typedef pcl::Normal NormalType;//法线类型
typedef pcl::ReferenceFrame RFType;//参考帧
typedef pcl::SHOT352 DescriptorType;//SHOT特征的数据结构（32*11=352）

std::string model_filename_;//模型的文件名
std::string scene_filename_;//场景文件名

//算法参数 Algorithm params
bool show_keypoints_ (false);
bool show_correspondences_ (false);
bool use_cloud_resolution_ (false);
bool use_hough_ (true);
float model_ss_ (0.01f);//模型采样率
float scene_ss_ (0.03f);//场景采样率
float rf_rad_ (0.015f);
float descr_rad_ (0.02f);
float cg_size_ (0.01f);//聚类 霍夫空间设置每个bin的大小
float cg_thresh_ (5.0f);//聚类阈值

// 打印帮组信息 程序用法
void
showHelp (char *filename)
{
  std::cout << std::endl;
  std::cout << "***************************************************************************" << std::endl;
  std::cout << "*                                                                         *" << std::endl;
  std::cout << "*             Correspondence Grouping Tutorial - Usage Guide              *" << std::endl;
  std::cout << "*                                                                         *" << std::endl;
  std::cout << "***************************************************************************" << std::endl << std::endl;
  std::cout << "Usage: " << filename << " model_filename.pcd scene_filename.pcd [Options]" << std::endl << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "     -h:                     Show this help." << std::endl;
  std::cout << "     -k:                     Show used keypoints." << std::endl;//关键点
  std::cout << "     -c:                     Show used correspondences." << std::endl;//分组算法
  std::cout << "     -r:                     Compute the model cloud resolution and multiply" << std::endl;
  std::cout << "                             each radius given by that value." << std::endl;
  std::cout << "     --algorithm (Hough|GC): Clustering algorithm used (default Hough)." << std::endl;//聚类算法
  std::cout << "     --model_ss val:         Model uniform sampling radius (default 0.01)" << std::endl;//模型采样率
  std::cout << "     --scene_ss val:         Scene uniform sampling radius (default 0.03)" << std::endl;//场景采样率
  std::cout << "     --rf_rad val:           Reference frame radius (default 0.015)" << std::endl;//参考帧 半径
  std::cout << "     --descr_rad val:        Descriptor radius (default 0.02)" << std::endl;//描述子计算半径
  std::cout << "     --cg_size val:          Cluster size (default 0.01)" << std::endl;//聚类
  std::cout << "     --cg_thresh val:        Clustering threshold (default 5)" << std::endl << std::endl;//聚类数量阈值
}

// 命令行参数解析
void
parseCommandLine (int argc, char *argv[])
{
  // -h 打印帮组信息Show help
  if (pcl::console::find_switch (argc, argv, "-h"))
  {
    showHelp (argv[0]);
    exit (0);
  }

  //  模型和场景文件 Model & scene filenames
  std::vector<int> filenames;
  filenames = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");
  if (filenames.size () != 2)
  {
    std::cout << "Filenames missing.\n";
    showHelp (argv[0]);
    exit (-1);
  }

  model_filename_ = argv[filenames[0]];//模型文件
  scene_filename_ = argv[filenames[1]];//场景文件

  //程序 行为定义 Program behavior
  if (pcl::console::find_switch (argc, argv, "-k"))
  {
    show_keypoints_ = true;//显示关键点
  }
  if (pcl::console::find_switch (argc, argv, "-c"))
  {
    show_correspondences_ = true;//显示对应分组
  }
  if (pcl::console::find_switch (argc, argv, "-r"))
  {
    use_cloud_resolution_ = true;//点云分辨率
  }
  // 聚类算法 
  std::string used_algorithm;
  if (pcl::console::parse_argument (argc, argv, "--algorithm", used_algorithm) != -1)
  {
    if (used_algorithm.compare ("Hough") == 0)
    {
      use_hough_ = true;
    }else if (used_algorithm.compare ("GC") == 0)
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
}

// 计算点云分辨率 点云 每个点距离最近点之间的距离和 的平均值
double
computeCloudResolution (const pcl::PointCloud<PointType>::ConstPtr &cloud)
{
  double res = 0.0;
  int n_points = 0;
  int nres;//临近点的索引
  std::vector<int> indices (2);// 索引
  std::vector<float> sqr_distances (2);//距离平方 
  pcl::search::KdTree<PointType> tree;//搜索方法 kdtree
  tree.setInputCloud (cloud);//输入点云

  for (size_t i = 0; i < cloud->size (); ++i)//遍历每一个点
  {
    if (! pcl_isfinite ((*cloud)[i].x))//剔除 NAN点
    {
      continue;
    }
    //Considering the second neighbor since the first is the point itself.
    nres = tree.nearestKSearch (i, 2, indices, sqr_distances);//得到的是距离平方
    if (nres == 2)//最近点第一个为自身第二个为除了自己离自己最近的一个点
    {
      res += sqrt (sqr_distances[1]);//开根号后想家
      ++n_points;
    }
  }
  if (n_points != 0)
  {
    res /= n_points;//最近点之间的距离 的平均值
  }
  return res;
}

// 主函数
int
main (int argc, char *argv[])
{
//======== 【1】命令行参数解析========================
  parseCommandLine (argc, argv);

//======== 【2】新建必要的 指针变量===================
  pcl::PointCloud<PointType>::Ptr model (new pcl::PointCloud<PointType> ());//模型点云
  pcl::PointCloud<PointType>::Ptr model_keypoints (new pcl::PointCloud<PointType> ());//模型点云的关键点 点云
  pcl::PointCloud<PointType>::Ptr scene (new pcl::PointCloud<PointType> ());//场景点云
  pcl::PointCloud<PointType>::Ptr scene_keypoints (new pcl::PointCloud<PointType> ());//场景点云的 关键点 点云
  pcl::PointCloud<NormalType>::Ptr model_normals (new pcl::PointCloud<NormalType> ());//模型点云的 法线向量
  pcl::PointCloud<NormalType>::Ptr scene_normals (new pcl::PointCloud<NormalType> ());//场景点云的 法线向量
  pcl::PointCloud<DescriptorType>::Ptr model_descriptors (new pcl::PointCloud<DescriptorType> ());//模型点云 特征点的 特征描述子
  pcl::PointCloud<DescriptorType>::Ptr scene_descriptors (new pcl::PointCloud<DescriptorType> ());//场景点云 特征点的 特征描述子

  //
  //=======【3】载入点云==========================
  //
  if (pcl::io::loadPCDFile (model_filename_, *model) < 0)//模型点云
  {
    std::cout << "Error loading model cloud." << std::endl;
    showHelp (argv[0]);
    return (-1);
  }
  if (pcl::io::loadPCDFile (scene_filename_, *scene) < 0)//场景点云
  {
    std::cout << "Error loading scene cloud." << std::endl;
    showHelp (argv[0]);
    return (-1);
  }

  //
  //======【4】设置分辨率 变量==========================
  //
  if (use_cloud_resolution_)//使用分辨率
  {
    float resolution = static_cast<float> (computeCloudResolution (model));//计算分辨率
    if (resolution != 0.0f)
    {
      model_ss_   *= resolution;//更新参数
      scene_ss_   *= resolution;
      rf_rad_     *= resolution;
      descr_rad_  *= resolution;
      cg_size_    *= resolution;
    }

    std::cout << "Model resolution:       " << resolution << std::endl;
    std::cout << "Model sampling size:    " << model_ss_  << std::endl;
    std::cout << "Scene sampling size:    " << scene_ss_  << std::endl;
    std::cout << "LRF support radius:     " << rf_rad_    << std::endl;
    std::cout << "SHOT descriptor radius: " << descr_rad_ << std::endl;
    std::cout << "Clustering bin size:    " << cg_size_   << std::endl << std::endl;
  }

  //
  //========【5】计算法线向量============== 
  //
  pcl::NormalEstimationOMP<PointType, NormalType> norm_est;//多核 计算法线模型 OpenMP
//  pcl::NormalEstimation<PointType, NormalType> norm_est;//多核 计算法线模型 OpenMP
  norm_est.setKSearch (10);//最近10个点 协方差矩阵PCA分解 计算 法线向量
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
  //norm_est.setSearchMethod (tree);// 多核模式 不需要设置 搜索算法
  norm_est.setInputCloud (model);//模型点云
  norm_est.compute (*model_normals);//模型点云的法线向量

  norm_est.setInputCloud (scene);//场景点云
  norm_est.compute (*scene_normals);//场景点云的法线向量

  //
  //=======【6】下采样滤波使用均匀采样（可以试试体素格子下采样）得到关键点=========
  //

  pcl::UniformSampling<PointType> uniform_sampling;//下采样滤波模型
  uniform_sampling.setInputCloud (model);//模型点云
  uniform_sampling.setRadiusSearch (model_ss_);//模型点云搜索半径
  uniform_sampling.filter (*model_keypoints);//下采样得到的关键点
  std::cout << "Model total points: " << model->size () << "; Selected Keypoints: " << model_keypoints->size () << std::endl;

  uniform_sampling.setInputCloud (scene);//场景点云
  uniform_sampling.setRadiusSearch (scene_ss_);//点云搜索半径
  uniform_sampling.filter (*scene_keypoints);//下采样得到的关键点
  std::cout << "Scene total points: " << scene->size () << "; Selected Keypoints: " << scene_keypoints->size () << std::endl;


  //
  //========【7】为keypoints关键点计算SHOT描述子Descriptor===========
  //
  pcl::SHOTEstimationOMP<PointType, NormalType, DescriptorType> descr_est;//shot描述子
  descr_est.setRadiusSearch (descr_rad_);

  descr_est.setInputCloud (model_keypoints);
  descr_est.setInputNormals (model_normals);
  descr_est.setSearchSurface (model);
  descr_est.compute (*model_descriptors);//模型点云描述子

  descr_est.setInputCloud (scene_keypoints);
  descr_est.setInputNormals (scene_normals);
  descr_est.setSearchSurface (scene);
  descr_est.compute (*scene_descriptors);//场景点云描述子

  //
  //========【8】按存储方法KDTree匹配两个点云（描述子向量匹配）点云分组得到匹配的组====== 
  //
  pcl::CorrespondencesPtr model_scene_corrs (new pcl::Correspondences ());//最佳匹配点对组

  pcl::KdTreeFLANN<DescriptorType> match_search;//匹配搜索
  match_search.setInputCloud (model_descriptors);//模型点云描述子
  // 在 场景点云中 为 模型点云的每一个关键点 匹配一个 描述子最相似的 点
  for (size_t i = 0; i < scene_descriptors->size (); ++i)//遍历场景点云
  {
    std::vector<int> neigh_indices (1);//索引
    std::vector<float> neigh_sqr_dists (1);//描述子距离
    if (!pcl_isfinite (scene_descriptors->at(i).descriptor[0])) //跳过NAN点
    {
      continue;
    }
    int found_neighs = match_search.nearestKSearch (scene_descriptors->at (i), 1, neigh_indices, neigh_sqr_dists);
    if(found_neighs == 1 && neigh_sqr_dists[0] < 0.25f) //在模型点云中 找 距离 场景点云点i shot描述子距离 <0.25 的点  
    {
      pcl::Correspondence corr (neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
      //   neigh_indices[0] 为模型点云中 和 场景点云 点   scene_descriptors->at (i) 最佳的匹配 距离为 neigh_sqr_dists[0]  
      model_scene_corrs->push_back (corr);
    }
  }
  std::cout << "Correspondences found: " << model_scene_corrs->size () << std::endl;//匹配点云对 数量

  //
  //===========【9】执行聚类================
  //
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rototranslations;//变换矩阵 旋转矩阵与平移矩阵
// 对eigen中的固定大小的类使用STL容器的时候，如果直接使用就会出错 需要使用 Eigen::aligned_allocator 对齐技术
  std::vector<pcl::Correspondences> clustered_corrs;//匹配点 相互连线的索引
// clustered_corrs[i][j].index_query 模型点 索引
// clustered_corrs[i][j].index_match 场景点 索引

  //  使用 Hough3D 3D霍夫 算法寻找匹配点
  if (use_hough_)
  {
    //
    //=========计算参考帧的Hough（也就是关键点）=========
    //
    pcl::PointCloud<RFType>::Ptr model_rf (new pcl::PointCloud<RFType> ());//模型参考帧
    pcl::PointCloud<RFType>::Ptr scene_rf (new pcl::PointCloud<RFType> ());//场景参考帧
    //======估计模型参考帧（特征估计的方法（点云，法线，参考帧）
    pcl::BOARDLocalReferenceFrameEstimation<PointType, NormalType, RFType> rf_est;
    rf_est.setFindHoles (true);
    rf_est.setRadiusSearch (rf_rad_); //设置搜索半径

    rf_est.setInputCloud (model_keypoints);//模型关键点
    rf_est.setInputNormals (model_normals);//法线向量
    rf_est.setSearchSurface (model);//模型点云
    rf_est.compute (*model_rf);//计算模型参考帧

    rf_est.setInputCloud (scene_keypoints);//场景关键点
    rf_est.setInputNormals (scene_normals);//法线向量
    rf_est.setSearchSurface (scene);//场景点云
    rf_est.compute (*scene_rf);//场景参考帧

    //  聚类 聚类的方法 Clustering
   //对输入点与的聚类，以区分不同的实例的场景中的模型
    pcl::Hough3DGrouping<PointType, PointType, RFType, RFType> clusterer;
    clusterer.setHoughBinSize (cg_size_);//霍夫空间设置每个bin的大小
    clusterer.setHoughThreshold (cg_thresh_);//阈值
    clusterer.setUseInterpolation (true);
    clusterer.setUseDistanceWeight (false);

    clusterer.setInputCloud (model_keypoints);//模型点云 关键点
    clusterer.setInputRf (model_rf);//模型点云参考帧
    clusterer.setSceneCloud (scene_keypoints);//场景点云关键点
    clusterer.setSceneRf (scene_rf);//场景点云参考帧
    clusterer.setModelSceneCorrespondences (model_scene_corrs);//对于组关系

    //clusterer.cluster (clustered_corrs);//辨认出聚类的对象
    clusterer.recognize (rototranslations, clustered_corrs);
  }
  else // 或者使用几何一致性性质 Using GeometricConsistency
  {
    pcl::GeometricConsistencyGrouping<PointType, PointType> gc_clusterer;
    gc_clusterer.setGCSize (cg_size_);//设置几何一致性的大小
    gc_clusterer.setGCThreshold (cg_thresh_);//阀值

    gc_clusterer.setInputCloud (model_keypoints);
    gc_clusterer.setSceneCloud (scene_keypoints);
    gc_clusterer.setModelSceneCorrespondences (model_scene_corrs);

    //gc_clusterer.cluster (clustered_corrs);//辨认出聚类的对象
    gc_clusterer.recognize (rototranslations, clustered_corrs);
  }

  //
  //========【10】输出识别结果=====Output results=====
  // 
  // 找出输入模型是否在场景中出现
  std::cout << "Model instances found: " << rototranslations.size () << std::endl;
  for (size_t i = 0; i < rototranslations.size (); ++i)
  {
    std::cout << "\n    Instance " << i + 1 << ":" << std::endl;
    std::cout << "        Correspondences belonging to this instance: " << clustered_corrs[i].size () << std::endl;

    // 打印 相对于输入模型的旋转矩阵与平移矩阵  rotation matrix and translation vector
    // [R t]
    Eigen::Matrix3f rotation = rototranslations[i].block<3,3>(0, 0);//旋转矩阵
    Eigen::Vector3f translation = rototranslations[i].block<3,1>(0, 3);//平移向量

    printf ("\n");
    printf ("            | %6.3f %6.3f %6.3f | \n", rotation (0,0), rotation (0,1), rotation (0,2));
    printf ("        R = | %6.3f %6.3f %6.3f | \n", rotation (1,0), rotation (1,1), rotation (1,2));
    printf ("            | %6.3f %6.3f %6.3f | \n", rotation (2,0), rotation (2,1), rotation (2,2));
    printf ("\n");
    printf ("        t = < %0.3f, %0.3f, %0.3f >\n", translation (0), translation (1), translation (2));
  }

  //
  //======可视化 Visualization===============================
  //
  pcl::visualization::PCLVisualizer viewer ("Correspondence Grouping");//对应组
  viewer.addPointCloud (scene, "scene_cloud");//添加场景点云

  pcl::PointCloud<PointType>::Ptr off_scene_model (new pcl::PointCloud<PointType> ());// 模型点云 变换后的点云
  pcl::PointCloud<PointType>::Ptr off_scene_model_keypoints (new pcl::PointCloud<PointType> ());//关键点

  if (show_correspondences_ || show_keypoints_) //可视化 平移后的模型点云
  {
    //  We are translating the model so that it doesn't end in the middle of the scene representation
     //就是要对输入的模型进行旋转与平移，使其在可视化界面的中间位置 x轴负方向平移1个单位 
    pcl::transformPointCloud (*model, *off_scene_model, Eigen::Vector3f (-1,0,0), Eigen::Quaternionf (1, 0, 0, 0));
    pcl::transformPointCloud (*model_keypoints, *off_scene_model_keypoints, Eigen::Vector3f (-1,0,0), Eigen::Quaternionf (1, 0, 0, 0));

    pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_color_handler (off_scene_model, 255, 255, 128);
    viewer.addPointCloud (off_scene_model, off_scene_model_color_handler, "off_scene_model");//显示平移后的模型点云
  }

  if (show_keypoints_)//可视化 场景关键点 和 模型关键点
  {
    pcl::visualization::PointCloudColorHandlerCustom<PointType> scene_keypoints_color_handler (scene_keypoints, 0, 0, 255);
    viewer.addPointCloud (scene_keypoints, scene_keypoints_color_handler, "scene_keypoints");//可视化场景关键点
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "scene_keypoints");
    // 可视化点参数 大小

    pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_keypoints_color_handler (off_scene_model_keypoints, 0, 0, 255);//可视化模型关键点
    viewer.addPointCloud (off_scene_model_keypoints, off_scene_model_keypoints_color_handler, "off_scene_model_keypoints");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene_model_keypoints");
  }

  for (size_t i = 0; i < rototranslations.size (); ++i)//对于 模型在场景中 匹配的 点云
  {
    pcl::PointCloud<PointType>::Ptr rotated_model (new pcl::PointCloud<PointType> ());//按匹配变换矩阵 模型点云
    pcl::transformPointCloud (*model, *rotated_model, rototranslations[i]);//将模型点云按匹配的变换矩阵旋转

    std::stringstream ss_cloud;//字符串输出流
    ss_cloud << "instance" << i;//识别出的实例

    pcl::visualization::PointCloudColorHandlerCustom<PointType> rotated_model_color_handler (rotated_model, 255, 0, 0);
    viewer.addPointCloud (rotated_model, rotated_model_color_handler, ss_cloud.str ());//添加模型按识别 变换矩阵变换后 显示

    if (show_correspondences_)//显示匹配点 连线
    {
      for (size_t j = 0; j < clustered_corrs[i].size (); ++j)
      {
        std::stringstream ss_line;//匹配点 连线 字符串
        ss_line << "correspondence_line" << i << "_" << j;
        PointType& model_point = off_scene_model_keypoints->at (clustered_corrs[i][j].index_query);//模型点
        PointType& scene_point = scene_keypoints->at (clustered_corrs[i][j].index_match);//场景点

        //  显示点云匹配对中每一对匹配点对之间的连线
        viewer.addLine<PointType, PointType> (model_point, scene_point, 0, 255, 0, ss_line.str());
      }
    }
  }

  while (!viewer.wasStopped ())
  {
    viewer.spinOnce ();
  }

  return (0);
}
