/*
隐式形状模型 ISM （隐形状模型 （Implicit Shape Model））
原理类似视觉词袋模型
计算所有　训练数据　点云的　特征点和特征描述子　ｋ均值聚类　得到视觉词典

	这个算法是把Hough转换和特征近似包进行结合。
	有训练集，这个算法将计算一个确定的模型用来预测一个物体的中心。

ISM算法是 《Robust Object Detection with Interleaved Categorization and Segmentation》正式提出的。
大牛作者叫Bastian Leibe，他还写过其它几篇关于ISM算法的文章。
该算法用于行人和车的识别效果良好。
[主页](http://www.vision.rwth-aachen.de/software/ism)

Full-Resolution Residual Networks for Semantic Segmentation in Street Scenes
用于街景语义分割的全分辨率残差网络（CVPR-12）


这个算法由两部分组成，第一部分是训练，第二部分是物体识别。
训练，它有以下6步:
	1.检测关键点，keypoint detection。这只是一个训练点云的简化。
		在这个步骤里面所有的点云都将被简化，通过体素格下采样　voxel grid　这个途径。
		余下来的点就是特征点；
	2.对特征点，计算快速点特征直方图特征　FPFH，需要计算法线特征；
	3.通过k-means聚类算法对特征进行聚类得到视觉（几何）单词词典；
	4.计算每一个实例（聚类簇，一个视觉单词）里面的特征关键点　到　聚类中心关键点　的　方向向量；
	5.对每一个视觉单词，依据每个关键点和中心的方向向量，计算其统计学权重；

	6.对每一个关键点计算学习权重，与关键点到聚类中心距离有关。

我们在训练的过程结束以后，接下来就是对象搜索的进程。
	1.特征点检测。
	2.对每个特征点计算　特征描述子。
	3.对于每个特征点对应的特征描述子搜索最近的　训练阶段得到的视觉单词。
	4.对于每一个特征点计算　类别投票权重（视觉单词统计学权重　关键点学习权重）。
	5.前面的步骤给了我们一个方向集用来预测中心与能量。

上面的步骤很多涉及机器学习之类的，大致明白那个过程即可.

./implicit_shape_model
      ism_train_cat.pcd      0
      ism_train_horse.pcd    1
      ism_train_lioness.pcd  2
      ism_train_michael.pcd  3
      ism_train_wolf.pcd     4
      ism_test_cat.pcd       0
[数据](http://pointclouds.org/documentation/tutorials/implicit_shape_model.php#implicit-shape-model)


*/
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>//计算法线特征
#include <pcl/features/feature.h>
#include <pcl/visualization/cloud_viewer.h>//可视化
#include <pcl/features/fpfh.h>//　快速点特征直方图特征
#include <pcl/features/impl/fpfh.hpp>// 
#include <pcl/recognition/implicit_shape_model.h>
#include <pcl/recognition/impl/implicit_shape_model.hpp>

int
main (int argc, char** argv)
{
  if (argc == 0 || argc % 2 == 0)
    return (-1);

  unsigned int number_of_training_clouds = (argc - 3) / 2;
// 法线向量　特征　估计
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
  normal_estimator.setRadiusSearch (25.0);

  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> training_clouds;//训练点云　所有的
  std::vector<pcl::PointCloud<pcl::Normal>::Ptr> training_normals;//训练点云　法线特征
  std::vector<unsigned int> training_classes;// 类别

// 对训练数据点云　提取　法线特征
  for (unsigned int i_cloud = 0; i_cloud < number_of_training_clouds - 1; i_cloud++)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr tr_cloud(new pcl::PointCloud<pcl::PointXYZ> ());//单个点云
    if ( pcl::io::loadPCDFile <pcl::PointXYZ> (argv[i_cloud * 2 + 1], *tr_cloud) == -1 )
      return (-1);

    pcl::PointCloud<pcl::Normal>::Ptr tr_normals = (new pcl::PointCloud<pcl::Normal>)->makeShared ();//共享指针
    normal_estimator.setInputCloud (tr_cloud);
    normal_estimator.compute (*tr_normals);//估计这个　点云的　法线特征

    unsigned int tr_class = static_cast<unsigned int> (strtol (argv[i_cloud * 2 + 2], 0, 10));//类别

    training_clouds.push_back (tr_cloud);//训练点云
    training_normals.push_back (tr_normals);//点云法线
    training_classes.push_back (tr_class);//类别
  }
// 在法线特征上　提取　fphf快速点特征直方图　计算法线夹角　直方图统计
  pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::Histogram<153> >::Ptr fpfh
    (new pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::Histogram<153> >);
  fpfh->setRadiusSearch (30.0);//搜索半价
  pcl::Feature< pcl::PointXYZ, pcl::Histogram<153> >::Ptr feature_estimator(fpfh);

  pcl::ism::ImplicitShapeModelEstimation<153, pcl::PointXYZ, pcl::Normal> ism;
  ism.setFeatureEstimator(feature_estimator);//特征估计器
  ism.setTrainingClouds (training_clouds);//训练点云
  ism.setTrainingNormals (training_normals);//训练法线
  ism.setTrainingClasses (training_classes);//类别
  ism.setSamplingSize (2.0f);//采样

  pcl::ism::ImplicitShapeModelEstimation<153, pcl::PointXYZ, pcl::Normal>::ISMModelPtr model = boost::shared_ptr<pcl::features::ISMModel>
    (new pcl::features::ISMModel);//模型
  ism.trainISM (model);//训练模型

  std::string file ("trained_ism_model.txt");
  model->saveModelToFile (file);//保存模型

  model->loadModelFromfile (file);//载入模型


// 载入测试点云　
  unsigned int testing_class = static_cast<unsigned int> (strtol (argv[argc - 1], 0, 10));
  pcl::PointCloud<pcl::PointXYZ>::Ptr testing_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
  if ( pcl::io::loadPCDFile <pcl::PointXYZ> (argv[argc - 2], *testing_cloud) == -1 )
    return (-1);
//　估计测试点云　的　法线
  pcl::PointCloud<pcl::Normal>::Ptr testing_normals = (new pcl::PointCloud<pcl::Normal>)->makeShared ();
  normal_estimator.setInputCloud (testing_cloud);
  normal_estimator.compute (*testing_normals);
// 寻找点云　
  boost::shared_ptr<pcl::features::ISMVoteList<pcl::PointXYZ> > vote_list = ism.findObjects (
    model,
    testing_cloud,
    testing_normals,
    testing_class);

// 启动分类的进程。代码将会告诉算法去找testing_class类型的物体，
//在给定的testing_cloud这个点云里面。注意算法将会使用任何你放进去进行训练的模型。
//在分类操作以后，一列的决策将会以pcl::ism::ISMVoteList这个形式返回。
  double radius = model->sigmas_[testing_class] * 10.0;
  double sigma = model->sigmas_[testing_class];
  std::vector<pcl::ISMPeak, Eigen::aligned_allocator<pcl::ISMPeak> > strongest_peaks;
  vote_list->findStrongestPeaks (strongest_peaks, testing_class, radius, sigma);

  pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = (new pcl::PointCloud<pcl::PointXYZRGB>)->makeShared ();
  colored_cloud->height = 0;
  colored_cloud->width = 1;

  pcl::PointXYZRGB point;
  point.r = 255;
  point.g = 255;
  point.b = 255;

  for (size_t i_point = 0; i_point < testing_cloud->points.size (); i_point++)
  {
    point.x = testing_cloud->points[i_point].x;
    point.y = testing_cloud->points[i_point].y;
    point.z = testing_cloud->points[i_point].z;
    colored_cloud->points.push_back (point);
  }
  colored_cloud->height += testing_cloud->points.size ();

  point.r = 255;
  point.g = 0;
  point.b = 0;
  for (size_t i_vote = 0; i_vote < strongest_peaks.size (); i_vote++)
  {
    point.x = strongest_peaks[i_vote].x;
    point.y = strongest_peaks[i_vote].y;
    point.z = strongest_peaks[i_vote].z;
    colored_cloud->points.push_back (point);
  }
  colored_cloud->height += strongest_peaks.size ();

  pcl::visualization::CloudViewer viewer ("Result viewer");
  viewer.showCloud (colored_cloud);
  while (!viewer.wasStopped ())
  {
  }

  return (0);
}
