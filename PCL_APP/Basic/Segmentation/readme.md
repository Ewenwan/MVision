# 点云分割

	点云分割是根据空间，几何和纹理等特征对点云进行划分，
	使得同一划分内的点云拥有相似的特征，点云的有效分割往往是许多应用的前提，
	例如逆向工作，CAD领域对零件的不同扫描表面进行分割，
	然后才能更好的进行空洞修复曲面重建，特征描述和提取，
	进而进行基于3D内容的检索，组合重用等。


	点云的分割与分类也算是一个大Topic了，这里因为多了一维就和二维图像比多了许多问题，
	点云分割又分为区域提取、线面提取、语义分割与聚类等。
	同样是分割问题，点云分割涉及面太广，确实是三言两语说不清楚的。
	只有从字面意思去理解了，遇到具体问题再具体归类。
	一般说来，点云分割是目标识别分类的基础。

## 分割：
	区域声场、
	Ransac线面提取、
	NDT-RANSAC、
	K-Means、
	Normalize Cut、
	3D Hough Transform(线面提取)、
	连通分析。

## 分类：
	基于点的分类，
	基于分割的分类，
	监督分类与非监督分类

## 语义分类：
	获取场景点云之后，如何有效的利用点云信息，如何理解点云场景的内容，
	进行点云的分类很有必要，需要为每个点云进行Labeling。
	可以分为基于点的分类方法和基于分割的分类方法。
	从方法上可以分为基于监督分类的技术或者非监督分类技术，
	深度学习也是一个很有希望应用的技术



## 随机采样一致性　采样一致性算法　sample_consensus
    在计算机视觉领域广泛的使用各种不同的采样一致性参数估计算法用于排除错误的样本，
    样本不同对应的应用不同，例如剔除错误的配准点对，分割出 处在模型上的点集，
    PCL中以随机采样一致性算法（RANSAC）为核心，
    
    同时实现了五种类似与随机采样一致形算法的随机参数估计算法:
        1. 例如随机采样一致性算法（RANSAC）、
        2. 最大似然一致性算法（MLESAC）、
        3. 最小中值方差一致性算法（LMEDS）等，
    所有估计参数算法都符合一致性原则。
    在PCL中设计的采样一致性算法的应用主要就是对点云进行分割，根据设定的不同的几个模型，
    估计对应的几何参数模型的参数，在一定容许的范围内分割出在模型上的点云。 
    
### RANSAC随机采样一致性算法的介绍
    RANSAC是“RANdom SAmple Consensus（随机抽样一致）”的缩写。
    
    它可以从一组包含“局外点”的观测数据集中，通过迭代方式估计数学模型的参数。
    
    它是一种不确定的算法——它有一定的概率得出一个合理的结果；
  
    为了提高概率必须提高迭代次数。
    数据分两种：
       有效数据（inliers）和
       无效数据（outliers）。
       
     偏差不大的数据称为有效数据，
     偏差大的数据是无效数据。
     
     如果有效数据占大多数，无效数据只是少量时，
     我们可以通过最小二乘法或类似的方法来确定模型的参数和误差；
     如果无效数据很多（比如超过了50%的数据都是无效数据），
     最小二乘法就 失效了，我们需要新的算法。


    一个简单的例子是从一组观测数据中找出合适的2维直线。
    假设观测数据中包含局内点和局外点，
    其中局内点近似的被直线所通过，而局外点远离于直线。
    简单的最 小二乘法不能找到适应于局内点的直线，原因是最小二乘法尽量去适应包括局外点在内的所有点。
    相反，RANSAC能得出一个仅仅用局内点计算出模型，并且概 率还足够高。
    但是，RANSAC并不能保证结果一定正确，为了保证算法有足够高的合理概率，我们必须小心的选择算法的参数。

    左图：包含很多局外点的数据集   右图：RANSAC找到的直线（局外点并不影响结果）
    
#### RANSAC算法概述
	RANSAC算法的输入是一组观测数据，一个可以解释或者适应于观测数据的参数化模型，一些可信的参数。
	RANSAC通过反复选择数据中的一组随机子集来达成目标。被选取的子集被假设为局内点，
	并用下述方法进行验证：
	
	1.有一个模型适应于假设的局内点，即所有的未知参数都能从假设的局内点计算得出。
	2.用1中得到的模型去测试所有的其它数据，如果某个点适用于估计的模型，认为它也是局内点。
	3.如果有足够多的点被归类为假设的局内点，那么估计的模型就足够合理。
	4.然后，用所有假设的局内点去重新估计模型，因为它仅仅被初始的假设局内点估计过。
	5.最后，通过估计局内点与模型的错误率来评估模型。
	
#### 算法
    伪码形式的算法如下所示：
##### 输入：
    data —— 一组观测数据
    model —— 适应于数据的模型
    n —— 适用于模型的最少数据个数
    k —— 算法的迭代次数
    t —— 用于决定数据是否适应于模型的阀值
    d —— 判定模型是否适用于数据集的数据数目

##### 输出：
    best_model —— 跟数据最匹配的模型参数（如果没有找到好的模型，返回null）
    best_consensus_set —— 估计出模型的数据点
    best_error —— 跟数据相关的估计出的模型错误

          开始：
    iterations = 0
    best_model = null
    best_consensus_set = null
    best_error = 无穷大
    while ( iterations < k )
        maybe_inliers = 从数据集中随机选择n个点
        maybe_model = 适合于maybe_inliers的模型参数
        consensus_set = maybe_inliers

        for ( 每个数据集中不属于maybe_inliers的点 ）
        if ( 如果点适合于maybe_model，且错误小于t ）
          将点添加到consensus_set
        if （ consensus_set中的元素数目大于d ）
        已经找到了好的模型，现在测试该模型到底有多好
        better_model = 适合于consensus_set中所有点的模型参数
        this_error = better_model究竟如何适合这些点的度量
        if ( this_error < best_error )
          我们发现了比以前好的模型，保存该模型直到更好的模型出现
          best_model =  better_model
          best_consensus_set = consensus_set
          best_error =  this_error
        增加迭代次数
    返回 best_model, best_consensus_set, best_error    
    
### 最小中值法（LMedS）
    LMedS的做法很简单，就是从样本中随机抽出N个样本子集，使用最大似然（通常是最小二乘）
    对每个子集计算模型参数和该模型的偏差，记录该模型参
    数及子集中所有样本中偏差居中的那个样本的偏差（即Med偏差），
    最后选取N个样本子集中Med偏差最小的所对应的模型参数作为我们要估计的模型参数。

### 在PCL中sample_consensus模块支持的几何模型：
    1.平面模型   SACMODEL_PLANE  参数  [normal_x normal_y normal_z d]
    
    2.线模型     SACMODEL_LINE   参数   
    [point_on_line.x point_on_line.y point_on_line.z line_direction.x line_direction.y line_direction.z]

    3.平面圆模型 SACMODEL_CIRCLE2D  参数 [center.x center.y radius]
    
    4.三维圆模型 SACMODEL_CIRCLE3D
        参数  [center.x, center.y, center.z, radius, normal.x, normal.y, normal.z]
        
    5.球模型    SACMODEL_SPHERE
    
    6.圆柱体模型 SACMODEL_CYLINDER 参数
    [point_on_axis.x point_on_axis.y point_on_axis.z axis_direction.x axis_direction.y axis_direction.z radius]
    
    7.圆锥体模型 SACMODEL_CONE  参数
    [apex.x, apex.y, apex.z, axis_direction.x, axis_direction.y, axis_direction.z, opening_angle]
    
    8.平行线     SACMODEL_PARALLEL_LINE 参数同 线模型 
    
### PCL中Sample_consensus模块及类的介绍
    PCL中Sample_consensus库实现了随机采样一致性及其泛化估计算法，
    例如平面，柱面，等各种常见的几何模型，用不同的估计算法和不同的
    几何模型自由的结合估算点云中隐含的具体几何模型的系数，
    实现对点云中所处的几何模型的分割，线，平面，柱面 ，和球面都可以在PCL 库中实现，
    平面模型经常被用到常见的室内平面的分割提取中， 比如墙，地板，桌面，
    其他模型常应用到根据几何结构检测识别和分割物体中，
    一共可以分为两类：
      一类是针对采样一致性及其泛化函数的实现，
      一类是几个不同模型的具体实现，例如：平面，直线，圆球等


    pcl::SampleConsensusModel< PointT >是随机采样一致性估计算法中不同模型实现的基类，
    所有的采样一致性估计模型都继承于此类，定义了采样一致性模型的相关的一般接口，具体实现由子类完成，其继承关系

#### 模型
    pcl::SampleConsensusModel< PointT >
      |
      |
      |
      |_______________->pcl::SampleConsensusModelPlane< PointT >
          ->pcl::SampleConsensusModelLine< PointT >
          ->pcl::SampleConsensusModelCircle2D< PointT > 实现采样一致性 计算二维平面圆周模型
          ->pcl::SampleConsensusModelCircle3D< PointT >  实现采样一致性计算的三维椎体模型
          ->pcl::SampleConsensusModelSphere< PointT >
          ->pcl::SampleConsensusModelCone< PointT ,PointNT >
          ->pcl::SampleConsensusModelCylinder< PointT ,PointNT >
          ->pcl::SampleConsensusModelRegistration< PointT >			
          ->pcl::SampleConsensusModelStick< PointT >



#### pcl::SampleConsensus< T > 是采样一致性算法的基类
    1. SampleConsensus (const SampleConsensusModelPtr &model, double threshold, bool random=false)
      其中model设置随机采样性算法使用的模型，threshold 阀值 
    2.设置模型      void     setSampleConsensusModel (const SampleConsensusModelPtr &model)
    3.设置距离阈值  void     setDistanceThreshold (double threshold)
    4.获取距离阈值  double   getDistanceThreshold ()
    5.设置最大迭代次数 void  setMaxIterations (int max_iterations)
    6.获取最大迭代次数 int   getMaxIterations ()
     
## 1 随机采样一致性 球模型 和 平面模型 pcl::SampleConsensusModelSphere  pcl::SampleConsensusModelPlane
	在没有任何参数的情况下，三维窗口显示创建的原始点云（含有局内点和局外点），
	如图所示，很明显这是一个带有噪声的菱形平面，
	噪声点是立方体，自己要是我们在产生点云是生成的是随机数生在（0，1）范围内。
	./random_sample_consensus
	./random_sample_consensus -f
	./random_sample_consensus -sf
### code	
	#include <pcl/sample_consensus/ransac.h>          // 采样一致性
	#include <pcl/sample_consensus/sac_model_plane.h> // 平面模型
	#include <pcl/sample_consensus/sac_model_sphere.h>// 球模型

	//创建随机采样一致性对象
	pcl::SampleConsensusModelSphere<pcl::PointXYZ>::Ptr
	model_s(new pcl::SampleConsensusModelSphere<pcl::PointXYZ> (cloud));   //针对球模型的对象
	pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr
	model_p (new pcl::SampleConsensusModelPlane<pcl::PointXYZ> (cloud));   //针对平面模型的对象

	//根据命令行参数，来随机估算对应平面模型，并存储估计的局内点
	pcl::RandomSampleConsensus<pcl::PointXYZ> ransac (model_p);
	ransac.setDistanceThreshold (.01);    //与平面距离小于0.01 的点称为局内点考虑
	ransac.computeModel();                //执行随机参数估计
	ransac.getInliers(inliers);           //存储估计所得的局内点

	//根据命令行参数  来随机估算对应的圆球模型，存储估计的内点
	pcl::RandomSampleConsensus<pcl::PointXYZ> ransac (model_s);
	ransac.setDistanceThreshold (.01);
	ransac.computeModel();
	ransac.getInliers(inliers);

[随机采样一致性 球模型 和 平面模型 SampleConsensusModelSphere Plane](random_sample_consensus.cpp)
---------------------------------------------------------------------

## 2 分割平面 平面模型分割  基于随机采样一致性   pcl::SACSegmentation  pcl::SACMODEL_PLANE
	平面模型分割
	
	基于随机采样一致性
	// 模型系数
	  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
	  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);//内点索引
	 // pcl::PointIndices::Ptr outliers (new pcl::PointIndices);//外点索引
	// 创建一个点云分割对象
	  pcl::SACSegmentation<pcl::PointXYZ> seg;
	  // 是否优化模型系数
	  seg.setOptimizeCoefficients (true);
	  // 设置模型　和　采样方法
	  seg.setModelType (pcl::SACMODEL_PLANE);//　平面模型
	  seg.setMethodType (pcl::SAC_RANSAC);// 随机采样一致性算法
	  seg.setDistanceThreshold (0.01);//是否在平面上的阈值

	  seg.setInputCloud (cloud);//输入点云
	  seg.segment (*inliers, *coefficients);//分割　得到平面系数　已经在平面上的点的　索引
[平面模型分割 ModelCoefficients SACMODEL_PLANE SACSegmentation ](planar_segmentation.cpp)	  

## 3 圆柱体分割　依据法线信息分割 平面上按　圆柱体模型分割得到圆柱体点云


[圆柱体分割 依据法线信息分割 ](cylinder_segmentation.cpp)

## 4 欧氏距离分割 平面模型分割平面　平面上按　聚类得到　多个点云团


[欧氏距离分割 聚类得到　多个点云团 ](clusters_segmentation.cpp)	 


## 5 基于　法线差值　和　曲率差值的　区域聚类分割算法


[基于法线差值和曲率差值的区域聚类分割算法](region_growing_normal_cur.cpp)

## 6 基于颜色的　区域聚类分割算法

[](color_based_region_growing_segmentation.cpp)

## 7 最小分割算法  (分割点云) 基于距离加权的最小图分割算法


[最小分割 距离加权的最小图分割算法 ](min_Cut_Based_Segmentation.cpp)

## 8 基于不同领域半径估计的　法线的差异类 欧氏聚类 分割 点云


[基于不同领域半径估计的　法线的差异类 欧氏聚类 分割 ](Difference_of_Normals_in_diff_radis__Segmentation.cpp)	 

## 9 超体聚类是一种图像的分割方法


[超体聚类是一种图像的分割方法 ](supervoxel_clustering.cpp)


## 10 使用渐进式形态学滤波器 识别地面 
[使用渐进式形态学滤波器 识别地面 ](ProgressiveMorphologicalFilter_segmentation.cpp)

