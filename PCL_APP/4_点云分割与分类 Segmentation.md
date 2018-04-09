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

[随机采样一致性 球模型 和 平面模型 SampleConsensusModelSphere Plane](Basic/Segmentation/random_sample_consensus.cpp)
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
[平面模型分割 ModelCoefficients SACMODEL_PLANE SACSegmentation ](Basic/Segmentation/planar_segmentation.cpp)	  

## 3 圆柱体分割　依据法线信息分割 平面上按　圆柱体模型分割得到圆柱体点云 pcl::SACSegmentationFromNormals 
	圆柱体分割　依据法线信息分割
	先分割平面　得到平面上的点云
	在平面上的点云中　分割圆柱体点云
	实现圆柱体模型的分割：
	采用随机采样一致性估计从带有噪声的点云中提取一个圆柱体模型。
  	        pcl::SACSegmentationFromNormals<PointT, pcl::Normal> seg;//依据法线　分割对象
		// 在平面上的点云　分割　圆柱体　
		seg.setOptimizeCoefficients (true);   //设置对估计模型优化
		seg.setModelType (pcl::SACMODEL_CYLINDER);//设置分割模型为圆柱形
		seg.setMethodType (pcl::SAC_RANSAC);      //参数估计方法　随机采样一致性算法
		seg.setNormalDistanceWeight (0.1);        //设置表面法线权重系数
		seg.setMaxIterations (10000);             //设置迭代的最大次数10000
		seg.setDistanceThreshold (0.05);          //设置内点到模型的距离允许最大值
		seg.setRadiusLimits (0, 0.1);             //设置估计出的圆柱模型的半径的范围
		seg.setInputCloud (cloud_filtered2);      //输入点云
		seg.setInputNormals (cloud_normals2);     //输入点云对应的法线特征

		// 获取符合圆柱体模型的内点　和　对应的系数
		seg.segment (*inliers_cylinder, *coefficients_cylinder);


[圆柱体分割 依据法线信息分割  ACSegmentationFromNormals SACMODEL_CYLINDER ](Basic/Segmentation/cylinder_segmentation.cpp)


### 3.1 体素格下采样 pcl::VoxelGrid －>　平面分割 pcl::SACSegmentation -> 平面模型

	// 下采样，体素叶子大小为0.01
	  pcl::VoxelGrid<pcl::PointXYZRGBA> vg;
	  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGBA>);
	  vg.setInputCloud (cloud);
	  vg.setLeafSize (0.01f, 0.01f, 0.01f);
	  vg.filter (*cloud_filtered);
	  std::cout << "PointCloud after filtering has: " << 
	  		cloud_filtered->points.size ()  << 
			" data points." << std::endl; //*
	  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
	  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
	  // Create the segmentation object
	  pcl::SACSegmentation<pcl::PointXYZRGBA> seg;
	  // Optional
	  seg.setOptimizeCoefficients (true);
	  // Mandatory
	  seg.setModelType (pcl::SACMODEL_PLANE);
	  //  seg.setModelType (pcl::SACMODEL_LINE );
	  seg.setMethodType (pcl::SAC_RANSAC);
	  seg.setDistanceThreshold (0.01);

	  seg.setInputCloud (cloud_filtered);
	  seg.segment (*inliers, *coefficients);//得到平面模型

### 3.2 体素格下采样 pcl::VoxelGrid －>　平面模型　->投影滤波 pcl::ProjectInliers ->得到平面

	// 下采样，体素叶子大小为0.01
	  pcl::VoxelGrid<pcl::PointXYZ> vg;
	  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
	  vg.setInputCloud (cloud);
	  vg.setLeafSize (0.01f, 0.01f, 0.01f);
	  vg.filter (*cloud_filtered);
	  std::cout << "PointCloud after filtering has: " << 
	  	cloud_filtered->points.size ()  << 
	 	 " data points." << 
	 	 std::endl; //*

	  // Create a set of planar coefficients with X=Y=
	  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
	  coefficients->values.resize (4);
	  coefficients->values[0] = 0.140101;
	  coefficients->values[1] = 0.126715;
	  coefficients->values[2] = 0.981995;
	  coefficients->values[3] = -0.702224;

	  // Create the filtering object
	  pcl::ProjectInliers<pcl::PointXYZ> proj;
	  proj.setModelType (pcl::SACMODEL_PLANE);
	  proj.setInputCloud (cloud_filtered);
	  proj.setModelCoefficients (coefficients);
	  proj.filter (*cloud_projected);//得到平面


## 4 欧氏距离分割 平面模型分割平面　平面上按　聚类得到　多个点云团  pcl::EuclideanClusterExtraction
	基于欧式距离的分割

	基于欧式距离的分割和基于区域生长的分割本质上都是用区分邻里关系远近来完成的。
	由于点云数据提供了更高维度的数据，故有很多信息可以提取获得。
	欧几里得算法使用邻居之间距离作为判定标准，
	而区域生长算法则利用了法线，曲率，颜色等信息来判断点云是否应该聚成一类。

### 1）欧几里德算法  pcl::EuclideanClusterExtraction
	具体的实现方法大致是（原理是将一个点云团聚合成一类）：
	    1. 找到空间中某点p10，用kdTree找到离他最近的n个点，判断这n个点到p10的距离。
		将距离小于阈值r的点p12,p13,p14....放在类Q里
	    2. 在 Q\p10 里找到一点p12,重复1
	    3. 在 Q\p10,p12 找到一点，重复1，找到p22,p23,p24....全部放进Q里
	    4. 当 Q 再也不能有新点加入了，则完成搜索了

	因为点云总是连成片的，很少有什么东西会浮在空中来区分。
	但是如果结合此算法可以应用很多东东。

	   1. 半径滤波(统计学滤波)删除离群点　体素格下采样等
	   2. 采样一致找到桌面（平面）或者除去滤波
	   3. 提取除去平面内点的　外点　（桌上的物体就自然成了一个个的浮空点云团）
	   4. 欧式聚类　提取出我们想要识别的东西
	--------------------------------------------
	 // 桌子平面上　的点云团　使用　欧式聚类的算法　kd树搜索　对点云聚类分割
	  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
	  tree->setInputCloud (cloud_filtered);//　桌子平面上其他的点云
	  std::vector<pcl::PointIndices> cluster_indices;// 点云团索引
	  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;// 欧式聚类对象
	  ec.setClusterTolerance (0.02);                    // 设置近邻搜索的搜索半径为2cm
	  ec.setMinClusterSize (100);                       // 设置一个聚类需要的最少的点数目为100
	  ec.setMaxClusterSize (25000);                     // 设置一个聚类需要的最大点数目为25000
	  ec.setSearchMethod (tree);                        // 设置点云的搜索机制
	  ec.setInputCloud (cloud_filtered);
	  ec.extract (cluster_indices);           //从点云中提取聚类，并将点云索引保存在cluster_indices中
	-----------------------------------------------------
	
### 2）条件欧几里德聚类法  pcl::ConditionalEuclideanClustering
	这个条件的设置是可以由我们自定义的，因为除了距离检查，聚类的点还需要满足一个特殊的自定义的要求，
	就是以第一个点为标准作为种子点，候选其周边的点作为它的对比或者比较的对象，
	如果满足条件就加入到聚类的对象中.

	#include <pcl/segmentation/conditional_euclidean_clustering.h>

	//如果此函数返回true，则将添加候选点到种子点的簇类中。
	bool
	customCondition(const pcl::PointXYZ& seedPoint, const pcl::PointXYZ& candidatePoint, float squaredDistance)
	{
	    // 在这里你可以添加你自定义的条件
	    if (candidatePoint.y < seedPoint.y)// 是检查候选点的Y坐标是否小于种子的Y坐标，没有什么实际意义。
		return false;
	    return true;
	}
	--------------------------------
	    // 申明一个条件聚类的对象
	    pcl::ConditionalEuclideanClustering<pcl::PointXYZ> clustering;
	    clustering.setClusterTolerance(0.02);
	    clustering.setMinClusterSize(100);
	    clustering.setMaxClusterSize(25000);
	    clustering.setInputCloud(cloud);
	    // 设置要检查每对点的函数。
	    clustering.setConditionFunction(&customCondition);//附加条件函数
	    std::vector<pcl::PointIndices> clusters;
	    clustering.segment(clusters);

	    // 对于每一个聚类结果
	    int currentClusterNum = 1;
	    for (std::vector<pcl::PointIndices>::const_iterator i = clusters.begin(); i != clusters.end(); ++i)
	    {
		// ...add all its points to a new cloud...
		pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
		for (std::vector<int>::const_iterator point = i->indices.begin(); point != i->indices.end(); point++)
		    cluster->points.push_back(cloud->points[*point]);
		cluster->width = cluster->points.size();
		cluster->height = 1;
		cluster->is_dense = true;

		// ...and save it to disk.
		if (cluster->points.size() <= 0)
		    break;
		std::cout << "Cluster " << currentClusterNum << " has " 
			  << cluster->points.size() << " points." << std::endl;
		std::string fileName = "cluster" + boost::to_string(currentClusterNum) + ".pcd";
		pcl::io::savePCDFileASCII(fileName, *cluster);

		currentClusterNum++;
	    }

[欧氏距离分割 聚类得到　多个点云团 EuclideanClusterExtraction ](Basic/Segmentation/clusters_segmentation.cpp)	 

## 5 基于　法线差值　和　曲率差值的　区域聚类分割算法 pcl::RegionGrowing
	区域生成的分割法
	区 域生长的基本 思想是： 将具有相似性的像素集合起来构成区域。
	首先对每个需要分割的区域找出一个种子像素作为生长的起点，
	然后将种子像素周围邻域中与种子有相同或相似性质的像素 
	（根据事先确定的生长或相似准则来确定）合并到种子像素所在的区域中。
	而新的像素继续作为种子向四周生长，
	直到再没有满足条件的像素可以包括进来，一个区 域就生长而成了。

	区域生长算法直观感觉上和欧几里德算法相差不大，
	都是从一个点出发，最终占领整个被分割区域，
	欧几里德算法是通过距离远近，
	对于普通点云的区域生长，其可由法线、曲率估计算法获得其法线和曲率值。
	通过法线和曲率来判断某点是否属于该类。

### 算法的主要思想是：
		首先依据点的曲率值对点进行排序，之所以排序是因为，
		区域生长算法是从曲率最小的点开始生长的，这个点就是初始种子点，
		初始种子点所在的区域即为最平滑的区域，
		从最平滑的区域开始生长可减少分割片段的总数，提高效率，
		设置一空的种子点序列和空的聚类区域，选好初始种子后，
		将其加入到种子点序列中，并搜索邻域点，
		对每一个邻域点，比较邻域点的法线与当前种子点的法线之间的夹角，
		小于平滑阀值的将当前点加入到当前区域，
		然后检测每一个邻域点的曲率值，小于曲率阀值的加入到种子点序列中，
		删除当前的种子点，循环执行以上步骤，直到种子序列为空.

### 其算法可以总结为：
	    0. 计算 法线normal 和 曲率curvatures，依据曲率升序排序；
	    1. 选择曲率最低的为初始种子点，种子周围的临近点和种子点云相比较；
	    2. 法线的方向是否足够相近（法线夹角足够 r p y），法线夹角阈值；
	    3. 曲率是否足够小(　表面处在同一个弯曲程度　)，区域差值阈值；
	    4. 如果满足2，3则该点可用做种子点;
	    5. 如果只满足2，则归类而不做种;
	    从某个种子出发，其“子种子”不再出现，则一类聚集完成
	    类的规模既不能太大也不能太小.


	  显然，上述算法是针对小曲率变化面设计的。
	尤其适合对连续阶梯平面进行分割：比如SLAM算法所获得的建筑走廊。

	  //区域增长聚类分割对象　<点，法线>
	  pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
	  reg.setMinClusterSize (50);     //最小的聚类的点数
	  reg.setMaxClusterSize (1000000);//最大的聚类的点数
	  reg.setSearchMethod (tree);     //搜索方式
	  reg.setNumberOfNeighbours (30); //设置搜索的邻域点的个数
	  reg.setInputCloud (cloud);      //输入点
	  //reg.setIndices (indices);
	  reg.setInputNormals (normals);  //输入的法线
	  reg.setSmoothnessThreshold (3.0 / 180.0 * M_PI);//设置平滑度 法线差值阈值
	  reg.setCurvatureThreshold (1.0);                //设置曲率的阀值

	  std::vector <pcl::PointIndices> clusters;
	  reg.extract (clusters);//提取点的索引


[基于法线差值和曲率差值的区域聚类分割算法 RegionGrowing ](Basic/Segmentation/region_growing_normal_cur.cpp)
 
## 6 基于颜色的　区域聚类分割算法   pcl::RegionGrowingRGB
	基于颜色的区域生长分割法
	除了普通点云之外，还有一种特殊的点云，成为RGB点云
	。显而易见，这种点云除了结构信息之外，还存在颜色信息。
	将物体通过颜色分类，是人类在辨认果实的 过程中进化出的能力，
	颜色信息可以很好的将复杂场景中的特殊物体分割出来。
	比如Xbox Kinect就可以轻松的捕捉颜色点云。
	基于颜色的区域生长分割原理上和基于曲率，法线的分割方法是一致的。
	只不过比较目标换成了颜色，去掉了点云规模上 限的限制。
	可以认为，同一个颜色且挨得近，是一类的可能性很大，不需要上限来限制。
	所以这种方式比较适合用于室内场景分割。
	尤其是复杂室内场景，颜色分割 可以轻松的将连续的场景点云变成不同的物体。
	哪怕是高低不平的地面，设法用采样一致分割器抽掉平面，
	颜色分割算法对不同的颜色的物体实现分割。

	算法分为两步：

	（1）分割，当前种子点和领域点之间色差小于色差阀值的视为一个聚类

	（2）合并，聚类之间的色差小于色差阀值和并为一个聚类，
	  且当前聚类中点的数量小于聚类点数量的与最近的聚类合并在一起

	----------------------------------------------------------
	 //基于颜色的区域生成的对象
	  pcl::RegionGrowingRGB<pcl::PointXYZRGB> reg;
	  reg.setInputCloud (cloud);
	  reg.setIndices (indices);   //点云的索引
	  reg.setSearchMethod (tree);
	  reg.setDistanceThreshold (10);//距离的阀值
	  reg.setPointColorThreshold (6);//点与点之间颜色容差
	  reg.setRegionColorThreshold (5);//区域之间容差
	  reg.setMinClusterSize (600);    //设置聚类的大小
	  std::vector <pcl::PointIndices> clusters;
	  reg.extract (clusters);//

	  pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud ();



[基于颜色的　区域聚类分割算法 RegionGrowingRGB ](Basic/Segmentation/color_based_region_growing_segmentation.cpp)

## 7 最小分割算法  (分割点云) 基于距离加权的最小图分割算法  pcl::MinCutSegmentation
	最小分割算法  (分割点云)
	该算法是将一幅点云图像分割为两部分：
	前景点云（目标物体）和背景物体（剩余部分）

[论文的地址](http://gfx.cs.princeton.edu/pubs/Golovinskiy_2009_MBS/paper_small.pdf)

	The Min-Cut (minimum cut) algorithm最小割算法是图论中的一个概念，
	其作用是以某种方式，将两个点分开，当然这两个点中间可能是通过无数的点再相连的。

	如果要分开最左边的点和最右边的点，红绿两种割法都是可行的，
	但是红线跨过了三条线，绿线只跨过了两条。
	单从跨线数量上来论可以得出绿线这种切割方法更优 的结论。
	但假设线上有不同的权值，那么最优切割则和权值有关了。
	当你给出了点之间的 “图” ，以及连线的权值时，
	最小割算法就能按照要求把图分开。


	所以那么怎么来理解点云的图呢？
	显而易见，切割有两个非常重要的因素，
	第一个是获得点与点之间的拓扑关系，这种拓扑关系就是生成一张 “图”。
	第二个是给图中的连线赋予合适的权值。
	只要这两个要素合适，最小割算法就会正确的分割出想要的结果。
	点云是分开的点。只要把点云中所有的点连起来就可以了。

	连接算法如下：
		   1. 找到每个点临近的n个点
		   2. 将这n个点和父点连接
		   3. 找到距离最小的两个块（A块中某点与B块中某点距离最小），并连接
		   4. 重复3，直至只剩一个块
		   
	经过上面的步骤现在已经有了点云的“图”，只要给图附上合适的权值，就满足了最小分割的前提条件。
	物体分割比如图像分割给人一个直观印象就是属于该物体的点，应该相互之间不会太远。
	也就是说，可以用点与点之间的欧式距离来构造权值。
	所有线的权值可映射为线长的函数。 

	cost = exp(-(dist/cet)^2)  距离越远　cost越小　越容易被分割

	我们知道这种分割是需要指定对象的，也就是我们指定聚类的中心点（center）以及聚类的半径（radius），
	当然我们指定了中心点和聚类的半径，那么就要被保护起来，保护的方法就是增加它的权值.

	dist2Center / radius

	dist2Center　＝　sqrt((x-x_center)^2+(y-y_center)^2)

	--------------------------------------------------
	// 申明一个Min-cut的聚类对象
	pcl::MinCutSegmentation<pcl::PointXYZ> clustering;
	clustering.setInputCloud(cloud);   //设置输入
	//创建一个点云，列出所知道的所有属于对象的点 
	// （前景点）在这里设置聚类对象的中心点（想想是不是可以可以使用鼠标直接选择聚类中心点的方法呢？）
	pcl::PointCloud<pcl::PointXYZ>::Ptr foregroundPoints(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointXYZ point;
	point.x = 100.0;
	point.y = 100.0;
	point.z = 100.0;
	foregroundPoints->points.push_back(point);
	clustering.setForegroundPoints(foregroundPoints);//设置聚类对象的前景点

	//设置sigma，它影响计算平滑度的成本。它的设置取决于点云之间的间隔（分辨率）
	clustering.setSigma(0.02);// cet cost = exp(-(dist/cet)^2) 
	// 设置聚类对象的半径.
	clustering.setRadius(0.01);// dist2Center / radius

	 //设置需要搜索的临近点的个数，增加这个也就是要增加边界处图的个数
	clustering.setNumberOfNeighbours(20);

	//设置前景点的权重（也就是排除在聚类对象中的点，它是点云之间线的权重，）
	clustering.setSourceWeight(0.6);

	std::vector <pcl::PointIndices> clusters;
	clustering.extract(clusters);


[最小分割 距离加权的最小图分割算法 MinCutSegmentation ](Basic/Segmentation/min_Cut_Based_Segmentation.cpp)

## 8 基于不同领域半径估计的　法线的差异类 欧氏聚类 分割 点云 pcl::DifferenceOfNormalsEstimation
	基于不同领域半径估计的　法线的差异类分割点云
[paper](https://arxiv.org/pdf/1209.1759v1.pdf)

	步骤：
		1. 估计大领域半径 r_l 下的　法线    
		2. 估计small领域半径 r_ｓ 下的　法线 
		3. 法线的差异  det(n,r_l, r_s) = (n_l - n_s)/2
		4. 条件滤波器 
		5. 欧式聚类 法线的差异

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

[基于不同半径估计的法线的差异欧氏聚类分割 DifferenceOfNormalsEstimation](Basic/Segmentation/Difference_of_Normals_in_diff_radis__Segmentation.cpp)	 

## 9 超体聚类是一种图像的分割方法   pcl::SupervoxelClustering
	超体聚类  
	超体聚类是一种图像的分割方法。

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
	-----------------------------------------------------
	//如何使用SupervoxelClustering函数
	  pcl::SupervoxelClustering<PointT> super (voxel_resolution, seed_resolution);
	  if (disable_transform)//如果设置的是参数--NT  就用默认的参数
	  super.setUseSingleCameraTransform (false);
	  super.setInputCloud (cloud);
	  super.setColorImportance (color_importance); //0.2f
	  super.setSpatialImportance (spatial_importance); //0.4f
	  super.setNormalImportance (normal_importance); //1.0f

	  std::map <uint32_t, pcl::Supervoxel<PointT>::Ptr > supervoxel_clusters;
	  super.extract (supervoxel_clusters);
	

[超体聚类是一种图像的分割方法 SupervoxelClustering ](Basic/Segmentation/supervoxel_clustering.cpp)



## 10 使用渐进式形态学滤波器 识别地面  pcl::ProgressiveMorphologicalFilter

[PCL]点云渐进形态学滤波 实现地面点分割
[paper](http://users.cis.fiu.edu/~chens/PDF/TGRS.pdf)
[reference](http://pointclouds.org/documentation/tutorials/progressive_morphological_filtering.php#progressive-morphological-filtering)

	PCL支持点云的形态学滤波，四种操作：腐蚀、膨胀、开（先腐蚀后膨胀）、闭（先膨胀后腐蚀）
	图像的膨胀:白色区域扩展黑色变小
	图像的腐蚀:白色区域变小黑色区域扩展

	在#include <pcl/filters/morphological_filter.h>中定义了枚举类型
	enum MorphologicalOperators
	  {
	    MORPH_OPEN,//开
	    MORPH_CLOSE,//闭
	    MORPH_DILATE,//膨胀　　cloud_out.points[p_idx].z = max_pt.z ();
	    MORPH_ERODE//腐蚀　　cloud_out.points[p_idx].z = min_pt.z ();
	  };

	点云渐进形态学滤波 实现地面点分割
	#include <pcl/segmentation/progressive_morphological_filter.h>

	    // 创建形态学滤波器对象
	    pcl::ProgressiveMorphologicalFilter<pcl::PointXYZ> pmf;
	    pmf.setInputCloud(cloud);
	    // 设置过滤点最大的窗口尺寸
	    pmf.setMaxWindowSize(20);
	    // 设置计算高度阈值的斜率值
	    pmf.setSlope(1.0f);
	    // 设置初始高度参数被认为是地面点
	    pmf.setInitialDistance(0.5f);
	    // 设置被认为是地面点的最大高度
	    pmf.setMaxDistance(3.0f);
	    pmf.extract(ground->indices);
	------------------------------------------

[使用渐进式形态学滤波器 识别地面 ProgressiveMorphologicalFilter ](Basic/Segmentation/ProgressiveMorphologicalFilter_segmentation.cpp)
