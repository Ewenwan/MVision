# 点云滤波

## 点云滤波，顾名思义，就是滤掉噪声。原始采集的点云数据往往包含大量散列点、孤立点，
[reference](http://blog.csdn.net/qq_34719188/article/details/79179430)

      点云滤波，顾名思义，就是滤掉噪声。原始采集的点云数据往往包含大量散列点、孤立点，
      在获取点云数据时 ，由于设备精度，操作者经验环境因素带来的影响，以及电磁波的衍射特性，
      被测物体表面性质变化和数据拼接配准操作过程的影响，点云数据中讲不可避免的出现一些噪声。
      在点云处理流程中滤波处理作为预处理的第一步，对后续的影响比较大，只有在滤波预处理中
      将噪声点 ，离群点，孔洞，数据压缩等按照后续处理定制，
      才能够更好的进行配准，特征提取，曲面重建，可视化等后续应用处理.
      其类似于信号处理中的滤波，
      
## 单实现手段却和信号处理不一样，主要有以下几方面原因：

         1. 点云不是函数，无法建立横纵坐标之间的关系
         2. 点云在空间中是离散的，不像图像信号有明显的定义域
         3. 点云在空间中分布广泛，建立点与点之间的关系较为困难
         4. 点云滤波依赖于集合信息而非数值信息

## 点云滤波方法主要有: 
	1. 直通滤波器　　pcl::PassThrough<pcl::PointXYZ> pass
      
	2. 体素格滤波器　pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
	
	3. 统计滤波器    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
      
	4. 半径滤波器    pcl::RadiusOutlierRemoval<pcl::PointXYZ> outrem;
      
	5. 双边滤波  pcl::BilateralFilter<pcl::PointXYZ> bf;
          　该类的实现利用的并非XYZ字段的数据进行，
	    而是利用强度数据进行双边滤波算法的实现，
	    所以在使用该类时点云的类型必须有强度字段，否则无法进行双边滤波处理，
          　双边滤波算法是通过取临近采样点和加权平均来修正当前采样点的位置，
	   从而达到滤波效果，同时也会有选择剔除与当前采样点“差异”太大的相邻采样点，从而保持原特征的目的 
          
	6. 高斯滤波    pcl::filters::GaussianKernel< PointInT, PointOutT >  
         　是基于高斯核的卷积滤波实现  高斯滤波相当于一个具有平滑性能的低通滤波器 

	7. 立方体滤波 pcl::CropBox< PointT>    
	   过滤掉在用户给定立方体内的点云数据
	 	
	8. 封闭曲面滤波 pcl::CropHull< PointT>   
	    过滤在给定三维封闭曲面或二维封闭多边形内部或外部的点云数据     
          
	9. 空间剪裁：
            pcl::Clipper3D<pcl::PointXYZ>
            pcl::BoxClipper3D<pcl::PointXYZ>
            pcl::CropBox<pcl::PointXYZ>
            pcl::CropHull<pcl::PointXYZ> 剪裁并形成封闭曲面
            
	10. 卷积滤波:实现将两个函数通过数学运算产生第三个函数，可以设定不同的卷积核
            pcl::filters::Convolution<PointIn, PointOut>
            pcl::filters::ConvolvingKernel<PointInT, PointOutT>
            
	11. 随机采样一致滤波
        等，
        通常组合使用完成任务。
      
## PCL中总结了几种需要进行点云滤波处理情况，这几种情况分别如下：
      （1）  点云数据密度不规则需要平滑
      （2） 因为遮挡等问题造成离群点需要去除
      （3） 大量数据需要下采样
      （4） 噪声数据需要去除
## 对应的方案如下：
      （1）按照给定的规则限制过滤去除点
      （2） 通过常用滤波算法修改点的部分属性
      （3）对数据进行下采样

    
      -----------------------------------------------------------------------------
      -----------------------------------------------------------------

##  a. 直通滤波器 pcl::PassThrough　直接指定保留哪个轴上的范围内的点
      #include <pcl/filters/passthrough.h>
      如果使用线结构光扫描的方式采集点云，必然物体沿z向分布较广，
      但x,y向的分布处于有限范围内。
      此时可使用直通滤波器，确定点云在x或y方向上的范围，
      可较快剪除离群点，达到第一步粗处理的目的。

      // 创建点云对象　指针
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
      // 原点云获取后进行滤波
      pcl::PassThrough<pcl::PointXYZ> pass;// 创建滤波器对象
      pass.setInputCloud (cloud);//设置输入点云
      pass.setFilterFieldName ("z");//滤波字段名被设置为Z轴方向
      pass.setFilterLimits (0.0, 1.0);//可接受的范围为（0.0，1.0） 
      //pass.setFilterLimitsNegative (true);//设置保留范围内 还是 过滤掉范围内
      pass.filter (*cloud_filtered); //执行滤波，保存过滤结果在cloud_filtered
[直通滤波器 PassThrough](Basic/Filtering/PassThroughfilter.cpp)
      ----------------------------------------------------------------------------------
      ----------------------------------------------------------------------------

## b.体素格滤波器VoxelGrid　下采样 　在网格内减少点数量保证重心位置不变　  pcl::VoxelGrid

      在网格内减少点数量保证重心位置不变　 
	下采样 同时去除 NAN点
      
      #include <pcl/filters/voxel_grid.h>

	// 转换为模板点云 pcl::PointCloud<pcl::PointXYZ>
	pcl::fromPCLPointCloud2 (*cloud_filtered_blob, *cloud_filtered);


      如果使用高分辨率相机等设备对点云进行采集，往往点云会较为密集。
      过多的点云数量会对后续分割工作带来困难。
      体素格滤波器可以达到向下采样同时不破坏点云本身几何结构的功能。
      点云几何结构 不仅是宏观的几何外形，也包括其微观的排列方式，
      比如横向相似的尺寸，纵向相同的距离。
      随机下采样虽然效率比体素滤波器高，但会破坏点云微观结构.

      使用体素化网格方法实现下采样，即减少点的数量 减少点云数据，
      并同时保存点云的形状特征，在提高配准，曲面重建，形状识别等算法速度中非常实用，
      PCL是实现的VoxelGrid类通过输入的点云数据创建一个三维体素栅格，
      容纳后每个体素内用体素中所有点的重心来近似显示体素中其他点，
      这样该体素内所有点都用一个重心点最终表示，对于所有体素处理后得到的过滤后的点云，
      这种方法比用体素中心（注意中心和重心）逼近的方法更慢，但是对于采样点对应曲面的表示更为准确。

      // 创建点云对象　指针
      pcl::PCLPointCloud2::Ptr cloud (new pcl::PCLPointCloud2 ());
      pcl::PCLPointCloud2::Ptr cloud_filtered (new pcl::PCLPointCloud2 ());
      // 源点云读取　获取　后
        pcl::VoxelGrid<pcl::PCLPointCloud2> sor;  //创建滤波对象
        sor.setInputCloud (cloud);            　　　　//设置需要过滤的点云(指针)　给滤波对象
        sor.setLeafSize (0.01f, 0.01f, 0.01f);  　　//设置滤波时创建的体素体积为1cm的立方体
        sor.filter (*cloud_filtered);           　　//执行滤波处理，存储输出
	
	
	//  Approximate 体素格滤波器VoxelGrid　
	pcl::ApproximateVoxelGrid<pcl::PointXYZ> approximate_voxel_filter;
	approximate_voxel_filter.setLeafSize (0.2, 0.2, 0.2);
	approximate_voxel_filter.setInputCloud (input_cloud);// 第二次扫描点云数据作为源点云
	approximate_voxel_filter.filter (*filtered_cloud);
	std::cout << "Filtered cloud contains " << filtered_cloud->size ()
	<< " data points from room_scan2.pcd" << std::endl;


[体素格滤波器 VoxelGrid](Basic/Filtering/VoxelGrid_filter.cpp)
      --------------------------------------------------------------
### 均匀采样 pcl::UniformSampling
      这个类基本上是相同的，但它输出的点云索引是选择的关键点,是在计算描述子的常见方式。
      原理同体素格 （正方体立体空间内 保留一个点（重心点））
      而 均匀采样：半径求体内 保留一个点（重心点）
      #include <pcl/filters/uniform_sampling.h>//均匀采样
      ----------------------------------------------------------
       // 创建滤波器对象　Create the filtering object
          pcl::UniformSampling<pcl::PointXYZ> filter;// 均匀采样
          filter.setInputCloud(cloud_ptr);//输入点云
          filter.setRadiusSearch(0.01f);//设置半径
          //pcl::PointCloud<int> keypointIndices;// 索引
          filter.filter(*cloud_filtered_ptr);
      -------------------------------------------

[详情：](https://www.cnblogs.com/li-yao7758258/p/6527969.html)

### 增采样  setUpsamplingMethod
      增采样是一种表面重建方法，当你有比你想象的要少的点云数据时，
      增采样可以帮你恢复原有的表面（S），通过内插你目前拥有的点云数据，
      这是一个复杂的猜想假设的过程。所以构建的结果不会百分之一百准确，
      但有时它是一种可选择的方案。
      所以，在你的点云云进行下采样时，一定要保存一份原始数据！
      #include <pcl/surface/mls.h>
      ------------
      // 滤波对象
          pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointXYZ> filter;
          filter.setInputCloud(cloud);
          //建立搜索对象
          pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree;
          filter.setSearchMethod(kdtree);
          //设置搜索邻域的半径为3cm
          filter.setSearchRadius(0.03);
          // Upsampling 采样的方法有 DISTINCT_CLOUD, RANDOM_UNIFORM_DENSITY
          filter.setUpsamplingMethod(pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointXYZ>::SAMPLE_LOCAL_PLANE);
          // 采样的半径是
          filter.setUpsamplingRadius(0.03);
          // 采样步数的大小
          filter.setUpsamplingStepSize(0.02);

          filter.process(*filteredCloud);

      --------------------------------------------------------------------------
      -------------------------------------------------------------------------------

##  c.统计滤波器  pcl::StatisticalOutlierRemoval 去除明显离群点
      #include <pcl/filters/statistical_outlier_removal.h>

      统计滤波器用于去除明显离群点（离群点往往由测量噪声引入）。
      其特征是在空间中分布稀疏，可以理解为：每个点都表达一定信息量，
      某个区域点越密集则可能信息量越大。噪声信息属于无用信息，信息量较小。
      所以离群点表达的信息可以忽略不计。考虑到离群点的特征，
      则可以定义某处点云小于某个密度，既点云无效。计算每个点到其最近的k(设定)个点平均距离
      。则点云中所有点的距离应构成高斯分布。给定均值与方差，可剔除ｎ个∑之外的点

      激光扫描通常会产生密度不均匀的点云数据集，另外测量中的误差也会产生稀疏的离群点，
      此时，估计局部点云特征（例如采样点处法向量或曲率变化率）时运算复杂，
      这会导致错误的数值，反过来就会导致点云配准等后期的处理失败。

      解决办法：对每个点的邻域进行一个统计分析，并修剪掉一些不符合标准的点。
      具体方法为在输入数据中对点到临近点的距离分布的计算，对每一个点，
      计算它到所有临近点的平均距离（假设得到的结果是一个高斯分布，
      其形状是由均值和标准差决定），那么平均距离在标准范围之外的点，
      可以被定义为离群点并从数据中去除。

      // 创建点云对象　指针
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
      // 源点云读取　获取　后
      // 创建滤波器，对每个点分析的临近点的个数设置为50 ，并将标准差的倍数设置为1  这意味着如果一
      //个点的距离超出了平均距离一个标准差以上，则该点被标记为离群点，并将它移除，存储起来
      pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;//创建滤波器对象
      sor.setInputCloud (cloud);                        //设置待滤波的点云
      sor.setMeanK (50);                               　//设置在进行统计时考虑查询点临近点数
      sor.setStddevMulThresh (1.0);                    　//设置判断是否为离群点的阀值
      sor.filter (*cloud_filtered);                    　//存储


[统计滤波器 StatisticalOutlierRemoval](Basic/Filtering/statistical_removal_filter.cpp)

      ----------------------------------------------------------------------
      ------------------------------------------------------------------------------
## d.球半径滤波器 去除离群点  去除离散点  pcl::RadiusOutlierRemoval
      #include <pcl/filters/radius_outlier_removal.h>
      
      球半径滤波器与统计滤波器相比更加简单粗暴。
      以某点为中心　画一个球计算落在该球内的点的数量，当数量大于给定值时，
      则保留该点，数量小于给定值则剔除该点。
      此算法运行速度快，依序迭代留下的点一定是最密集的，
      但是球的半径和球内点的数目都需要人工指定。

      // 创建点云对象　指针
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
      // 源点云读取　获取　后

      pcl::RadiusOutlierRemoval<pcl::PointXYZ> outrem;  //创建滤波器
      outrem.setInputCloud(cloud);    //设置输入点云
      outrem.setRadiusSearch(0.8);    //设置半径为0.8的范围内找临近点
      outrem.setMinNeighborsInRadius (2);//设置查询点的邻域点集数小于2的删除
          // apply filter
      outrem.filter (*cloud_filtered);//执行条件滤波  在半径为0.8 在此半径内必须要有两个邻居点，此点才会保存

[球半径滤波器 RadiusOutlierRemoval](Basic/Filtering/radius_outlier_filter.cpp)
      ------------------------------------------------------------------------------------
      -------------------------------------------------------------------------------------
      ------------------------------------------------------
## e. 条件滤波器 pcl::ConditionalRemoval

          可以一次删除满足对输入的点云设定的一个或多个条件指标的所有的数据点
          删除点云中不符合用户指定的一个或者多个条件的数据点

      #include <pcl/filters/conditional_removal.h>

      //创建条件限定的下的滤波器 

      // 创建点云对象　指针
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
      // 源点云读取　获取　后

      //创建条件定义对象
      pcl::ConditionAnd<pcl::PointXYZ>::Ptr range_cond (new　pcl::ConditionAnd<pcl::PointXYZ>());   
      //为条件定义对象添加比较算子
      range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZ>::ConstPtr (new
            pcl::FieldComparison<pcl::PointXYZ> ("z", pcl::ComparisonOps::GT, 0.0)));   
      //添加在Z字段上大于（pcl::ComparisonOps::GT　great Then）0的比较算子

      range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZ>::ConstPtr (new
            pcl::FieldComparison<pcl::PointXYZ> ("z", pcl::ComparisonOps::LT, 0.8)));   
      //添加在Z字段上小于（pcl::ComparisonOps::LT　Lower Then）0.8的比较算子

      // 曲率条件 
      // 创建条件定义对象  曲率
      // pcl::ConditionOr<PointNormal>::Ptr range_cond (new pcl::ConditionOr<PointNormal> () );
      // range_cond->addComparison (pcl::FieldComparison<PointNormal>::ConstPtr (// 曲率 大于 
                               new pcl::FieldComparison<PointNormal> ("curvature", pcl::ComparisonOps::GT, threshold))
                             );
      // Build the filter
      // pcl::ConditionalRemoval<PointNormal> condrem(*range_cond);
      // pcl::ConditionalRemoval<PointNormal> condrem;//创建条件滤波器
      // condrem.setCondition (range_cond);    //并用条件定义对象初始化      
      // condrem.setInputCloud (doncloud);
      // pcl::PointCloud<PointNormal>::Ptr doncloud_filtered (new pcl::PointCloud<PointNormal>);
      // Apply filter
      // condrem.filter (*doncloud_filtered);
      
      // 创建滤波器并用条件定义对象初始化
          pcl::ConditionalRemoval<pcl::PointXYZ> condrem;//创建条件滤波器
          condrem.setCondition (range_cond); //并用条件定义对象初始化            
          condrem.setInputCloud (cloud);     //输入点云
          condrem.setKeepOrganized(true);    //设置保持点云的结构
          // 执行滤波
          condrem.filter(*cloud_filtered);  //大于0.0小于0.8这两个条件用于建立滤波器
          
[条件滤波器 ConditionalRemoval](Basic/Filtering/conditional_removal_filter.cpp)
      -----------------------------------------------------------------------
      -------------------------------------------
##   f. 投影滤波　 pcl::ProjectInliers
      使用参数化模型投影点云
      如何将点投影到一个参数化模型上（平面或者球体等），
      参数化模型通过一组参数来设定，对于平面来说使用其等式形式。
      在PCL中有特定存储常见模型系数的数据结构。
      #include <iostream>
      #include <pcl/io/pcd_io.h>
      #include <pcl/point_types.h>

      #include <pcl/ModelCoefficients.h>        //模型系数头文件
      #include <pcl/filters/project_inliers.h> 　//投影滤波类头文件


      // 创建点云对象　指针
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_projected (new pcl::PointCloud<pcl::PointXYZ>);
      // 源点云读取　获取　后

        //随机创建点云并打印出来
        cloud->width  = 5;
        cloud->height = 1;
        cloud->points.resize (cloud->width * cloud->height);

        for (size_t i = 0; i < cloud->points.size (); ++i)
        {
          cloud->points[i].x = 1024 * rand () / (RAND_MAX + 1.0f);
          cloud->points[i].y = 1024 * rand () / (RAND_MAX + 1.0f);
          cloud->points[i].z = 1024 * rand () / (RAND_MAX + 1.0f);
        }


      // 填充ModelCoefficients的值,使用ax+by+cz+d=0平面模型，其中 a=b=d=0,c=1 也就是X——Y平面
        //定义模型系数对象，并填充对应的数据　创建投影滤波模型重会设置模型类型　pcl::SACMODEL_PLANE
        pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients());
        coefficients->values.resize (4);
        coefficients->values[0] = coefficients->values[1] = 0;
        coefficients->values[2] = 1.0;
        coefficients->values[3] = 0;


        // 创建投影滤波模型ProjectInliers对象，使用ModelCoefficients作为投影对象的模型参数
        pcl::ProjectInliers<pcl::PointXYZ> proj;     //创建投影滤波对象
        proj.setModelType (pcl::SACMODEL_PLANE);     //设置对象对应的投影模型　　平面模型
        proj.setInputCloud (cloud);                  //设置输入点云
        proj.setModelCoefficients (coefficients);    //设置模型对应的系数
        proj.filter (*cloud_projected);            　 //投影结果存储

      // 输出
       std::cerr << "Cloud after projection: " << std::endl;
        for (size_t i = 0; i < cloud_projected->points.size (); ++i)
          std::cerr << "    " << cloud_projected->points[i].x << " " 
                              << cloud_projected->points[i].y << " " 
                              << cloud_projected->points[i].z << std::endl;
                              
[投影滤波 ProjectInliers](Basic/Filtering/project_inliers_filter.cpp)

## g. 模型 滤波器 pcl::ModelOutlierRemoval
	  pcl::ModelCoefficients sphere_coeff;
	  sphere_coeff.values.resize (4);
	  sphere_coeff.values[0] = 0;
	  sphere_coeff.values[1] = 0;
	  sphere_coeff.values[2] = 0;
	  sphere_coeff.values[3] = 1;

	  pcl::ModelOutlierRemoval<pcl::PointXYZ> sphere_filter;
	  sphere_filter.setModelCoefficients (sphere_coeff);
	  sphere_filter.setThreshold (0.05);
	  sphere_filter.setModelType (pcl::SACMODEL_SPHERE);
	  sphere_filter.setInputCloud (cloud);
	  sphere_filter.filter (*cloud_sphere_filtered);
[模型 滤波器 ModelOutlierRemoval](Basic/Filtering/ModelOutlierRemoval.cpp)
                              
                              
 ## h. 从一个点云中提取索引 根据点云索引提取对应的点云                     
	基于某一分割算法提取点云中的一个子集
	#include <pcl/filters/voxel_grid.h>
	#include <pcl/filters/extract_indices.h>
	// 设置ExtractIndices的实际参数
	pcl::ExtractIndices<pcl::PointXYZ> extract;        //创建点云提取对象
	// 创建点云索引对象　
	pcl::PointIndices::Ptr inliers (new pcl::PointIndices());
	// 分割点云
	// 为了处理点云包含的多个模型，在一个循环中执行该过程并在每次模型被提取后，保存剩余的点进行迭代
	seg.setInputCloud (cloud_filtered);// 设置源　滤波后的点云
	seg.segment (*inliers, *coefficients);// 输入分割系数得到　分割后的点云　索引inliers
	// 提取索引　Extract the inliers
	extract.setInputCloud (cloud_filtered);
	extract.setIndices (inliers);
	extract.setNegative (false);
	extract.filter (*cloud_p);// 按索引提取　点云
	
	//　外点　绿色
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_other(new pcl::PointCloud<pcl::PointXYZ>);
	// *cloud_other = *cloud - *output;
	// 移去平面局内点，提取剩余点云
	extract_indices.setNegative (true);
	extract_indices.filter (*cloud_other);
	std::cout << "Output has " << output->points.size () << " points." << std::endl;
	  
[从一个点云中提取索引 根据点云索引提取对应的点云](Basic/Filtering/extract_indices.cpp)
                             
