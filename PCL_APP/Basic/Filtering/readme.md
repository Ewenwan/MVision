# 点云滤波

##　点云滤波，顾名思义，就是滤掉噪声。原始采集的点云数据往往包含大量散列点、孤立点，


      http://blog.csdn.net/qq_34719188/article/details/79179430

      其类似于信号处理中的滤波，单实现手段却和信号处理不一样，主要有以下几方面原因：

          点云不是函数，无法建立横纵坐标之间的关系
          点云在空间中是离散的，不像图像信号有明显的定义域
          点云在空间中分布广泛，建立点与点之间的关系较为困难
          点云滤波依赖于集合信息而非数值信息

      点云滤波方法主要有: 
      直通滤波器　　pcl::PassThrough<pcl::PointXYZ> pass、
      体素格滤波器　pcl::VoxelGrid<pcl::PCLPointCloud2> sor;、
      统计滤波器    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;、
      半径滤波器    pcl::RadiusOutlierRemoval<pcl::PointXYZ> outrem;
      双边滤波  pcl::BilateralFilter<pcl::PointXYZ> bf;

      空间剪裁：
      pcl::Clipper3D<pcl::PointXYZ>
      pcl::BoxClipper3D<pcl::PointXYZ>
      pcl::CropBox<pcl::PointXYZ>
      pcl::CropHull<pcl::PointXYZ> 剪裁并形成封闭曲面
      卷积滤波:实现将两个函数通过数学运算产生第三个函数，可以设定不同的卷积核
      pcl::filters::Convolution<PointIn, PointOut>
      pcl::filters::ConvolvingKernel<PointInT, PointOutT>
      随机采样一致滤波
      等，
      通常组合使用完成任务。
      -----------------------------------------------------------------------------
      -----------------------------------------------------------------

      a. 直通滤波器 PassThrough　　　　　　　　　　　直接指定保留哪个轴上的范围内的点
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

      ----------------------------------------------------------------------------------
      ----------------------------------------------------------------------------

      b.体素格滤波器VoxelGrid　　在网格内减少点数量保证重心位置不变　PCLPointCloud2()

      #include <pcl/filters/voxel_grid.h>

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



      --------------------------------------------------------------------------
      -------------------------------------------------------------------------------

      c.统计滤波器 StatisticalOutlierRemoval
      #include <pcl/filters/statistical_outlier_removal.h>

      统计滤波器用于去除明显离群点（离群点往往由测量噪声引入）。
      其特征是在空间中分布稀疏，可以理解为：每个点都表达一定信息量，
      某个区域点越密集则可能信息量越大。噪声信息属于无用信息，信息量较小。
      所以离群点表达的信息可以忽略不计。考虑到离群点的特征，
      则可以定义某处点云小于某个密度，既点云无效。计算每个点到其最近的k(设定)个点平均距离
      。则点云中所有点的距离应构成高斯分布。给定均值与方差，可剔除ｎ个∑之外的点

      激光扫描通常会产生密度不均匀的点云数据集，另外测量中的误差也会产生稀疏的离群点，此时，估计局部点云特征（例如采样点处法向量或曲率变化率）时运算复杂，这会导致错误的数值，反过来就会导致点云配准等后期的处理失败。

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




      ----------------------------------------------------------------------
      ------------------------------------------------------------------------------
      d.球半径滤波器
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


      ------------------------------------------------------------------------------------
      -------------------------------------------------------------------------------------
      ------------------------------------------------------
      e. 条件滤波器

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


      // 创建滤波器并用条件定义对象初始化
          pcl::ConditionalRemoval<pcl::PointXYZ> condrem;//创建条件滤波器
          condrem.setCondition (range_cond); //并用条件定义对象初始化            
          condrem.setInputCloud (cloud);     //输入点云
          condrem.setKeepOrganized(true);    //设置保持点云的结构
          // 执行滤波
          condrem.filter(*cloud_filtered);  //大于0.0小于0.8这两个条件用于建立滤波器

      -----------------------------------------------------------------------
      -------------------------------------------
      f. 投影滤波　
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
