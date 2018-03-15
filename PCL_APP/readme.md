# 机器人三维视觉
## 使用PCL点云库


## 命令行安装　　编译好的二进制文件
        仓库
        sudo add-apt-repository ppa:v-launchpad-jochen-sprickerhof-de/pcl
        sudo apt-get update
        sudo apt-get install libpcl-all



## 【１】源码安装

        安装依赖
        Boost，Eigen，FlANN，VTK，OpenNI，QHull

        sudo apt-get install　build-essential　libboost-all-dev　

        sudo apt-get install libvtk5-dev 

        Vtk，（visualization toolkit 可视化工具包）是一个开源的免费软件系统，
        教程　http://blog.csdn.net/www_doling_net/article/details/8763686
        主要用于三维计算机图形学、图像处理和可视化。
        它在三维函数库OpenGL的基础上采用面向对象的设计方法发展而来，且具有跨平台的特性。 
        Vtk是在面向对象原理的基础上设计和实现的，它的内核是用C++构建的
        VTK面向对象，含有大量的对象模型。 
        源对象是可视化流水线的起点，

        映射器（Mapper）对象是可视化流水线的终点，是图形模型和可视化模型之间的接口. 
        回调（或用户方法）: 观察者监控一个对象所有被调用的事件，
        如果正在监控的一个事件被触发，一个与之相应的回调函数就会被调用。

        图形模型：
        Renderer 渲染器，vtkRenderWindow 渲染窗口

        可视化模型：
        vtkDataObject 可以被看作是一个二进制大块（blob）
        vtkProcessObject 过程对象一般也称为过滤器，按照某种运算法则对数据对象进行处理




         FLANN介绍
        FLANN库全称是Fast Library for Approximate Nearest Neighbors，
        它是目前最完整的（近似）最近邻开源库。
        http://www.cs.ubc.ca/research/flann/uploads/FLANN/flann_manual-1.8.4.pdf
        去下载　http://www.cs.ubc.ca/research/flann/#download

        linux下安装
        sudo apt-get install libeigen3-dev
        定位安装位置
        locate eigen3
        sudo updatedb


        下载源码
        git clone https://github.com/PointCloudLibrary/pcl pcl-trunk

        cd pcl-trunk && mkdir build && cd build
        cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
        make -j2
        sudo make -j2 install

        ############################################################
        ####################################################
## 【２】使用
        ####################################
## 【Ａ】点云滤波　/filters
        #####################################
        
        点云滤波，顾名思义，就是滤掉噪声。原始采集的点云数据往往包含大量散列点、孤立点，


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

### a. 直通滤波器 PassThrough　　　　　　　　　　　直接指定保留哪个轴上的范围内的点


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

### b.体素格滤波器VoxelGrid　　在网格内减少点数量保证重心位置不变　PCLPointCloud2()

        注意此点云类型为　pcl::PCLPointCloud2　类型  blob　格子类型
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





        --------------------------------------------------------------------------
        -------------------------------------------------------------------------------

### c.统计滤波器 StatisticalOutlierRemoval

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
### d.球半径滤波器

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
### e. 条件滤波器

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
###f. 投影滤波　

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

        ########################################
## B 从一个点云中提取索引
        ###################################
        
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



        ######################################################################
## 【C】PCL库使用中遇到的一些问题及解决方法
        ######################################################
        
        ----------------------------------------------
        a. pcl::PointCloud对象变量　与pcl::PointCloud::Ptr　对象指针　的相互转换

        #include <pcl/io/pcd_io.h>  
        #include <pcl/point_types.h>  
        #include <pcl/point_cloud.h>  
                                 // 对象指针
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPointer(new pcl::PointCloud<pcl::PointXYZ>);  
        pcl::PointCloud<pcl::PointXYZ> cloud;  // 对象变量
        cloud = *cloudPointer;  
        cloudPointer = cloud.makeShared(); 

        ---------------------------------------------------------------------

        b.max的宏名冲突
         解决方法：
         使用NOMINMAX宏可以防止stl库定义min和max
         或者在max出现处使用括弧标识 (std::max)(a,b)，这样可以避免编译器将max解释为宏


        -------------------------------------------------------------------
        c. C++11标准问题
            解决方法：
            boost库中vector的this报错时，建议开启C++11的支持


        ------------------------------------------------------------------

        d. flann库与opencv中的flann冲突
        解决方法：

          在flann前面加上全局命名空间 ::flann
          在要使用opencv中flann的地方，使用cv::flann这样可以避免冲突

        ----------------------------------------------------------------


        ######################################################################
## D点云关键点  /Keypoints
        ############################################################################
        
        我们都知道在二维图像上，有Harris、SIFT、SURF、KAZE这样的关键点提取算法，
        这种特征点的思想可以推广到三维空间。

        关键点的数量相比于原始点云或图像的数据量减小很多，与局部特征描述子结合在一起，
        组成　关键点描述子　常用来形成原始数据的表示，而且不失代表性和描述性，
        从而加快了后续的识别，追踪等对数据的处理了速度，
        故而，关键点技术成为在2D和3D 信息处理中非常关键的技术。

        常见的三维点云关键点提取算法有一下几种：ISS3D、Harris3D、NARF、SIFT3D
        这些算法在PCL库中都有实现，其中NARF算法是博主见过用的比较多的。




        ###############################################################
## E 点云　特征和特征描述　/Features
        ###################################################################
        
        如果要对一个三维点云进行描述，光有点云的位置是不够的，常常需要计算一些额外的参数，
        比如法线方向、曲率、文理特征等等。
        如同图像的特征一样，我们需要使用类似的方式来描述三维点云的特征。

        常用的特征描述算法有：
        法线和曲率计算、
        特征值分析、
        PFH  点特征直方图描述子、
        FPFH 跨苏点特征直方图描述子 FPFH是PFH的简化形式、
        3D Shape Context、 文理特征
        Spin Image等。




        ##########################################################
##  F 点云配准 Registration 
        #################################################
        
        给定两个来自不同坐标系的三维数据点集，找到两个点集空间的变换关系，
        使得两个点集能统一到同一坐标系统中，即配准过程
        求得旋转和平移矩阵
        P2 = R*P1  + T　　　　[R t]
        点云配准的概念也可以类比于二维图像中的配准，
        只不过二维图像配准获取得到的是x，y，alpha，beta等放射变化参数

        三维点云配准可以模拟三维点云的移动和对其，也就是会获得一个旋转矩阵和一个平移向量，
        通常表达为一个4×3的矩阵，其中3×3是旋转矩阵，1*3是平移向量。
        严格说来是6个参数，因为旋转矩阵也可以通过罗格里德斯变换转变成1*3的旋转向量。

        常用的点云配准算法有两种：
        正态分布变换方法  NDT
        和著名的ICP点云配准，此外还有许多其它算法，列举如下：
        ICP：稳健ICP、point to plane ICP、point to line ICP、MBICP、GICP

        NDT 3D、Multil-Layer NDT

        FPCS、KFPSC、SAC-IA

        Line Segment Matching、ICL

        两个数据集的计算步骤：
        1. 识别最能代表两个数据集中的场景的兴趣点（interest points）（即关键点 keypoints）
        2. 在每个关键点处，计算特征描述符;
        3. 从特征描述符集合以及它们在两个数据集中的x,y,z位置，基于特征和位置之间的相似性来估计对应关系;
        4. 假设数据被认为包含噪声的，并不是所有的对应关系都是有效的，
           所以舍弃对配准过程产生负面影响的那些负影响对应关系;
        5. 从剩下的一组好的对应关系中，估计一个变换行为。

        迭代最近点 Iterative Closest Point （ICP）
        ICP算法本质上是基于最小二乘法的最优配准方法。
        该算法重复进行选择对应关系点对，计算最优刚体变换这一过程，直到满足正确配准的收敛精度要求。
        算法的输入：参考点云和目标点云，停止迭代的标准。
        算法的输出：旋转和平移矩阵，即转换矩阵。



        #########################################################
## G  点云分割与分类 Segmentation
        #################################################################
        
        点云的分割与分类也算是一个大Topic了，这里因为多了一维就和二维图像比多了许多问题，
        点云分割又分为区域提取、线面提取、语义分割与聚类等。
        同样是分割问题，点云分割涉及面太广，确实是三言两语说不清楚的。
        只有从字面意思去理解了，遇到具体问题再具体归类。
        一般说来，点云分割是目标识别分类的基础。

        分割：区域声场、Ransac线面提取、NDT-RANSAC、K-Means、
        Normalize Cut、3D Hough Transform(线面提取)、连通分析。

        分类：基于点的分类，基于分割的分类，监督分类与非监督分类

        语义分类：
        获取场景点云之后，如何有效的利用点云信息，如何理解点云场景的内容，
        进行点云的分类很有必要，需要为每个点云进行Labeling。
        可以分为基于点的分类方法和基于分割的分类方法。
        从方法上可以分为基于监督分类的技术或者非监督分类技术，
        深度学习也是一个很有希望应用的技术


        #############################################################
## H SLAM图优化
        #####################################################
        

        SLAM技术中，在图像前端主要获取点云数据，而在后端优化主要就是依靠图优化工具。
        而SLAM技术近年来的发展也已经改变了这种技术策略。
        在过去的经典策略中，为了求解LandMark和Location，将它转化为一个稀疏图的优化，
        常常使用g2o工具来进行图优化。下面是一些常用的工具和方法。

        g2o、LUM、ELCH、Toro、SPA

        SLAM方法：ICP、MBICP、IDC、likehood Field、 Cross Correlation、NDT

        3D SLAM：
        点云匹配（最近点迭代算法 ICP、正态分布变换方法 NDT）+
        位姿图优化（g2o、LUM、ELCH、Toro、SPA）；
        实时3D SLAM算法 （LOAM）；
        Kalman滤波方法。
        3D SLAM通常产生3D点云，或者Octree Map。
        基于视觉（单目、双目、鱼眼相机、深度相机）方法的SLAM，
        比如orbSLAM，lsdSLAM...



        ##################################################
##  I 目标识别检索
        ###############################################
        
        这是点云数据处理中一个偏应用层面的问题，
        简单说来就是Hausdorff(豪斯多夫距离)距离常被用来进行深度图的目标识别和检索，
        现在很多三维人脸识别都是用这种技术来做的。

        豪斯多夫距离量度度量空间中真子集之间的距离。
        Hausdorff距离是描述两组点集之间相似程度的一种量度，
        它是两个点集之间距离的一种定义形式：


        无人驾驶汽车中基于激光数据检测场景中的行人、
        汽车、自行车、以及道路和道路附属设施（行道树、路灯、斑马线等）。



        ############################################
        J 变化检测 Octree 八叉树算法
        ########################################################

        当无序点云在连续变化中，八叉树算法常常被用于检测变化，
        这种算法需要和关键点提取技术结合起来，八叉树算法也算是经典中的经典了。




        #######################################################
## K 三维重建  SFM（运动恢复结构）
        ##########################################################
        
        我们获取到的点云数据都是一个个孤立的点，
        如何从一个个孤立的点得到整个曲面呢，这就是三维重建的topic话题。

        常用的三维重建算法和技术有：
        泊松重建、Delauary triangulatoins
        表面重建，人体重建，建筑物重建，输入重建
        实时重建：植被或者农作物的4D（3D+时间）生长态势；
        人体姿势识别，表情识别.

        多视图三维重建：计算机视觉中多视图一般利用图像信息，考虑多视几何的一些约束，
        相关研究目前很火，射影几何和多视图几何是视觉方法的基础。在摄影测量中类似的存在共线方程，
        光束平差法等研究。这里也将点云的多视匹配放在这里，比如人体的三维重建，
        点云的多视重建不再是简单的逐帧的匹配，还需要考虑不同角度观测产生误差累积，
        因此存在一个针对三维模型进行优化或者平差的Fusion融合过程在里面。
        通常SLAM是通过观测形成闭环进行整体平差实现，优先保证位姿的精确；
        而多视图重建通过FUSion过程实现对模型的整体优化，保证模型最优。
        多视图三维重建可以只使用图像，或者点云，也可以两者结合（深度图像）实现，
        重建的结果通常是Mesh网格。最典型的例子是KinectFusion，Kinfu等等.


        ##################################################
##  L 点云数据管理 
        ###################################################
        
        点云压缩，点云索引（KDtree、Octree），点云LOD（金字塔），海量点云的渲染

        KDTree　　一种递归的邻近搜索策略
         kd树（k-dimensional树的简称），是一种分割k维数据空间的数据结构。
        主要应用于多维空间关键数据的搜索（如：范围搜索和最近邻搜索）。
        其实KDTree就是二叉搜索树的变种。这里的K = 3.

        OcTree是一种更容易理解也更自然的思想。
        对于一个空间，如果某个角落里有个盒子我们却不知道在哪儿。
        但是"神"可以告诉我们这个盒子在或者不在某范围内，
        显而易见的方法就是把空间化成8个卦限，然后询问在哪个卦限内。
        再将存在的卦限继续化成8个。
        意思大概就是太极生两仪，两仪生四象，四象生八卦，
        就这么一直划分下去，最后一定会确定一个非常小的空间。
        对于点云而言，只要将点云的立方体凸包用octree生成很多很多小的卦限，
        那么在相邻卦限里的点则为相邻点。

        显然，对于不同点云应该采取不同的搜索策略，如果点云是疏散的，
        分布很广泛，且每什么规律（如lidar　雷达　测得的点云或双目视觉捕捉的点云）kdTree能更好的划分，
        而octree则很难决定最小立方体应该是多少。太大则一个立方体里可能有很多点云，太小则可能立方体之间连不起来。
        如果点云分布非常规整，是某个特定物体的点云模型，则应该使用ocTree，
        因为很容易求解凸包并且点与点之间相对距离无需再次比对父节点和子节点，更加明晰。
        典型的例子是斯坦福的兔子。



        1点云获取　(智能扫描　点云配准)
        2点云处理　( 点云去噪　特征增强　法向估计)
        3点云表示　( 点云渲染　骨架提取)
        4点云重构　(静态建模　动态建模)

        #########################################################
## [PCL]点云渐进形态学滤波
        ##############################################################
        

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


        ######################################################
# SLAM　实例
        ###########################################################


        #####################################################
## ORB SLAM
        ##########################################################


        ###################################################
##  lsd SLAM
        ###################################################



        ######################################
##  DSO SLAM　　　Direct Sparse Odometry 
        ###########################################


        ###############################################
## cartographer　

        谷歌在 2016年 10 月 6 日开源的 SLAM 算法
        基本思路和orbslam类似。
        2D激光SLAM，利用激光数据进行匹配的算法。
        官方　https://google-cartographer-ros.readthedocs.io/en/latest/tuning.html
        惯导追踪ImuTracker、
        位姿估算PoseExtrapolator、
        自适应体素滤波AdaptiveVoxelFilter、
        扫描匹配、
        子图构建、
        闭环检测和图优化。
        ##############################################
        https://github.com/hitcm/cartographer

        用Grid（2D/3D）（栅格地图）的形式建地图；
        局部匹配直接建模成一个非线性优化问题，
        利用IMU提供一个比较靠谱的初值；
        后端用Graph来优化，用分支定界算法来加速；
        2D和3D的问题统一在一个框架下解决。

        先来感受一下算法的设计目标：
        低计算资源消耗，实时优化，不追求高精度。
        这个算法的目标应用场景昭然若揭：
        室内用服务机器人（如扫地机器人）、
        无人机等等计算资源有限、对精度要求不高、
        且需要实时避障的和寻路的应用。
        特别是3D SLAM，如果能用在无人机上，岂不是叼炸天。

        Cartographer这个库最重要的东西还不是算法，而是实现。
        这玩意儿实现得太TM牛逼了，只有一个操字能形容我看到代码时的感觉。
        2D/3D的SLAM的核心部分仅仅依赖于以下几个库：
        Boost：准标准的C++库。
        Eigen3： 准标准的线性代数库。
        Lua：非常轻量的脚本语言，主要用来做Configuration
        Ceres：这是Google开源的做非线性优化的库，仅依赖于Lapack和Blas
        Protobuf：这是Google开源的很流行的跨平台通信库


        没有PCL，g2o, iSAM, sophus, OpenCV, ROS 等等，所有轮子都是自己造的。
        这明显不是搞科研的玩儿法，就是奔着产品去的。

        不用ROS、PCL、OpenCV等庞然大物也能做2D甚至3D SLAM，而且效果还不错。



        ############################################################
## Ethzasl MSF Framework 多传感器融合框架 扩展卡尔曼滤波

        多传感器卡尔曼融合框架
        Multi-sensor Fusion MSF
        http://wiki.ros.org/ethzasl_sensor_fusion/Tutorials/Introductory%20Tutorial%20for%20Multi-Sensor%20Fusion%20Framework

        #################################################################
        多传感器融合是机器人导航上面一个非常基本的问题，
        通常在一个稳定可用的机器人系统中，会使用视觉（RGB或RGBD）、激光、IMU、马盘
        等一系列传感器的数据来最终输出一个稳定和不易丢失的姿态。
        Ethzasl MSF Framework 是一个机器人上面的多传感器融合框架，它使用了扩展卡尔曼的原理对多传感器进行融合。
        同时内部使用了很多工程上的 trick 解决多传感器的刷新率同步等问题，API 封装也相对简单，非常适合新手使用。

        // ros下的多传感器融合框架使用
        mkdir -p MSF/src
        cd ./MSF/src
        catkin_init_workspace
        //编译
        cd ..
        catkin_make
        //下载
        cd .src
        // 下载依赖库
        git clone https://github.com/ethz-asl/glog_catkin.git
        git clone https://github.com/catkin/catkin_simple.git
        git clone https://github.com/ethz-asl/asctec_mav_framework.git
        // 最后下载 Ethzasl MSF Framework 框架源代码
        git clone https://github.com/ethz-asl/ethzasl_msf.git

        // 再次编译
        cd ..
        catkin_make

        运行例子： MSF Viconpos Sensor Framework（使用 ROS Bag）：
        官方的例子使用了 Vicon 的设备进行 6ROF 的姿态估计，这个传感器很专业，
        但是我们一般没有。这里面我们使用官方提供的一个 bag 文件来进行模拟。

        1）首先从 ros 网站下载 Vicon 的数据集：
        这个数据包有 3.8 MB 左右，如果速度慢的可以下载我百度网盘的文件：
        http://pan.baidu.com/s/1eShq7lg　　　　
        我这里将其放置在 PATH_TO_MSF/data 目录下面。

        2）修改 src/ethzasl_msf/msf_updates/viconpos_sensor_fix.yaml 文件：
        将其中所有的：
        /pose_sensor/pose_sensor/
        替换为：
        /msf_viconpos_sensor/pose_sensor/

        找到：	
        /pose_sensor/core/data_playback: false
        /pose_sensor/core/data_playback: true


        3）修改 src/ethzasl_msf/msf_updates/launch/viconpos_sensor.launch 文件：
        找到：
        <rosparam file="$(find msf_updates)/viconpos_sensor_fix.yaml"/>
        在这一行的前面加入两行 remap 操作，将传感器的 topic 与　ａｐｐ的 topic 对应上：	
        <remap from="/msf_core/imu_state_input" to="/auk/fcu/imu"  />
        <remap from="msf_updates/transform_input" to="/vicon/auk/auk" />

        找到：	
        </node>
        在其之后添加（这一步是初始化卡尔曼滤波器的，非常重要）：
        <node pkg="rosservice" type="rosservice" name="initialize" args="call --wait /msf_viconpos_sensor/pose_sensor/initialize_msf_scale 1"/>

        4）启动 ros 内核：
        在一个窗口打开　
        roscore

        5）启动 MSF pose_sensor 节点：
        快捷键 Ctrl + Alt + T 新建窗口，在 PATH_TO_MSF 目录下执行如下命令打开　pose_sensor 节点：	
        source devel/setup.bash 
        roslaunch msf_updates viconpos_sensor.launch

        6）打开动态配置参数功能（可选）：
        快捷键 Ctrl + Alt + T 新建窗口，执行如下命令打开动态配置功能：

        rosrun rqt_reconfigure rqt_reconfigure
        可以看到如下窗口，在窗口中选中 msf_viconpos_sensor 下面菜单：

        在菜单中即可动态设置参数。

        7）播放 vicon 的 bag 文件：
        快捷键 Ctrl + Alt + T 新建窗口，在 PATH_TO_MSF 目录下执行如下命令：

        rosbag play data/dataset.bag --pause -s 25

        这一行命令是暂停并从第 25s 后开始播放 bag 文件，
        文档中说这是为了等待 MAV 硬件系统站稳并处于非观察模式（不理解）。

        总之，如果你准备好运行了，就可以开始点击空格键进行数据播放了，播放的数据大约剩余 86s 左右

        切换到 MSF pose_sensor 节点的窗口，如果你看到输出类似如下的窗口，就是表示系统运行成功了：


        5、数据模拟：
        刚才跑成功了数据融合节点，但是并没有任何可视化的输出可以给我们看到。
        ethzasl msf 提供了一些脚本来进行数据模拟的功能，可以让我们更直观地看到结果。

        1）修改 src/ethzasl_msf/msf_core/scripts/plot_relevant 文件：
        找到：
        rxplot msf_core/state_out/data[0]:data[1]:data[2] msf_core/state_out/data[3]:data[4]:data[5] -b $T -t "position & velocity" -l px,py,pz,vx,vy,vz &
        rxplot msf_core/state_out/data[13]:data[14]:data[15] msf_core/state_out/data[16] -b $T -t "acc bias 

        修改成：
        rqt_plot msf_core/state_out/data[0]:data[1]:data[2]


        2）启动 plot_relevant 脚本：
        快捷键 Ctrl + Alt + T 新建窗口，在 PATH_TO_MSF 目录下执行如下命令打开　plot_relevant 脚本：	
        source devel/setup.bash 
        rosrun msf_core plot_relevant

        另外也可以直接在命令行运行：	
        rqt_plot msf_core/state_out/data[0]:data[1]:data[2]


        如果一切正常，即可看到如下曲线绘制，这样就表示成功运行起来了：


