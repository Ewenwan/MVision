# 机器人三维视觉

[三维计算机视觉  pcl 滤波分割聚类关键点 深度学习 点云卷积网络](https://www.cnblogs.com/ironstark/category/759418.html)
 
[GeometryHub(几何空间) 点云处理库 ](http://geometryhub.net/myspace)
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

        sudo apt-get install libvtk5-dev libopenni-dev

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

        //注意　PCL_ROS 其实引用了PCL库，不要随意编译PCL库，可能导致PCL-ROS不能使用！    
        // PCL自动安装的时候与C11不兼容，如果想使用C11,需要自己编译PCL库，
        //并在PCL编译之前的CMakelist.txt中加入C11的编译项！

        //所以　ros项目中如果使用　支持　c++11　
        // 那么使用pcl时pcl必须源码编译，并且需要修改pcl源码的　CMakelist.txt　 加入支持c++11的选项
        //# 添加c++ 11标准支持
        //set( CMAKE_CXX_FLAGS "-std=c++11" )
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
            该类的实现利用的并非XYZ字段的数据进行，
        而是利用强度数据进行双边滤波算法的实现，所以在使用该类时点云的类型必须有强度字段，否则无法进行双边滤波处理，


        立方体滤波 pcl::CropBox< PointT>    
           过滤掉在用户给定立方体内的点云数据

        封闭曲面滤波 pcl::CropHull< PointT>   
            过滤在给定三维封闭曲面或二维封闭多边形内部或外部的点云数据
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
[直通滤波器 PassThrough](Basic/Filtering/PassThroughfilter.cpp)

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
[体素格滤波器VoxelGrid](Basic/Filtering/VoxelGrid_filter.cpp)

        下采样 同时去除 NAN点
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
[统计滤波器 StatisticalOutlierRemoval](Basic/Filtering/statistical_removal_filter.cpp)

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
[球半径滤波器 radius ](Basic/Filtering/radius_outlier_filter.cpp)

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
[条件滤波器 conditional_removal_filter ](Basic/Filtering/conditional_removal_filter.cpp)

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
[投影滤波　project_inliers_filter](Basic/Filtering/project_inliers_filter.cpp)

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
        ------------------------------------------------------------------

        e. 去除NAN点
        // STL
        #include <iostream>
        #include <limits>

        // PCL
        #include <pcl/point_cloud.h>
        #include <pcl/point_types.h>
        #include <pcl/filters/filter.h>

        int
        main (int, char**)
        {
          typedef pcl::PointCloud<pcl::PointXYZ> CloudType;
          CloudType::Ptr cloud (new CloudType);
          cloud->is_dense = false;
          CloudType::Ptr output_cloud (new CloudType);

          CloudType::PointType p_nan;
          p_nan.x = std::numeric_limits<float>::quiet_NaN();
          p_nan.y = std::numeric_limits<float>::quiet_NaN();
          p_nan.z = std::numeric_limits<float>::quiet_NaN();
          cloud->push_back(p_nan);

          CloudType::PointType p_valid;
          p_valid.x = 1.0f;
          cloud->push_back(p_valid);

          std::cout << "size: " << cloud->points.size () << std::endl;

          std::vector<int> indices;//索引
          pcl::removeNaNFromPointCloud(*cloud, *output_cloud, indices);//去除NAN点
          std::cout << "size: " << output_cloud->points.size () << std::endl;

          return 0;
        }
        ----------------------------------------------------------------

        ######################################################################
## D点云关键点  /Keypoints   ISS3D、 Harris3D、 NARF、SIFT3D 
        ############################################################################
        我们都知道在二维图像上，有Harris、SIFT、SURF、KAZE这样的关键点提取算法，
        这种特征点的思想可以推广到三维空间。

        关键点的数量相比于原始点云或图像的数据量减小很多，与局部特征描述子结合在一起，
        组成　关键点描述子　常用来形成原始数据的表示，而且不失代表性和描述性，
        从而加快了后续的识别，追踪等对数据的处理了速度，
        故而，关键点技术成为在2D和3D 信息处理中非常关键的技术。

        常见的三维点云关键点提取算法有一下几种：ISS3D、Harris3D、NARF、SIFT3D
        这些算法在PCL库中都有实现，其中NARF算法是博主见过用的比较多的。

        PCL—低层次视觉—关键点检测
         关键点检测往往需要和特征提取联合在一起，关键点检测的一个重要性质就是旋转不变性，
        也就是说，物体旋转后还能够检测出对应的关键点。不过说实话我觉的这个要求对机器人视觉来说是比较鸡肋的。
        因为机器人采集到的三维点云并不是一个完整的物体，没哪个相机有透视功能。
        机器人采集到的点云也只是一层薄薄的蒙皮。所谓的特征点又往往在变化剧烈的曲面区域，
        那么从不同的视角来看，变化剧烈的曲面区域很难提取到同样的关键点。
        想象一下一个人的面部，正面的时候鼻尖可以作为关键点，但是侧面的时候呢？
        会有一部分面部在阴影中，模型和之前可能就完全不一样了。

         也就是说现在这些关键点检测算法针对场景中较远的物体，
        也就是物体旋转带来的影响被距离减弱的情况下，是好用的。
        一旦距离近了，旋转往往造成捕获的仅有模型的侧面，关键点检测算法就有可能失效。

        ---------------------------------------------------------------------------
###  ISS算法的全程是Intrinsic Shape Signatures，第一个词叫做内部，这个词挺有讲究。
        说内部，那必然要有个范围，具体是什么东西的范围还暂定。
        如果说要描述一个点周围的局部特征，而且这个物体在全局坐标下还可能移动，
        那么有一个好方法就是在这个点周围建立一个局部坐标。
        只要保证这个局部坐标系也随着物体旋转就好。


#### 方法1.基于协方差矩阵
          协方差矩阵的思想其实很简单，实际上它是一种耦合，把两个步骤耦合在了一起
            1.把pi和周围点pj的坐标相减：本质上这生成了许多从pi->pj的向量，
            理想情况下pi的法线应该是垂直于这些向量的
            2.利用奇异值分解求这些向量的0空间，拟合出一个尽可能垂直的向量，作为法线的估计

          协方差矩阵本质是啥？就是奇异值分解中的一个步骤。。。。
           奇异值分解是需要矩阵乘以自身的转置从而得到对称矩阵的。
          当然，用协方差计算的好处是可以给不同距离的点附上不同的权重。

#### 方法2.基于齐次坐标
            1.把点的坐标转为齐次坐标
            2.对其次坐标进行奇异值分解
            3.最小奇异值对应的向量就是拟合平面的方程
            4.方程的系数就是法线的方向。
          显然，这种方法更加简单粗暴，省去了权重的概念，但是换来了运算速度，不需要反复做减法。
           其实本来也不需要反复做减法，做一个点之间向量的检索表就好。。。

          但是我要声明PCL的实现是利用反复减法的。

        都会有三个相互垂直的向量，一个是法线方向，另外两个方向与之构成了在某点的局部坐标系。
        在此局部坐标系内进行建模，就可以达到点云特征旋转不变的目的了。

        ISS特征点检测的思想也甚是简单：
          1.利用方法1建立模型
          2.其利用特征值之间关系来形容该点的特征程度。
          显然这种情况下的特征值是有几何意义的，特征值的大小实际上是椭球轴的长度。
            椭球的的形态则是对邻近点分布状态的抽象总结。
            试想，如果临近点沿某个方向分布致密则该方向会作为椭球的第一主方向，
            稀疏的方向则是第二主方向，法线方向当然是极度稀疏（只有一层），那么则作为第三主方向。

          如果某个点恰好处于角点，则第一主特征值，第二主特征值，第三主特征值大小相差不会太大。
          如果点云沿着某方向致密，而垂直方向系数则有可能是边界。
          总而言之，这种局部坐标系建模分析的方法是基于特征值分析的特征点提取。
        -----------------------------------------------------------------------
###  Trajkovic关键点检测算法

          角点的一个重要特征就是法线方向和周围的点存在不同，
        而本算法的思想就是和相邻点的法线方向进行对比，判定法线方向差异的阈值，
        最终决定某点是否是角点。并且需要注意的是，本方法所针对的点云应该只是有序点云。
          本方法的优点是快，缺点是对噪声敏感。
          手头没有有序点云，不做测试了。

        除去NARF这种和特征检测联系比较紧密的方法外，一般来说特征检测都会对曲率变化比较剧烈的点更敏感。
        Harris算法是图像检测识别算法中非常重要的一个算法，其对物体姿态变化鲁棒性好，
        对旋转不敏感，可以很好的检测出物体的角点。
        甚至对于标定算法而言，HARRIS角点检测是使之能成功进行的基础。
          HARRIS算法的思想还是很有意思的。很聪明也很trick.
        --------------------------------------------------------------------
###  Harris 算法　
          其思想及数学推导大致如下：
          1.在图像中取一个窗 w (矩形窗，高斯窗，XX窗，各种窗，
            某师姐要改标定算法不就可以从选Harris的窗开始做起么）

          2.获得在该窗下的灰度 I
          3.移动该窗，则灰度会发生变化，平坦区域灰度变化不大，
            边缘区域沿边缘方向灰度变化剧烈，角点处各个方向灰度变化均剧烈

          4.依据3中条件选出角点

        1.两个特征值都很大==========>角点（两个响应方向）
        2.一个特征值很大，一个很小=====>边缘（只有一个响应方向）
        3.两个特征值都小============>平原地区（响应都很微弱）
        ---------------------------------------------------------------------------
        3DHarris　 方块体内点数量变化确定角点
          在2DHarris里，我们使用了 图像梯度构成的 协方差矩阵。 
            图像梯度。。。嗯。。。。每个像素点都有一个梯度，
            在一阶信息量的情况下描述了两个相邻像素的关系。显然这个思想可以轻易的移植到点云上来。

        想象一下，如果在 点云中存在一点p
          1、在p上建立一个局部坐标系：z方向是法线方向，x,y方向和z垂直。
          2、在p上建立一个小正方体，不要太大，大概像材料力学分析应力那种就行
          3、假设点云的密度是相同的，点云是一层蒙皮，不是实心的。
          a、如果小正方体沿z方向移动，那小正方体里的点云数量应该不变
          b、如果小正方体位于边缘上，则沿边缘移动，点云数量几乎不变，沿垂直边缘方向移动，点云数量改
          c、如果小正方体位于角点上，则有两个方向都会大幅改变点云数量

          如果由法向量x,y,z构成协方差矩阵，那么它应该是一个对称矩阵。
        而且特征向量有一个方向是法线方向，另外两个方向和法线垂直。
          那么直接用协方差矩阵替换掉图像里的M矩阵，就得到了点云的Harris算法。
          其中，半径r可以用来控制角点的规模
          r小，则对应的角点越尖锐（对噪声更敏感）
          r大，则可能在平缓的区域也检测出角点

        ----------------------------------------------------------------------
### NARF　
        1. 边缘提取
        对点云而言，场景的边缘代表前景物体和背景物体的分界线。
        所以，点云的边缘又分为三种：

        前景边缘，背景边缘，阴影边缘。

        三维点云的边缘有个很重要的特征，
        就是点a 和点b 如果在 rangImage 上是相邻的，然而在三维距离上却很远，那么多半这里就有边缘。
        由于三维点云的规模和稀疏性，“很远”这个概念很难描述清楚。
        到底多远算远？这里引入一个横向的比较是合适的。这种比较方法可以自适应点云的稀疏性。
        所谓的横向比较就是和 某点周围的点相比较。 这个周围有多大？不管多大，
        反正就是在某点pi的rangeImage 上取一个方窗。
        假设像素边长为s. 那么一共就取了s^2个点。
        接下来分三种情况来讨论所谓的边缘：


         1.这个点在某个平面上，边长为 s 的方窗没有涉及到边缘
         2.这个点恰好在某条边缘上，边长 s 的方窗一半在边缘左边，一半在右边
         3.这个点恰好处于某个角点上，边长 s 的方窗可能只有 1/4 与 pi 处于同一个平面
        如果将 pi 与不同点距离进行排序，得到一系列的距离，d0 表示与 pi 距离最近的点，显然是 pi 自己。

         ds^2 是与pi 最远的点，这就有可能是跨越边缘的点了。 
        选择一个dm，作为与m同平面，但距离最远的点。
        也就是说，如果d0~ds^2是一个连续递增的数列，那么dm可以取平均值。

        如果这个数列存在某个阶跃跳动（可能会形成类似阶跃信号）
        那么则发生阶跃的地方应该是有边缘存在，不妨取阶跃点为dm(距离较小的按个阶跃点）
        原文并未如此表述此段落，原文取s=5, m=9 作为m点的一个合理估计。
        -------------------------------------------------------------------------------
####  关键点提取

            在提取关键点时，边缘应该作为一个重要的参考依据。
        但一定不是唯一的依据。对于某个物体来说关键点应该是表达了某些特征的点，而不仅仅是边缘点。
        所以在设计关键点提取算法时，需要考虑到以下一些因素：
        边缘和曲面结构都要考虑进去；
        关键点要能重复；
        关键点最好落在比较稳定的区域，方便提取法线。

        对于点云构成的曲面而言，某处的曲率无疑是一个非常重要的结构描述因素。
        某点的曲率越大，则该点处曲面变化越剧烈。
        在2D rangeImage 上，去 pi 点及其周边与之距离小于2deta的点，
        进行PCA主成分分析。可以得到一个 主方向v，以及曲率值 lamda. 
        注意， v 必然是一个三维向量。

        那么对于边缘点，可以取其 权重 w 为1 ， v 为边缘方向。
          对于其他点，取权重 w 为 1-(1-lamda)^3 ， 方向为 v 在平面 p上的投影。 
          平面 p 垂直于 pi 与原点连线。
          到此位置，每个点都有了两个量，一个权重，一个方向。
          将权重与方向带入下列式子 I 就是某点 为特征点的可能性。



        ###############################################################
## E 点云　特征和特征描述　/Features
        ###################################################################


    如果要对一个三维点云进行描述，光有点云的位置是不够的，常常需要计算一些额外的参数，
    比如法线方向、曲率、文理特征、颜色、领域中心距、协方差矩阵、熵等等。
    如同图像的特征（sifi surf orb）一样，我们需要使用类似的方式来描述三维点云的特征。

    常用的特征描述算法有：
    法线和曲率计算 normal_3d_feature 、
    特征值分析、
    PFH  点特征直方图描述子 nk2、
    FPFH 快速点特征直方图描述子 FPFH是PFH的简化形式 nk、
    3D Shape Context、 文理特征
    Spin Image
    VFH视点特征直方图(Viewpoint Feature Histogram)
    NARF关键点  pcl::NarfKeypoint narf特征 pcl::NarfDescriptor
    RoPs特征(Rotational Projection Statistics) 
    (GASD）全局一致的空间分布描述子特征 Globally Aligned Spatial Distribution (GASD) descriptors


### 【1 】估计表面法线的解决方案就变成了分析一个协方差矩阵的特征矢量和特征值
        （或者PCA—主成分分析），这个协方差矩阵从查询点的近邻元素中创建。
        更具体地说，对于每一个点Pi,对应的协方差矩阵。

        参考理解
        http://geometryhub.net/notes/pointcloudnormal

        PCA降维到 二维平面去法线
        http://blog.codinglabs.org/articles/pca-tutorial.html


        点云法线有什么用
        点云渲染：法线信息可以用于光照渲染，有些地方也称着色（立体感）。
        如下图所示，左边的点云没有法线信息，右边的点云有法线信息。
        比如Phone光照模型里，
        漫反射光照符合Lambert余弦定律：漫反射光强与N * L成正比，N为法线方向，L为点到光源的向量。
        所以，在模型边缘处，N与L近似垂直，着色会比较暗。

        点云的几何属性：法线可用于计算几何相关的信息，
        广泛应用于点云注册（配准），点云重建，特征点检测等。
        另外法线信息还可以用于区分薄板正反面。

        前面说的是二维降到一维时的情况，假如我们有一堆散乱的三维点云,则可以这样计算法线：
        1）对每一个点，取临近点，比如取最临近的50个点，当然会用到K-D树
        2）对临近点做PCA降维，把它降到二维平面上,
        可以想象得到这个平面一定是它的切平面(在切平面上才可以尽可能分散）
        3）切平面的法线就是该点的法线了，而这样的法线有两个，
        取哪个还需要考虑临近点的凸包方向
        #include <pcl/features/normal_3d.h>//法线特征
        --------------------------------------------------------------
        // 创建法线估计类====================================
        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
        ne.setInputCloud (cloud_ptr);
        // 添加搜索算法 kdtree search  最近的几个点 估计平面 协方差矩阵PCA分解 求解法线
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
        ne.setSearchMethod (tree);
        // 输出点云 带有法线描述
        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals_ptr (new pcl::PointCloud<pcl::Normal>);
        pcl::PointCloud<pcl::Normal>& cloud_normals = *cloud_normals_ptr;
        ne.setRadiusSearch (0.03);//半径内搜索临近点
        //ne.setKSearch(8);       //其二 指定临近点数量
        // 计算表面法线特征
        ne.compute (cloud_normals);

    -------------------------------------------------------------
### 【2】使用积分图计算一个有序点云的法线，注意该方法只适用于有序点云
    表面法线是几何体表面的重要属性，在很多领域都有大量应用，
    例如：在进行光照渲染时产生符合可视习惯的效果时需要表面法线信息才能正常进行，
    对于一个已知的几何体表面，根据垂直于点表面的矢量，因此推断表面某一点的法线方向通常比较简单。
    头文件
    #include <pcl/features/integral_image_normal.h>
    ---------------------------------------------------------
      // 估计的法线 normals
      pcl::PointCloud<pcl::Normal>::Ptr normals_ptr (new pcl::PointCloud<pcl::Normal>);
      pcl::PointCloud<pcl::Normal>& normals = *normals_ptr;
            // 积分图像方法 估计点云表面法线 
      pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal> ne;

    /*
    预测方法
    enum NormalEstimationMethod
    {
      COVARIANCE_MATRIX, 从最近邻的协方差矩阵创建了9个积分图去计算一个点的法线
      AVERAGE_3D_GRADIENT, 创建了6个积分图去计算3D梯度里面竖直和水平方向的光滑部分，同时利用两个梯度的卷积来计算法线。
      AVERAGE_DEPTH_CHANGE 造了一个单一的积分图，从平均深度的变化中来计算法线。
    };
    */ 
      ne.setNormalEstimationMethod (ne.AVERAGE_3D_GRADIENT);
      ne.setMaxDepthChangeFactor(0.02f);
      ne.setNormalSmoothingSize(10.0f);
      ne.setInputCloud(cloud_ptr);
      ne.compute(normals);

    --------------------------------------------

###  【3】 点特征直方图(PFH)描述子
    计算法线---计算临近点对角度差值-----直方图--
    点特征直方图(PFH)描述子
    点特征直方图(Point Feature Histograms)
    正如点特征表示法所示，表面法线和曲率估计是某个点周围的几何特征基本表示法。
    虽然计算非常快速容易，但是无法获得太多信息，因为它们只使用很少的
    几个参数值来近似表示一个点的k邻域的几何特征。然而大部分场景中包含许多特征点，
    这些特征点有相同的或者非常相近的特征值，因此采用点特征表示法，
    其直接结果就减少了全局的特征信息。

    http://www.pclcn.org/study/shownews.php?lang=cn&id=101

    是基于点与其k邻域之间的关系以及它们的估计法线，
    简言之，它考虑估计法线方向之间所有的相互作用，
    试图捕获最好的样本表面变化情况，以描述样本的几何特征。

    每一点对，原有12个参数，6个坐标值，6个坐标姿态（基于法线）
    PHF计算没一点对的 相对坐标角度差值三个值和 坐标点之间的欧氏距离 d
    从12个参数减少到4个参数
    默认PFH的实现使用5个区间分类（例如：四个特征值中的每个都使用5个区间来统计），
    其中不包括距离（在上文中已经解释过了——但是如果有需要的话，
    也可以通过用户调用computePairFeatures方法来获得距离值），
    这样就组成了一个125浮点数元素的特征向量（15），
    其保存在一个pcl::PFHSignature125的点类型中。

    头文件
    #include <pcl/features/pfh.h>
    #include <pcl/features/normal_3d.h>//法线特征
    ---------------------------------------------------
    // =====计算法线========创建法线估计类====================================
      pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
      ne.setInputCloud (cloud_ptr);

    // 添加搜索算法 kdtree search  最近的几个点 估计平面 协方差矩阵PCA分解 求解法线
      pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
      ne.setSearchMethod (tree);//设置近邻搜索算法 
      // 输出点云 带有法线描述
      pcl::PointCloud<pcl::Normal>::Ptr cloud_normals_ptr (new pcl::PointCloud<pcl::Normal>);
      pcl::PointCloud<pcl::Normal>& cloud_normals = *cloud_normals_ptr;
      // Use all neighbors in a sphere of radius 3cm
      ne.setRadiusSearch (0.03);//半价内搜索临近点 3cm
      // 计算表面法线特征
      ne.compute (cloud_normals);

    //=======创建PFH估计对象pfh，并将输入点云数据集cloud和法线normals传递给它=================
      pcl::PFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::PFHSignature125> pfh;// phf特征估计其器
      pfh.setInputCloud (cloud_ptr);
      pfh.setInputNormals (cloud_normals_ptr);
     //如果点云是类型为PointNormal,则执行pfh.setInputNormals (cloud);
     //创建一个空的kd树表示法，并把它传递给PFH估计对象。
     //基于已给的输入数据集，建立kdtree
      pcl::search::KdTree<pcl::PointXYZ>::Ptr tree2 (new pcl::search::KdTree<pcl::PointXYZ> ());
      //pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr tree2 (new pcl::KdTreeFLANN<pcl::PointXYZ> ()); //-- older call for PCL 1.5-
      pfh.setSearchMethod (tree2);//设置近邻搜索算法 
      //输出数据集
      pcl::PointCloud<pcl::PFHSignature125>::Ptr pfh_fe_ptr (new pcl::PointCloud<pcl::PFHSignature125> ());//phf特征
     //使用半径在5厘米范围内的所有邻元素。
      //注意：此处使用的半径必须要大于估计表面法线时使用的半径!!!
      pfh.setRadiusSearch (0.05);
      //计算pfh特征值
      pfh.compute (*pfh_fe_ptr);
    ----------------------------------------------------------------------
### 【4】phf点特征直方图 计算复杂度还是太高
    计算法线---计算临近点对角度差值-----直方图--
    因此存在一个O(nk^2) 的计算复杂性。
    k个点之间相互的点对 k×k条连接线

    每一点对，原有12个参数，6个坐标值，6个坐标姿态（基于法线）
    PHF计算没一点对的 相对坐标角度差值三个值和 坐标点之间的欧氏距离 d
    从12个参数减少到4个参数

    快速点特征直方图FPFH（Fast Point Feature Histograms）把算法的计算复杂度降低到了O(nk) ，
    但是任然保留了PFH大部分的识别特性。
    查询点和周围k个点的连线 的4参数特征
    也就是1×k=k个线

    默认的FPFH实现使用11个统计子区间（例如：四个特征值中的每个都将它的参数区间分割为11个），
    特征直方图被分别计算然后合并得出了浮点值的一个33元素的特征向量，
    这些保存在一个pcl::FPFHSignature33点类型中。

    头文件
    #include <pcl/features/fpfh.h>
    #include <pcl/features/normal_3d.h>//法线特征
    ----------------------------------------------------
    // =====计算法线========创建法线估计类====================================
      pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
      ne.setInputCloud (cloud_ptr);
    /*
     法线估计类NormalEstimation的实际计算调用程序内部执行以下操作：
    对点云P中的每个点p
      1.得到p点的最近邻元素
      2.计算p点的表面法线n
      3.检查n的方向是否一致指向视点，如果不是则翻转

     在PCL内估计一点集对应的协方差矩阵，可以使用以下函数调用实现：
    //定义每个表面小块的3x3协方差矩阵的存储对象
    Eigen::Matrix3fcovariance_matrix;
    //定义一个表面小块的质心坐标16-字节对齐存储对象
    Eigen::Vector4fxyz_centroid;
    //估计质心坐标
    compute3DCentroid(cloud,xyz_centroid);
    //计算3x3协方差矩阵
    computeCovarianceMatrix(cloud,xyz_centroid,covariance_matrix);
    */
    // 添加搜索算法 kdtree search  最近的几个点 估计平面 协方差矩阵PCA分解 求解法线
      pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
      ne.setSearchMethod (tree);//设置近邻搜索算法 
        // 输出点云 带有法线描述
      pcl::PointCloud<pcl::Normal>::Ptr cloud_normals_ptr (new pcl::PointCloud<pcl::Normal>);
      pcl::PointCloud<pcl::Normal>& cloud_normals = *cloud_normals_ptr;
        // Use all neighbors in a sphere of radius 3cm
      ne.setRadiusSearch (0.03);//半价内搜索临近点 3cm
        // 计算表面法线特征
      ne.compute (cloud_normals);

    //=======创建FPFH估计对象fpfh, 并将输入点云数据集cloud和法线normals传递给它=================
        //pcl::PFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::PFHSignature125> pfh;// phf特征估计其器
      pcl::FPFHEstimation<pcl::PointXYZ,pcl::Normal,pcl::FPFHSignature33> fpfh;
        // pcl::FPFHEstimationOMP<pcl::PointXYZ,pcl::Normal,pcl::FPFHSignature33> fpfh;//多核加速
      fpfh.setInputCloud (cloud_ptr);
      fpfh.setInputNormals (cloud_normals_ptr);
       //如果点云是类型为PointNormal,则执行pfh.setInputNormals (cloud);
       //创建一个空的kd树表示法，并把它传递给PFH估计对象。
       //基于已给的输入数据集，建立kdtree
      pcl::search::KdTree<pcl::PointXYZ>::Ptr tree2 (new pcl::search::KdTree<pcl::PointXYZ> ());
        //pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr tree2 (new pcl::KdTreeFLANN<pcl::PointXYZ> ()); //-- older call for PCL 1.5-
      fpfh.setSearchMethod (tree2);//设置近邻搜索算法 
        //输出数据集
        //pcl::PointCloud<pcl::PFHSignature125>::Ptr pfh_fe_ptr (new pcl::PointCloud<pcl::PFHSignature125> ());//phf特征
      pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_fe_ptr (new pcl::PointCloud<pcl::FPFHSignature33>());//fphf特征
        //使用半径在5厘米范围内的所有邻元素。
        //注意：此处使用的半径必须要大于估计表面法线时使用的半径!!!
      fpfh.setRadiusSearch (0.05);
        //计算pfh特征值
      fpfh.compute (*fpfh_fe_ptr);
    ---------------------------------------------------------------


### 【5】视点特征直方图VFH(Viewpoint Feature Histogram)描述子，
    它是一种新的特征表示形式，应用在点云聚类识别和六自由度位姿估计问题。

    视点特征直方图（或VFH）是源于FPFH描述子.
    由于它的获取速度和识别力，我们决定利用FPFH强大的识别力，
    但是为了使构造的特征保持缩放不变性的性质同时，
    还要区分不同的位姿，计算时需要考虑加入视点变量。

    我们做了以下两种计算来构造特征，以应用于目标识别问题和位姿估计：
    1.扩展FPFH，使其利用整个点云对象来进行计算估计（如2图所示），
    在计算FPFH时以物体中心点与物体表面其他所有点之间的点对作为计算单元。

    2.添加视点方向与每个点估计法线之间额外的统计信息，为了达到这个目的，
    我们的关键想法是在FPFH计算中将视点方向变量直接融入到相对法线角计算当中。

    因此新组合的特征被称为视点特征直方图（VFH）。
    下图表体现的就是新特征的想法，包含了以下两部分：

    1.一个视点方向相关的分量
    2.一个包含扩展FPFH的描述表面形状的分量

    对扩展的FPFH分量来说，默认的VFH的实现使用45个子区间进行统计，
    而对于视点分量要使用128个子区间进行统计，这样VFH就由一共308个浮点数组成阵列。
    在PCL中利用pcl::VFHSignature308的点类型来存储表示。P
    FH/FPFH描述子和VFH之间的主要区别是：

    对于一个已知的点云数据集，只一个单一的VFH描述子，
    而合成的PFH/FPFH特征的数目和点云中的点数目相同。

    头文件
    #include <pcl/features/vfh.h>
    #include <pcl/features/normal_3d.h>//法线特征

    ------------------------------------------------------
    // =====计算法线========创建法线估计类====================================
      pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
      ne.setInputCloud (cloud_ptr);

    // 添加搜索算法 kdtree search  最近的几个点 估计平面 协方差矩阵PCA分解 求解法线
      pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
      ne.setSearchMethod (tree);//设置近邻搜索算法 
      // 输出点云 带有法线描述
      pcl::PointCloud<pcl::Normal>::Ptr cloud_normals_ptr (new pcl::PointCloud<pcl::Normal>);
      pcl::PointCloud<pcl::Normal>& cloud_normals = *cloud_normals_ptr;
      // Use all neighbors in a sphere of radius 3cm
      ne.setRadiusSearch (0.03);//半价内搜索临近点 3cm
      // 计算表面法线特征
      ne.compute (cloud_normals);


    //=======创建VFH估计对象vfh，并把输入数据集cloud和法线normal传递给它================
      pcl::VFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::VFHSignature308> vfh;
      //pcl::PFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::PFHSignature125> pfh;// phf特征估计其器
      //pcl::FPFHEstimation<pcl::PointXYZ,pcl::Normal,pcl::FPFHSignature33> fpfh;// fphf特征估计其器
      // pcl::FPFHEstimationOMP<pcl::PointXYZ,pcl::Normal,pcl::FPFHSignature33> fpfh;//多核加速
      vfh.setInputCloud (cloud_ptr);
      vfh.setInputNormals (cloud_normals_ptr);
      //如果点云是PointNormal类型，则执行vfh.setInputNormals (cloud);
      //创建一个空的kd树对象，并把它传递给FPFH估计对象。
      //基于已知的输入数据集，建立kdtree
      pcl::search::KdTree<pcl::PointXYZ>::Ptr tree2 (new pcl::search::KdTree<pcl::PointXYZ> ());
      //pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr tree2 (new pcl::KdTreeFLANN<pcl::PointXYZ> ()); //-- older call for PCL 1.5-
      vfh.setSearchMethod (tree2);//设置近邻搜索算法 
      //输出数据集
      //pcl::PointCloud<pcl::PFHSignature125>::Ptr pfh_fe_ptr (new pcl::PointCloud<pcl::PFHSignature125> ());//phf特征
      //pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_fe_ptr (new pcl::PointCloud<pcl::FPFHSignature33>());//fphf特征
      pcl::PointCloud<pcl::VFHSignature308>::Ptr vfh_fe_ptr (new pcl::PointCloud<pcl::VFHSignature308> ());//vhf特征
      //使用半径在5厘米范围内的所有邻元素。
      //注意：此处使用的半径必须要大于估计表面法线时使用的半径!!!
      //fpfh.setRadiusSearch (0.05);
      //计算pfh特征值
      vfh.compute (*vfh_fe_ptr);

    ----------------------------------------------------------

### 【6】NARF　
    从深度图像(RangeImage)中提取NARF关键点  pcl::NarfKeypoint   
     然后计算narf特征 pcl::NarfDescriptor
    边缘提取
    直接把三维的点云投射成二维的图像不就好了。
    这种投射方法叫做range_image.

    #include <pcl/range_image/range_image.h>// RangeImage 深度图像
    #include <pcl/keypoints/narf_keypoint.h>// narf关键点检测
    #include <pcl/features/narf_descriptor.h>// narf特征

    ------------------------------------------------------

    // ======从点云数据，创建深度图像=====================
      // 直接把三维的点云投射成二维的图像
      float noise_level = 0.0;
    //noise level表示的是容差率，因为1°X1°的空间内很可能不止一个点，
    //noise level = 0则表示去最近点的距离作为像素值，如果=0.05则表示在最近点及其后5cm范围内求个平均距离
    //minRange表示深度最小值，如果=0则表示取1°X1°的空间内最远点，近的都忽略
      float min_range = 0.0f;
    //bordersieze表示图像周边点 
      int border_size = 1;
      boost::shared_ptr<pcl::RangeImage> range_image_ptr (new pcl::RangeImage);//创建RangeImage对象（智能指针）
      pcl::RangeImage& range_image = *range_image_ptr; //RangeImage的引用  
    //半圆扫一圈就是整个图像了
     range_image.createFromPointCloud (point_cloud, angular_resolution, pcl::deg2rad (360.0f), pcl::deg2rad (180.0f),
                                       scene_sensor_pose, coordinate_frame, noise_level, min_range, border_size);
      range_image.integrateFarRanges (far_ranges);//整合远距离点云
      if (setUnseenToMaxRange)
        range_image.setUnseenToMaxRange ();

     // ====================提取NARF关键点=======================
      pcl::RangeImageBorderExtractor range_image_border_extractor;//创建深度图像的边界提取器，用于提取NARF关键点
      pcl::NarfKeypoint narf_keypoint_detector (&range_image_border_extractor);//创建NARF对象
      narf_keypoint_detector.setRangeImage (&range_image);//设置点云对应的深度图
      narf_keypoint_detector.getParameters ().support_size = support_size;// 感兴趣点的尺寸（球面的直径）
      //narf_keypoint_detector.getParameters ().add_points_on_straight_edges = true;
      //narf_keypoint_detector.getParameters ().distance_for_additional_points = 0.5;

      pcl::PointCloud<int> keypoint_indices;//用于存储关键点的索引 PointCloud<int>
      narf_keypoint_detector.compute (keypoint_indices);//计算NARF关键

    //========================提取 NARF 特征 ====================
      // ------------------------------------------------------
      std::vector<int> keypoint_indices2;//用于存储关键点的索引 vector<int>  
      keypoint_indices2.resize (keypoint_indices.points.size ());
      for (unsigned int i=0; i<keypoint_indices.size (); ++i) // This step is necessary to get the right vector type
        keypoint_indices2[i] = keypoint_indices.points[i];//narf关键点 索引
      pcl::NarfDescriptor narf_descriptor (&range_image, &keypoint_indices2);//narf特征描述子
      narf_descriptor.getParameters().support_size = support_size;
      narf_descriptor.getParameters().rotation_invariant = rotation_invariant;
      pcl::PointCloud<pcl::Narf36> narf_descriptors;
      narf_descriptor.compute (narf_descriptors);

    ---------------------------------------------------------------
### 【7】RoPs特征(Rotational Projection Statistics) 描述子
    0.在关键点出建立局部坐标系。
    1.在一个给定的角度在当前坐标系下对关键点领域(局部表面) 进行旋转
    2.把 局部表面 投影到 xy，yz，xz三个2维平面上
    3.在每个投影平面上划分不同的盒子容器，把点分到不同的盒子里
    4.根据落入每个盒子的数量，来计算每个投影面上的一系列数据分布
    （熵值，低阶中心矩
    5.M11,M12,M21,M22，E。E是信息熵。4*2+1=9）进行描述
    计算值将会组成子特征。
    盒子数量 × 旋转次数×9 得到特征维度

    我们把上面这些步骤进行多次迭代。不同坐标轴的子特征将组成RoPS描述器
    我们首先要找到目标模型:
    points 包含点云
    indices 点的下标
    triangles包含了多边形
    -------------------------------------

    #include <pcl/features/rops_estimation.h>
    ------------------------------------------------
      float support_radius = 0.0285f;//局部表面裁剪支持的半径 (搜索半价)，
      unsigned int number_of_partition_bins = 5;//以及用于组成分布矩阵的容器的数量
      unsigned int number_of_rotations = 3;//和旋转的次数。最后的参数将影响描述器的长度。

    //搜索方法
      pcl::search::KdTree<pcl::PointXYZ>::Ptr search_method (new pcl::search::KdTree<pcl::PointXYZ>);
      search_method->setInputCloud (cloud_ptr);
    // rops 特征算法 对象 盒子数量 × 旋转次数×9 得到特征维度  3*5*9 =135
      pcl::ROPSEstimation <pcl::PointXYZ, pcl::Histogram <135> > feature_estimator;
      feature_estimator.setSearchMethod (search_method);//搜索算法
      feature_estimator.setSearchSurface (cloud_ptr);//搜索平面
      feature_estimator.setInputCloud (cloud_ptr);//输入点云
      feature_estimator.setIndices (indices);//关键点索引
      feature_estimator.setTriangles (triangles);//领域形状
      feature_estimator.setRadiusSearch (support_radius);//搜索半径
      feature_estimator.setNumberOfPartitionBins (number_of_partition_bins);//盒子数量
      feature_estimator.setNumberOfRotations (number_of_rotations);//旋转次数
      feature_estimator.setSupportRadius (support_radius);// 局部表面裁剪支持的半径 

      pcl::PointCloud<pcl::Histogram <135> >::Ptr histograms (new pcl::PointCloud <pcl::Histogram <135> > ());
      feature_estimator.compute (*histograms);


    -------------------------------------
### 【8】momentofinertiaestimation类获取基于惯性偏心矩 描述子。
    这个类还允许提取云的轴对齐和定向包围盒。
    但请记住，不可能的最小提取OBB包围盒。

    首先计算点云的协方差矩阵，提取其特征值和向量。
    你可以认为所得的特征向量是标准化的，
    总是形成右手坐标系（主要特征向量表示x轴，而较小的向量表示z轴）。
    在下一个步骤中，迭代过程发生。每次迭代时旋转主特征向量。
    旋转顺序总是相同的，并且在其他特征向量周围执行，
    这提供了点云旋转的不变性。
    此后，我们将把这个旋转主向量作为当前轴。

    然后在当前轴上计算转动惯量 和 将点云投影到以旋转向量为法线的平面上
    计算偏心距

    ------------------------
    #include <pcl/features/moment_of_inertia_estimation.h>
    -------------------------------------------------------------------------------------
     pcl::MomentOfInertiaEstimation <pcl::PointXYZ> feature_extractor;// 惯量 偏心距 特征提取
      feature_extractor.setInputCloud (cloud_ptr);//输入点云
      feature_extractor.compute ();//计算

      std::vector <float> moment_of_inertia;// 惯量
      std::vector <float> eccentricity;// 偏心距
      pcl::PointXYZ min_point_AABB;
      pcl::PointXYZ max_point_AABB;
      pcl::PointXYZ min_point_OBB;
      pcl::PointXYZ max_point_OBB;
      pcl::PointXYZ position_OBB;
      Eigen::Matrix3f rotational_matrix_OBB;
      float major_value, middle_value, minor_value;
      Eigen::Vector3f major_vector, middle_vector, minor_vector;
      Eigen::Vector3f mass_center;

      feature_extractor.getMomentOfInertia (moment_of_inertia);// 惯量
      feature_extractor.getEccentricity (eccentricity);// 偏心距
    // 以八个点的坐标 给出 包围盒
      feature_extractor.getAABB (min_point_AABB, max_point_AABB);// AABB 包围盒坐标 八个点的坐标
    // 以中心点 姿态 坐标轴范围 给出   包围盒
      feature_extractor.getOBB (min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);// 位置和方向姿态
      feature_extractor.getEigenValues (major_value, middle_value, minor_value);//特征值
      feature_extractor.getEigenVectors (major_vector, middle_vector, minor_vector);//特征向量
      feature_extractor.getMassCenter (mass_center);//点云中心点

    --------------------------------------------------------------------
### 【9】全局一致的空间分布描述子特征
    Globally Aligned Spatial Distribution (GASD) descriptors
    可用于物体识别和姿态估计。
    是对可以描述整个点云的参考帧的估计，
    这是用来对准它的正则坐标系统
    之后，根据其三维点在空间上的分布，计算出点云的描述符。
    种描述符也可以扩展到整个点云的颜色分布。
    匹配点云(icp)的全局对齐变换用于计算物体姿态。

    使用主成分分析PCA来估计参考帧
    三维点云P
    计算其中心点位置P_
    计算协方差矩阵 1/n × sum((pi - P_)*(pi - P_)转置)
    奇异值分解得到 其 特征值eigen values  和特征向量eigen vectors
    基于参考帧计算一个变换 [R t]
    ------------------------------------
    #include <pcl/features/gasd.h>
    --------------------------------------------------
    // 创建 GASD 全局一致的空间分布描述子特征 传递 点云
     // pcl::GASDColorEstimation<pcl::PointXYZRGBA, pcl::GASDSignature984> gasd;//包含颜色
      pcl::GASDColorEstimation<pcl::PointXYZ, pcl::GASDSignature984> gasd;
      gasd.setInputCloud (cloud_ptr);
      // 输出描述子
      pcl::PointCloud<pcl::GASDSignature984> descriptor;
      // 计算描述子
      gasd.compute (descriptor);
      // 得到匹配 变换
      Eigen::Matrix4f trans = gasd.getTransform();

    ---------------------------------------------------------------




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

        ############################################
## J 变化检测 Octree 八叉树算法
        ########################################################

        当无序点云在连续变化中，八叉树算法常常被用于检测变化，
        这种算法需要和关键点提取技术结合起来，八叉树算法也算是经典中的经典了。
         octree是一种用于管理稀疏3D数据的树状数据结构，
        我们学习如何利用octree实现用于多个无序点云之间的空间变化检测，
        这些点云可能在尺寸、分辨率、密度和点顺序等方面有所差异。
        通过递归地比较octree的树结构，
        可以鉴定出由octree产生的体素组成之间的区别所代表的空间变化，
        此外，我们解释了如何使用PCL的octree“双缓冲”技术，
        以便能实时地探测多个点云之间的空间组成差异。


        八叉树空间分割进行点分布区域的压缩
        通过对连续帧之间的数据相关分析，检测出重复的点云，并将其去除掉再进行传输.

        点云由庞大的数据集组成，这些数据集通过距离、颜色、法线等附加信息来描述空间三维点。
        此外，点云能以非常高的速率被创建出来，因此需要占用相当大的存储资源，
        一旦点云需要存储或者通过速率受限制的通信信道进行传输，提供针对这种数据的压缩方法就变得十分有用。
        PCL库提供了点云压缩功能，它允许编码压缩所有类型的点云，包括“无序”点云，
        它具有无参考点和变化的点尺寸、分辨率、分布密度和点顺序等结构特征。
        而且，底层的octree数据结构允许从几个输入源高效地合并点云数据
        .

        Octree八插树是一种用于描述三维空间的树状数据结构。
        八叉树的每个节点表示一个正方体的体积元素，每个节点有八个子节点，
        将八个子节点所表示的体积元素加在一起就等于父节点的体积。
        Octree模型：又称为八叉树模型，若不为空树的话，
        树中任一节点的子节点恰好只会有八个，或零个，也就是子节点不会有0与8以外的数目。

        Log8(房间内的所有物品数)的时间内就可找到金币。
        因此，八叉树就是用在3D空间中的场景管理，可以很快地知道物体在3D场景中的位置，
        或侦测与其它物体是否有碰撞以及是否在可视范围内。

        基于Octree的空间划分及搜索操作

        octree是一种用于管理稀疏3D数据的树状数据结构，每个内部节点都正好有八个子节点，
        本小节中我们学习如何用octree在点云数据中进行空间划分及近邻搜索，特别地，
        解释了如何完成
        “体素内近邻搜索(Neighbors within Voxel Search)”、
        “K近邻搜索(K Nearest Neighbor Search)”和
        “半径内近邻搜索(Neighbors within Radius Search)”。



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


