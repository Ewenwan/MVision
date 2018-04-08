# 点云曲面

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


## 滑动最小二乘 表面平滑
    重建算法
    在测量较小的数据时会产生一些误差，这些误差所造成的不规则数据如果直接拿来曲面重建的话，
    会使得重建的曲面不光滑或者有漏洞，可以采用对数据重采样来解决这样问题，
    通过对周围的数据点进行高阶多项式插值来重建表面缺少的部分，
    1）用最小二乘法对点云进行平滑处理
    Moving Least Squares (MLS) surface reconstruction method 
    滑动最小二乘 表面平滑　重建算法

    虽说此类放在了Surface下面，但是通过反复的研究与使用，
    我发现此类并不能输出拟合后的表面，不能生成Mesh或者Triangulations，
    只是将点云进行了MLS的映射，使得输出的点云更加平滑。

    因此，在我看来此类应该放在Filter下。
    通过多次的实验与数据的处理，
    我发现此类主要适用于点云的光顺处理，
    当然输入的点云最好是滤过离群点之后的点集，
    否则将会牺牲表面拟合精度的代价来获得输出点云。

    Pcl::MovingLeastSquares<PointInT, PointOutT> mls
    其中PointInT决定了输入点云类型，
    PointOutT为点云输出类型（
    当法线估计标志位设置为true时，输出向量必须加上normals，这一点我们将会在成员函数里介绍）。
[滑动最小二乘 表面平滑](resampling.cpp)

## 把点云投影到平面上，在平面模型上提取凸（凹）多边形 最大轮廓

    对点云　直通滤波
    使用采样一致性分割算法　提取平面模型，
    再通过该估计的平面模型系数从滤波后的点云，投影一组点集　到　投影平面上(投影滤波算法)，
    最后为投影后的平面点云　计算其对应的　二维凸多边形（凸包　包围盒）
[凸（凹）多边形 ](convex_hull2d.cpp)

## 无序点云的快速三角化  
    得到点云文件的mesh网格文件ply 文件
    使用贪婪投影三角化算法对有向点云进行三角化，
    具体方法是：
    （1）先将有向点云投影到某一局部二维坐标平面内
    （2）在坐标平面内进行平面内的三角化
    （3）根据平面内三位点的拓扑连接关系获得一个三角网格曲面模型.
    贪婪投影三角化算法原理：
    是处理一系列可以使网格“生长扩大”的点（边缘点）
    延伸这些点直到所有符合几何正确性和拓扑正确性的点都被连上，
    该算法可以用来处理来自一个或者多个扫描仪扫描到得到并且有多
    个连接处的散乱点云但是算法也是有很大的局限性，
    它更适用于采样点云来自表面连续光滑的曲面且点云的密度变化比较均匀的情况.
[无序点云的快速三角化](greedy_projection.cpp)    
    
## 无序点云的B曲线拟合 
    > 1.7 支持
    [参考](https://blog.csdn.net/shenziheng1/article/details/54411098)

    step 1
    主成分分析法PCA初始化B样条曲面
    step 2
    拟合B样条曲面
    step 3
    循环初始化B样条曲线
    step 4
    拟合B样条曲线
    step 5
    三角化B样条曲面

    Monte Carlo particle filter (MCPF)蒙特卡洛粒子滤波 

    NURBS fitting stuff (curve and surfaces) using 
    PDM (point-distance-minimization　点距离最小化), 
    TDM (tangent-distance-minimization　切线距离最小化) and 
    SDM (squared-distance-minimization　平方距离最小化. 
[无序点云的B曲线拟合](bspline_fitting.cpp) 

=================================================

## 三维重构之泊松重构
    pcl::Poisson<pcl::PointNormal> pn ;
    通过本教程，我们将会学会：
        如果通过泊松算法进行三维点云重构。
        程序支持两种文件格式：*.pcd和*.ply
        程序先读取点云文件，然后计算法向量，
        接着使用泊松算法进行重构，最后显示结果。

    #include <pcl/surface/poisson.h>// 泊松算法进行重构
    ======================================================
    //创建Poisson对象，并设置参数
        pcl::Poisson<pcl::PointNormal> pn;
        pn.setConfidence(false); //是否使用法向量的大小作为置信信息。如果false，所有法向量均归一化。
        pn.setDegree(2); //设置参数degree[1,5],值越大越精细，耗时越久。
        pn.setDepth(8); 
         //树的最大深度，求解2^d x 2^d x 2^d立方体元。
         // 由于八叉树自适应采样密度，指定值仅为最大深度。

        pn.setIsoDivide(8); //用于提取ISO等值面的算法的深度
        pn.setManifold(false); //是否添加多边形的重心，当多边形三角化时。 
    // 设置流行标志，如果设置为true，则对多边形进行细分三角话时添加重心，设置false则不添加
        pn.setOutputPolygons(false); //是否输出多边形网格（而不是三角化移动立方体的结果）
        pn.setSamplesPerNode(3.0); //设置落入一个八叉树结点中的样本点的最小数量。无噪声，[1.0-5.0],有噪声[15.-20.]平滑
        pn.setScale(1.25); //设置用于重构的立方体直径和样本边界立方体直径的比率。
        pn.setSolverDivide(8); //设置求解线性方程组的Gauss-Seidel迭代方法的深度
        //pn.setIndices();

        //设置搜索方法和输入点云
        pn.setSearchMethod(tree2);
        pn.setInputCloud(cloud_with_normals);
        //创建多变形网格，用于存储结果
        pcl::PolygonMesh mesh;
        //执行重构
        pn.performReconstruction(mesh);

======================================
[三维重构之泊松重构](recon_poisson.cpp) 

## 三维重构之移动立方体算法

    #include <pcl/surface/marching_cubes_hoppe.h>// 移动立方体算法
    #include <pcl/surface/marching_cubes_rbf.h>
    ====================================
      //初始化 移动立方体算法 MarchingCubes对象，并设置参数
      pcl::MarchingCubes<pcl::PointNormal> *mc;
      mc = new pcl::MarchingCubesHoppe<pcl::PointNormal> ();
      //创建多变形网格，用于存储结果
      pcl::PolygonMesh mesh;

      //设置MarchingCubes对象的参数
      mc->setIsoLevel (0.0f);
      mc->setGridResolution (50, 50, 50);
      mc->setPercentageExtendGrid (0.0f);

      //设置搜索方法
      mc->setInputCloud (cloud_with_normals);

      //执行重构，结果保存在mesh中
      mc->reconstruct (mesh);
[三维重构之移动立方体算法重构](recon_marchingCubes.cpp) 

