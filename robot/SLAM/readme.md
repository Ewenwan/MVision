# 同时定位和建图
# 激光雷达SLAM
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



# 1. GraphSLAM

# Gmapping

# cartographer
[官方](https://google-cartographer-ros.readthedocs.io/en/latest/tuning.html)
[代码](https://github.com/hitcm/cartographer)

    谷歌在 2016年 10 月 6 日开源的 SLAM 算法
    基本思路 和 orbslam 类似。
    2D激光SLAM，利用激光数据进行匹配的算法。
    
## 框架：
    惯导追踪 ImuTracker、
    位姿估算 PoseExtrapolator、
    自适应体素滤波 AdaptiveVoxelFilter、
    扫描匹配、
    子图构建、
    闭环检测和图优化。

    用Grid（2D/3D）（栅格地图）的形式建地图；
    局部匹配直接 建模 成一个非线性优化问题，
    利用IMU提供一个比较靠谱的初值；
    后端用Graph来优化，用 分支定界算法 来加速；
    2D和3D的问题统一在一个框架下解决。

## 先来感受一下算法的设计目标：
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
    
# Ethzasl MSF Framework 多传感器融合框架 扩展卡尔曼滤波
[多传感器卡尔曼融合框架 Multi-sensor Fusion MSF](http://wiki.ros.org/ethzasl_sensor_fusion/Tutorials/Introductory%20Tutorial%20for%20Multi-Sensor%20Fusion%20Framework)

    多传感器融合是机器人导航上面一个非常基本的问题，
    通常在一个稳定可用的机器人系统中，会使用视觉（RGB或RGBD）、激光、IMU、马盘
    等一系列传感器的数据来最终输出一个稳定和不易丢失的姿态。
    Ethzasl MSF Framework 是一个机器人上面的多传感器融合框架，它使用了扩展卡尔曼的原理对多传感器进行融合。
    同时内部使用了很多工程上的 trick 解决多传感器的刷新率同步等问题，API 封装也相对简单，非常适合新手使用。
