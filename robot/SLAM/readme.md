# 同时定位和建图

[机器人学 —— 机器人感知（Mapping）](https://www.cnblogs.com/ironstark/p/5559439.html)

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
# 0. GraphSLAM


# 1  tiny-slam
[tiny-slam-ros-cpp代码](https://github.com/Ewenwan/tiny-slam-ros-cpp)

        tinySLAM是openSLAM中实现最为简单的一个SLAM方法，相比于ORB-SLAM之类的，
        这个代码的核心实现没有超过两百行，所以还是相对简单一些。
        主要是基于particle-filter进行的。
        
[博客参考](https://blog.csdn.net/lilynothing/article/details/62043583)

[tinySLAM代码逻辑结构](https://blog.csdn.net/lilynothing/article/details/62881142)  

[CoreSLAM（tinySLAM）主要思想就是基于粒子滤波器（particle filter）将激光数据整合到定位子系统中](https://blog.csdn.net/myarrow/article/details/80340548)

   
## 粒子滤波的基本思想是：
        通过寻找一组在状态空间中传播的随机样本来近似的表示概率密度函数，
        用样本均值代替积分运算，进而获得系统状态的最小方差估计的过程，
        这些样本被形象的称为“粒子”，故而叫粒子滤波。
        这篇文章是基于蒙特卡洛的这种粒子滤波的方法。具体的实现方法之后代码部分具体学习。 
        主要想法是基于粒子滤波器，将 激光的信息整合到定位系统中去。
        
        

# 2 Gmapping 基于粒子滤波 的 激光雷达slam
[Rao-Blackwellized Particle Filters 论文参考](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa12/optreadings/GrisettiStachnissBurgard_gMapping_T-RO2006.pdf)

[参考](https://blog.csdn.net/roadseek_zw/article/details/53316177)

[百度文库代码解析](https://wenku.baidu.com/view/3a67461550e2524de4187e4d.html)

        ROS 提供的 gmaping 包是用来生成地图的，它是对著名的开源 OpenSlam 包在 ROS 框架下的一个实现。
        这个包提供了对激光设备的 Slam,根据激光设备的输入和姿态数据从而建立一个基于网格的的2D地图。
        它需要从 ROS 系统监听许多 Topic，并输出一个 Topic——map(nav_msgs/OccupancyGrid)，
        这也是 RViz 的输入 Topic。

## 已知精确位姿（坐标和朝向）的地图创建:
        机器人位置已知，通过激光雷达扫描到环境特征，即障碍物距离。
        可通过机器人坐标和朝向以及障碍物距离计算出障碍物的坐标，采用bresenham直线段扫面算法，
        障碍物所处的栅格标注为occupy，机器人所处的栅格与障碍物所处的栅格之间画直线，
        直线所到之处都为free。当然每个栅格并不是简单的非0即1，栅格的占据可用概率表示，
        若某一个栅格在激光束a扫描下标识为occupy，在激光束b扫描下也标识为occupy，
        那该栅格的占据概率就变大，反之，则变小。
        这样，机器人所处的环境就可以通过二维栅格地图来表征。

## 如何在已知地图的情况下采用粒子滤波算法进行精确定位，
        一般包括以下几个步骤：
        （1）给定初始位姿，初始化粒子群，采用高斯分布进行随机采样；
        （2）根据运动模型模拟粒子运动；
        （3）计算粒子评分
                每个粒子的位姿即为假设的机器人位姿，采用bresenham直线段扫面算法，
                可计算出粒子当前位置与障碍物之间的栅格占据集合，
                计算出的栅格占据集合与给定的地图进行匹配计算，
                从而对每个粒子进行评分，选择得分高的粒子作为该时间点的机器人位姿。
        （4）粒子群重采样
                将评分低的粒子舍弃，将评分高且很接近的粒子都保留下来，并对评分高的粒子进行复制，保持粒子数量不变。




# cartographer
[官方](https://google-cartographer-ros.readthedocs.io/en/latest/tuning.html)
[代码](https://github.com/hitcm/cartographer)
[cartographer_ros](https://github.com/Ewenwan/cartographer_ros)

    谷歌在 2016年 10 月 6 日开源的 SLAM 算法
    基本思路 和 orbslam 类似。
    2D激光SLAM，利用激光数据进行匹配的算法。
    
[cartographer论文翻译](https://blog.csdn.net/lilynothing/article/details/60875825)

[论文解读2](https://note.youdao.com/share/?id=d8d15963d4577236399aa52c2cd968a7&type=note#/)

[Cartographer 代码逻辑](https://blog.csdn.net/lilynothing/article/details/62036559)

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
