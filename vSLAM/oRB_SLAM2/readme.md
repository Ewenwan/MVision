# ORB-SLAM2 ORB特征点法SLAM 支持单目、双目、rgbd相机

[安装测试](https://github.com/Ewenwan/MVision/blob/master/vSLAM/oRB_SLAM2/install.md)

[本文github链接](https://github.com/Ewenwan/MVision/blob/master/vSLAM/oRB_SLAM2/readme.md)

    ORB-SLAM是一个基于特征点的实时单目SLAM系统，在大规模的、小规模的、室内室外的环境都可以运行。
    该系统对剧烈运动也很鲁棒，支持宽基线的闭环检测和重定位，包括全自动初始化。
    该系统包含了所有SLAM系统共有的模块：
        跟踪（Tracking）、建图（Mapping）、重定位（Relocalization）、闭环检测（Loop closing）。
    由于ORB-SLAM系统是基于特征点的SLAM系统，故其能够实时计算出相机的轨线，并生成场景的稀疏三维重建结果。
    ORB-SLAM2在ORB-SLAM的基础上，还支持标定后的双目相机和RGB-D相机。
 
**系统框架**

![](https://img-blog.csdn.net/20161114115058814)  

**贡献**

![](https://img-blog.csdn.net/20161114115026626)


# 1. 相关论文：

[ORB-SLAM 单目Monocular特征点法](http://webdiis.unizar.es/%7Eraulmur/MurMontielTardosTRO15.pdf)

[ORB-SLAM2 单目双目rgbd](https://128.84.21.199/pdf/1610.06475.pdf)

[词袋模型DBoW2 Place Recognizer](http://doriangalvez.com/papers/GalvezTRO12.pdf)


> 原作者目录:

[Raul Mur-Artal](http://webdiis.unizar.es/~raulmur/)

[Juan D. Tardos](http://webdiis.unizar.es/~jdtardos/),

[J. M. M. Montiel](http://webdiis.unizar.es/~josemari/) 

[Dorian Galvez-Lopez](http://doriangalvez.com/)

([DBoW2](https://github.com/dorian3d/DBoW2))

# 2. 简介
    ORB-SLAM2 是一个实时的 SLAM  库，
    可用于 **单目Monocular**, **双目Stereo** and **RGB-D** 相机，
    用来计算 相机移动轨迹 camera trajectory 以及稀疏三维重建sparse 3D reconstruction 。
    
    在 **双目Stereo** 和 **RGB-D** 相机 上的实现可以得到真是的 场景尺寸稀疏三维 点云图
    可以实现 实时回环检测detect loops 、相机重定位relocalize the camera 。 
    提供了在  
[KITTI 数据集](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) 上运行的 SLAM 系统实例，支持双目stereo 、单目monocular。

[TUM 数据集](http://vision.in.tum.de/data/datasets/rgbd-dataset) 上运行的实例，支持 RGB-D相机 、单目相机 monocular, 

[EuRoC 数据集](http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)支持 双目相机 stereo 、单目相机 monocular.

    也提供了一个 ROS 节点 实时运行处理  单目相机 monocular, 双目相机stereo 以及 RGB-D 相机 数据流。
    提供一个GUI界面 可以来 切换 SLAM模式 和重定位模式

>**支持的模式: 

    1. SLAM Mode  建图定位模式
        默认的模式,三个并行的线程: 跟踪Tracking, 局部建图 Local Mapping 以及 闭环检测Loop Closing. 
        定位跟踪相机localizes the camera, 建立新的地图builds new map , 检测到过得地方 close loops.

    2. Localization Mode 重定位模式
        适用与在工作地方已经有一个好的地图的情况下。执行 局部建图 Local Mapping 以及 闭环检测Loop Closing 两个线程.  
        使用重定位模式,定位相机

# 3. 系统工作原理
    可以看到ORB-SLAM主要分为三个线程进行，
    也就是论文中的下图所示的，
![](https://img-blog.csdn.net/20161114115114018)    
    
    分别是Tracking、LocalMapping和LoopClosing。
    ORB-SLAM2的工程非常清晰漂亮，
    三个线程分别存放在对应的三个文件中，
    分别是：
    Tracking.cpp、
    LocalMapping.cpp　和
    LoopClosing.cpp　文件中，很容易找到。 

## A. 跟踪（Tracking）
      前端位姿跟踪线程采用恒速模型，并通过优化重投影误差优化位姿，
      这一部分主要工作是从图像中提取ORB特征，
      根据上一帧进行姿态估计，
      或者进行通过全局重定位初始化位姿，
      然后跟踪已经重建的局部地图，
      优化位姿，再根据一些规则确定新的关键帧。

## B. 建图（LocalMapping）
      局部地图线程通过MapPoints维护关键帧之间的共视关系，
      通过局部BA优化共视关键帧位姿和MapPoints，这一部分主要完成局部地图构建。
      包括对关键帧的插入，验证最近生成的地图点并进行筛选，然后生成新的地图点，
      使用局部捆集调整（Local BA），
      最后再对插入的关键帧进行筛选，去除多余的关键帧。

## C. 闭环检测（LoopClosing）
      闭环检测线程通过bag-of-words加速闭环匹配帧的筛选，
      并通过Sim3优化尺度，通过全局BA优化Essential Graph和MapPoints，
      这一部分主要分为两个过程：
      分别是：　闭环探测　和　闭环校正。
       闭环检测：　
       　　　　　先使用WOB进行探测，然后通过Sim3算法计算相似变换。
       闭环校正：
       　　　　　主要是闭环融合和Essential Graph的图优化。
      
## D. 重定位 Localization

    使用bag-of-words加速匹配帧的筛选，并使用EPnP算法完成重定位中的位姿估计。
         
         
# 4. 代码分析
[ORB-SLAM2详解（二）代码逻辑](https://blog.csdn.net/u010128736/article/details/53169832)
         
         
    
