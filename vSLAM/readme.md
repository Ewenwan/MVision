# 目录
[移动机器人视觉里程计综述](http://html.rhhz.net/ZDHXBZWB/html/2018-3-385.htm#outline_anchor_20)

![](http://images2015.cnblogs.com/blog/276683/201511/276683-20151113165745244-1534145926.png)


**光流 双目 里程计 SLAM 区别**
![](https://github.com/Ewenwan/MVision/blob/master/vSLAM/img/flow_stereo_vo_slam.PNG)

**2d/3d 空间变换**
![](https://github.com/Ewenwan/MVision/blob/master/vSLAM/img/2D-planar-transformations.PNG)

[SLAM 开发学习资源与经验分享 ](https://github.com/Ewenwan/Lee-SLAM-source)

[AR 增强现实开发资源汇总 ](https://github.com/Ewenwan/AR-Source)

[VR 虚拟现实开发者必备资源汇总](https://github.com/Ewenwan/Lee-VR-Source)

[视觉SLAM基础知识总结](https://blog.csdn.net/myarrow/article/details/53704339)

[SLAM/VIO学习总结](https://zhuanlan.zhihu.com/p/34995102)
        
[Visual SLAM Tutorial ](http://frc.ri.cmu.edu/~kaess/vslam_cvpr14/)

[Features特征, Tracking跟踪, Essential Matrix本质矩阵, and RANSAC随机采样序列一致性](http://frc.ri.cmu.edu/~kaess/vslam_cvpr14/media/VSLAM-Tutorial-CVPR14-A11-VisualOdometry.pdf)

[Stereo Visual Odometry 双目视觉里程计](http://frc.ri.cmu.edu/~kaess/vslam_cvpr14/media/VSLAM-Tutorial-CVPR14-A12-StereoVO.pdf)

[BundleAdjustmen 集束优化](http://frc.ri.cmu.edu/~kaess/vslam_cvpr14/media/VSLAM-Tutorial-CVPR14-A13-BundleAdjustment.pdf)

[DealingWithScale 尺度问题](http://frc.ri.cmu.edu/~kaess/vslam_cvpr14/media/VSLAM-Tutorial-CVPR14-A21-DealingWithScale.pdf)

[Incremental 滤波器优化 非线性最小二乘优化等](http://frc.ri.cmu.edu/~kaess/vslam_cvpr14/media/VSLAM-Tutorial-CVPR14-A22-Incremental.pdf)

[LoopClosing闭环检测](http://frc.ri.cmu.edu/~kaess/vslam_cvpr14/media/VSLAM-Tutorial-CVPR14-A23-LoopClosing.pdf)

[大尺度地图高效构建](http://frc.ri.cmu.edu/~kaess/vslam_cvpr14/media/VSLAM-Tutorial-CVPR14-P11-LargeScaleEfficiency.pdf)

[稠密VO](http://frc.ri.cmu.edu/~kaess/vslam_cvpr14/media/VSLAM-Tutorial-CVPR14-P12-DenseVO.pdf)

[ptam svo 光流 ](http://frc.ri.cmu.edu/~kaess/vslam_cvpr14/media/VSLAM-Tutorial-CVPR14-P13-Hybrids.pdf)

[Dense mapping: KinectFusion and DTAM](http://frc.ri.cmu.edu/~kaess/vslam_cvpr14/media/VSLAM-Tutorial-CVPR14-P14-GreedyDenseSLAM.pdf)

[Kintinuous: Reconstruction of an Apartment ICP+RGBD](http://frc.ri.cmu.edu/~kaess/vslam_cvpr14/media/VSLAM-Tutorial-CVPR14-P21-Kintinuous.pdf)

[SLAM++ 目标水平地图 3d目标位置检测](http://frc.ri.cmu.edu/~kaess/vslam_cvpr14/media/VSLAM-Tutorial-CVPR14-P22-SLAM%2B%2B.pdf)

[稠密相机跟踪 误差雅克比矩阵求解](http://frc.ri.cmu.edu/~kaess/vslam_cvpr14/media/VSLAM-Tutorial-CVPR14-P12-DenseVO.pdf)

[LSD_slam & 激光雷达slam](http://www.cs.toronto.edu/~urtasun/courses/CSC2541/04_SLAM.pdf)

[视觉SLAM滑窗 局部全局优化 Double Window Optimisation for Constant Time Visual SLAM](http://www.doc.ic.ac.uk/~ajd/Publications/strasdat_etal_iccv2011.pdf)

[Visual Odometry(视觉里程计) StereoScan  viso2 ](https://github.com/Ewenwan/viso2)

闭环检测:

[DBoW2 二进制字符串特征 词袋模型](https://github.com/dorian3d/DBoW2)
[DBoW3 二进制、浮点型特征 词袋模型](https://github.com/rmsalinas/DBow3)
[FBOW  AVX,SSE and MMX指令集优化的 DBoW2 DBoW3](https://github.com/rmsalinas/fbow)
[haloc 图像特征哈希表示 图像与图像匹配](https://github.com/srv/libhaloc)


# 目录:

    ch1 Preface 
    ch2 Overview of SLAM & linux, cmake    概述，cmake基础
    ch3 Rigid body motion & Eigen          三维几何
    ch4 Lie group and Lie Algebra & Sophus 李群与李代数
    ch5 Cameras and Images & OpenCV        图像与相机模型
    ch6 Non-linear optimization & Ceres, g2o 非线性优化
    ch7 Feature based Visual Odometry      特征点法视觉里程计
    ch8 Direct (Intensity based) Visual Odometry 直接法视觉里程计
    ch9 Project
    ch10 Back end optimization & Ceres, g2o       后端优化1
    ch11 Pose graph and Factor graph & g2o, gtsam 位姿图优化
    ch12 Loop closure & DBoW3                     词袋方法 
    ch13 Dense reconstruction & REMODE, Octomap   稠密地图构建
    
    DTAM       像素点直接法稠密             2011年，Direct SLAM方法的鼻祖; 
    svo_slam   关键点直接法稀疏方法-半直接法 
    lsd_slam   纹理点直接法                 2014年一个半稠密SLAM系统           
    dso_slam   直接法   
    
    ORB_SLAM2  基于ORB特征点的 slam         2015年一个比较完整的基于特征点的SLAM系统
    PTAM       FAST角点                    2013 
    
    视觉惯性里程计Visual–Inertial Odometry(VIO)
    港科大的VIO VINS-Mono  A Robust and Versatile Monocular Visual-Inertial State Estimator
    ORB_SLAM2_IMU
    OKVIS: Open Keyframe-based Visual-Inertial SLAM.  OKVIS 属于 VIO（Visual Inertial Odometry），视觉融合 IMU 做 odometry。
    
    Dense + Direct: 
    基本方法：光流场的平滑度 + 几何误差（光流求导） 
    DTAM(直接法跟踪全部像素点，稠密方法)
    LSD（选取整幅图像中有梯度的部分来采用直接法，这种方法称为半稠密方法（simi-dense）），
    DSO ( )
    
    Sparse + Indirect:非直接法（即特征点法）SLAM，
    基本套路是：特征点+匹配+优化方法求解最小化重投影误差。 
    典型代表： 
    Mono-SLAM（扩展卡尔曼滤波 更新 相机特征点法得到的位姿
    PTAM(FAST角点),
    ORB-SLAM(ORB特征点),
    以及现在大部分SLAM 
    
    SVO（选取关键点来采用直接法，这类方法称为稀疏方法（sparse）, 结合了直接法和特征点法，因此，称它为半直接法）
    
      
    按照 2D−2D 数据关联方式的不同 ,视觉定位方法可以分为直接法、非直接法和混合法
    
    1. 直接法假设帧间光度值具有不变性 , 即相机运动前后特征点的灰度值是相同的 . 
       数据关联时 , 根据灰度值对特征点进行匹配，通过最小化光度误差，来优化匹配.
       直接法使用了简单的成像模型 , 
       适用于帧间运动较小的情形 , 但在场景的照明发生变化时容易失败 .
       
    2. 非直接法 , 又称为特征法 , 该方法提取图像中的特征进行匹配 , 
       最小化重投影误差得到位姿 . 图像中的特征点以及对应描述子用于数据关联 , 
       通过特征描述子的匹配 ,完成初始化中 2D−2D 以及之后的 3D−2D 的数据关联 .
       例如 ORB (Oriented FAST and rotatedBRIEF， ORBSLAM中 ) 、
            FAST (Features from accelerated seg-ment test) 、 
            BRISK (Binary robust invariant scalable keypoints) 、 
            SURF (Speeded up robustfeatures) , 
            或者直接的灰度块(PTAM中， 使用fast角点+灰度快匹配)
            可用于完成帧间点匹配。
            
    3. 混合法，又称为半直接法，结合直接法和特征点法
       使用特征点法中的特征点提取部分，而特征点匹配不使用 特征描述子进行匹配，
       而使用直接法进行匹配，利用最小化光度误差，来优化特征点的匹配，
       直接法中是直接对普通的像素点(DTAM),或者灰度梯度大的点(lsd-slam)进行直接法匹配。
    
    
    
    间接法与直接法的区别:
![](https://img-blog.csdn.net/20170428164809095?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvS2V2aW5fY2M5OA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

	ORB 、BRISK 等特征描述子、 LK光流法 等在 OpenCV均有实现 . 
	另外一个重要的问题是相机和 IMU 的标定问题 , 相机的标定中对于针孔相机 OpenCV Calib 和 MATLAB 相机标定工具箱使用了标准的模型 .
	Kalibr是一个工具箱 , 它能够标定多目相机系统、相机 IMU相对位姿和卷帘快门相机 . 
	常用的SFM工具 有Bundler 、 OpenMVG 和 MATLAB 多视几何工具箱等 .
	Bunlder 增量式地处理一组图像 , 提取其中的特征点进行匹配 , 完成三维重建并输出一个稀疏的场景结构 . 
	OpenMVG 则偏重于多视几何问题的求解 .
	优化方面 , 
	Sophus 库为三维空间的刚体变换及李群李代数 一 个 C++ 的 实 现 . 
	Eigen 为线性代数和 ( 稀疏 ) 矩阵的实现 , 对 LAPACK 实现了 C++ 的封装 . 
	g2o是一个针对非线性最小 二 乘 优 化 问 题 的 C++ 代 码 实 现 . 
	VO 问 题可以用图表示 , g2o 把非线性最小二乘问题表示为一个图或超图 , 图 的 边 可 以 连 接 多 个 节 点 ,
	一个超图是图的拓展问题 , 
	其他的优化实现还包括ceres.

# 基础资料
[基于视觉的 SLAM/Visual Odometry (VO) 开源资料、博客和论文列表](http://www.voidcn.com/article/p-amraedch-nh.html)
 
[视觉惯性里程计Visual–Inertial Odometry(VIO)概述](https://www.cnblogs.com/hitcm/p/6327442.html)
    
[视觉惯性单目SLAM （一）算法描述](https://blog.csdn.net/MyArrow/article/details/54694472)
[视觉惯性单目SLAM （二） 算法基础知识](https://blog.csdn.net/myarrow/article/details/55669033)
[视觉惯性单目SLAM （三）优化方法](https://blog.csdn.net/myarrow/article/details/56279273)
[视觉惯性单目SLAM （四）-泰勒展开式](https://blog.csdn.net/myarrow/article/details/56667168)
[视觉惯性单目SLAM （五）矩阵微积分](https://blog.csdn.net/myarrow/article/details/56834795)
[视觉惯性单目SLAM （五）-线性和二次型](https://blog.csdn.net/myarrow/article/details/58601497)

[VINS-Mono代码](https://github.com/Ewenwan/VINS-Mono)

[ORB_SLAM2_IMU](https://github.com/Ewenwan/LearnViORB_NOROS)

[OKVIS: Open Keyframe-based Visual-Inertial SLAM.](https://github.com/Ewenwan/okvis)



# 众人 SLAM 
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

	SLAM, Simultaneous Localization and Mapping,在一个未知得环境中创建并更新地图，同时跟踪智能体在地图中的位置 

[研究现状论文 Past, Present, and Future of Simultaneous Localization And Mapping: Towards the Robust-Perception Age](https://arxiv.org/abs/1606.05830)

# 参考书籍 Books
- [State Estimation for Robotic -- A Matrix Lie Group Approach](http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser15.pdf) by Timothy D. Barfoot, 2016
- [Simultaneous Localization and Mapping for Mobile Robots: Introduction and Methods](http://www.igi-global.com/book/simultaneous-localization-mapping-mobile-robots/66380) by Juan-Antonio Fernández-Madrigal and José Luis Blanco Claraco, 2012
- [Simultaneous Localization and Mapping: Exactly Sparse Information Filters ](http://www.worldscientific.com/worldscibooks/10.1142/8145/) by Zhan Wang, Shoudong Huang and Gamini Dissanayake, 2011
- [Probabilistic Robotics](http://www.probabilistic-robotics.org/) by Dieter Fox, Sebastian Thrun, and Wolfram Burgard, 2005
- [An Invitation to 3-D Vision -- from Images to Geometric Models](http://vision.ucla.edu/MASKS/) by Yi Ma, Stefano Soatto, Jana Kosecka and Shankar S. Sastry, 2005
- [Multiple View Geometry in Computer Vision](http://www.robots.ox.ac.uk/~vgg/hzbook/) by Richard Hartley and Andrew Zisserman, 2004
- [Numerical Optimization](http://home.agh.edu.pl/~pba/pdfdoc/Numerical_Optimization.pdf) by Jorge Nocedal and Stephen J. Wright, 1999



# 相关课程
- [SLAM Tutorial@ICRA 2016](http://www.dis.uniroma1.it/~labrococo/tutorial_icra_2016/)
- [Geometry and Beyond - Representations, Physics, and Scene Understanding for Robotics](http://rss16-representations.mit.edu/) at Robotics: Science and Systems (2016)
- [Robotics - UPenn](https://www.coursera.org/specializations/robotics) on Coursera by Vijay Kumar (2016)
- [Robot Mapping - UniFreiburg](http://ais.informatik.uni-freiburg.de/teaching/ws15/mapping/) by  Gian Diego Tipaldi and Wolfram Burgard (2015-2016)
- [Robot Mapping - UniBonn](http://www.ipb.uni-bonn.de/robot-mapping/) by Cyrill Stachniss (2016)
- [Introduction to Mobile Robotics - UniFreiburg](http://ais.informatik.uni-freiburg.de/teaching/ss16/robotics/) by Wolfram Burgard, Michael Ruhnke and Bastian Steder (2015-2016)
- [Computer Vision II: Multiple View Geometry  - TUM](http://vision.in.tum.de/teaching/ss2016/mvg2016) by Daniel Cremers ( Spring 2016)
- [Advanced Robotics - UCBerkeley](http://www.cs.berkeley.edu/~pabbeel/) by Pieter Abbeel (Fall 2015)
- [Mapping, Localization, and Self-Driving Vehicles](https://www.youtube.com/watch?v=x5CZmlaMNCs) at CMU RI seminar by John Leonard (2015)
- [The Problem of Mobile Sensors: Setting future goals and indicators of progress for SLAM](http://ylatif.github.io/movingsensors/) sponsored by Australian Centre for Robotics and Vision (2015)
- [Robotics - UPenn](https://alliance.seas.upenn.edu/~meam620/wiki/index.php?n=Main.HomePage) by Philip Dames and Kostas Daniilidis (2014)
- [Autonomous Navigation for Flying Robots](http://vision.in.tum.de/teaching/ss2014/autonavx) on EdX by Jurgen Sturm and Daniel Cremers (2014)
- [Robust and Efficient Real-time Mapping for Autonomous Robots](https://www.youtube.com/watch?v=_W3Ua1Yg2fk) at CMU RI seminar by Michael Kaess (2014)
- [KinectFusion - Real-time 3D Reconstruction and Interaction Using a Moving Depth Camera](https://www.youtube.com/watch?v=bRgEdqDiOuQ) by David Kim (2012)
- [SLAM Summer School](http://www.acfr.usyd.edu.au/education/summerschool.shtml) organized by Australian Centre for Field Robotics (2009)
- [SLAM Summer School](http://www.robots.ox.ac.uk/~SSS06/Website/index.html) organized by University of Oxford and Imperial College London (2006)
- [SLAM Summer School](http://www.cas.kth.se/SLAM/) organized by KTH Royal Institute of Technology (2002)


# 论文 Papers
- [Past, Present, and Future of Simultaneous Localization And Mapping: Towards the Robust-Perception Age](https://arxiv.org/abs/1606.05830) (2016)
- [Direct Sparse Odometry](https://arxiv.org/abs/1607.02565) (2016)
- [Modelling Uncertainty in Deep Learning for Camera Relocalization](https://arxiv.org/abs/1509.05909) (2016)
- [Large-Scale Cooperative 3D Visual-Inertial Mapping in a Manhattan World](http://mars.cs.umn.edu/papers/CM_line.pdf) (2016)
- [Towards Lifelong Feature-Based Mapping in Semi-Static Environments](http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/44821.pdf) (2016)
- [Tree-Connectivity: Evaluating the Graphical Structure of SLAM](http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=7487264) (2016)
- [Visual-Inertial Direct SLAM](webdiis.unizar.es/~jcivera/papers/concha_etal_icra16.pdf) (2016)
- [A Unified Resource-Constrained Framework for Graph SLAM](people.csail.mit.edu/lpaull/publications/Paull_ICRA_2016.pdf) (2016)
- [Multi-Level Mapping: Real-time Dense Monocular SLAM](https://groups.csail.mit.edu/rrg/papers/greene_icra16.pdf) (2016)
- [Lagrangian duality in 3D SLAM: Verification techniques and optimal solutions](http://arxiv.org/abs/1506.00746) (2015)
- [A Solution to the Simultaneous Localization and Map Building (SLAM) Problem](http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=938381&tag=1)
- [Simulataneous Localization and Mapping with the Extended Kalman Filter](http://www.iri.upc.edu/people/jsola/JoanSola/objectes/curs_SLAM/SLAM2D/SLAM%20course.pdf)

# 相关研究人员 Researchers

## 美国 United States
- [John Leonard](https://www.csail.mit.edu/user/817)
- [Sebastian Thrun](http://robots.stanford.edu/)
- [Frank Dellaert](http://borg.cc.gatech.edu/)
- [Dieter Fox](homes.cs.washington.edu/~fox/)
- [Stergios I. Roumeliotis](http://www-users.cs.umn.edu/~stergios/)
- [Vijay Kumar](http://www.kumarrobotics.org/)
- [Ryan Eustice](http://robots.engin.umich.edu/~ryan/)
- [Michael Kaess](http://frc.ri.cmu.edu/~kaess/)
- [Guoquan (Paul) Huang](http://udel.edu/~ghuang/)
- [Gabe Sibley](https://arpg.colorado.edu/people/)
- [Luca Carlone](http://www.lucacarlone.com/)
- [Andrea Censi](censi.mit.edu/)


## 欧洲 Europe
- [Paul Newman](http://mrg.robots.ox.ac.uk/)
- [Roland Siegwart](http://www.asl.ethz.ch/the-lab/people/person-detail.html?persid=29981)
- [Juan Nieto](http://www.nietojuan.com/)
- [Wolfram Burgard](http://www2.informatik.uni-freiburg.de/~burgard/)
- [Jose Neira](webdiis.unizar.es/~neira/)
- [Davide Scaramuzza](http://rpg.ifi.uzh.ch/people_scaramuzza.html)

## 澳大利亚 Australia
- [Cesar Cadena](http://cs.adelaide.edu.au/~cesar/)
- [Ian Reid](https://cs.adelaide.edu.au/~ianr/)
- [Tim Bailey](http://www-personal.acfr.usyd.edu.au/tbailey/)
- [Gamini Dissanayake](http://www.uts.edu.au/staff/gamini.dissanayake)
- [Shoudong Huang](http://services.eng.uts.edu.au/~sdhuang/)


# 数据集 Datasets

1.  [Intel Research Lab (Seattle)](http://kaspar.informatik.uni-freiburg.de/~slamEvaluation/datasets/intel.clf)


# 代码 Code

1.  [ORB-SLAM](https://github.com/raulmur/ORB_SLAM)  
2.  [LSD-SLAM](https://github.com/tum-vision/lsd_slam)
3.  [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2)
4.  [DVO: Dense Visual Odometry](https://github.com/tum-vision/dvo_slam)
5.  [SVO: Semi-Direct Monocular Visual Odometry](https://github.com/uzh-rpg/rpg_svo)
6.  [G2O: General Graph Optimization](https://github.com/RainerKuemmerle/g2o)
7.  [RGBD-SLAM](https://github.com/felixendres/rgbdslam_v2)



[参考源 Contributing pull request](https://github.com/kanster/awesome-slam/pulls).


# 数学知识
# 1. 数值优化算法
[数值优化算法](https://github.com/Ewenwan/Mathematics#%E6%95%B0%E5%80%BC%E4%BC%98%E5%8C%96) 
## 1. 函数f(x)的一阶泰勒展开
    对函数 f(x)一阶泰勒展开得 
    f(x+Δ)=f(x0) + f′(x0)Δ =   f(x0) + (x - x0)*f′(x0)     f'(x)为斜率 tan(a) 对边长度 = 斜率×直角边 = f'(x)×Δ
    求f(x)=0，得到  f(x0) + (x - x0)*f′(x0) = 0
    x = x0 − f(x0) / f′(x0) 不断迭代优化
    是函数f(x) 逼近0点附近
## 2. 函数f(x)的极值，是当其导数f'(x)=0时得到，所以对函数f'(x)一阶泰勒展开
    f′(x) ≈ f′(x0) + (x－x0)f′′(x0)
> 令  f′(x0) + (x－x0)f′′(x0) = 0

    得到 x = x0 − f'(x0) / f′'(x0) 不断迭代优化，使得 f'(x) 逼近0，使得f(x)达到极值
## 3. GS 高斯牛顿法 优化算法
    多维函数的一阶导数F'(X) = J 雅克比矩阵, X = [ x1, x2, x3, x4, ..., xn]
<img src="https://github.com/Ewenwan/Mathematics/blob/master/pic/1.png" width="600px" />

    多维函数的二阶导数F''(X) = H 海赛矩阵, X = [ x1, x2, x3, x4, ..., xn]
<img src="https://github.com/Ewenwan/Mathematics/blob/master/pic/2.png" width="600px" />

    当对多维函数F(X),进行优化时，原来一维函数优化迭代式子 x = x0 − f'(x0) / f′'(x0) 
    可写成 X = X0 - F'(X0)/F''(X0)*F(X0)
> X = X0 - J/H * F(X0)  = X0 - H逆 * J转置 * F(X0)
    
    雅克比矩阵(J) 代替了低维情况中的一阶导，海赛矩阵代替了二阶导，求逆代替了除法。
    因为 二阶导数求解太过耗时，使用 雅克比矩阵J来近似海赛矩阵H
    H ≈ J转置 * J
> 得到 X = X0 - H逆 * J转置 * F(X0) = X0 - (J转置 * J)逆 * J转置 * F(X0)

## LM Levenberg-Marquardt算法 莱文贝格－马夸特方法 优化算法
    调整 海赛矩阵H 的近似表达式
> H ≈ J转置 * J +  λI,  λ是一个可调参数， I是单位矩阵
    
    莱文贝格－马夸特方法（Levenberg–Marquardt algorithm）能提供数非线性最小化（局部最小）的数值解。
    此算法能借由执行时修改参数达到结合高斯-牛顿算法以及梯度下降法的优点，
    并对两者之不足作改善（比如高斯-牛顿算法 的逆矩阵不存在 或是 初始值离局部极小值太远）


> 在高斯牛顿迭代法中，我们已经知道 Δ = − (J转置 * J)逆 * J转置 * F(X0)
   
    X = X0 + Δ = X0 - (J转置 * J)逆 * J转置 * F(X0)
   
> 在莱文贝格－马夸特方法算法中则是 Δ = − (J转置 * J + λI )逆 * J转置 * F(X0)

    X = X0 + Δ = X0 - (J转置 * J + λI)逆 * J转置 * F(X0)

    Levenberg-Marquardt方法的好处就是在于可以调节
    如果下降太快，使用较小的λ，使之更接近高斯牛顿法
    如果下降太慢，使用较大的λ，使之更接近梯度下降法  Δ = − J转置 * F(X0)


# 2. 李群李代数知识

[李群李代数 反对称矩阵 指数映射 对数 刚体变换群SE3](https://blog.csdn.net/x_r_su/article/details/52749616)

[李群李代数 原版](https://blog.csdn.net/heyijia0327/article/details/50446140)

[李群李代数在计算机视觉中的应用](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6223449)

## 2.1 矢量叉乘 右手定则
    向量 u = (u1, u2, u3)
         v = (v1, v2, v3)
    两个向量叉乘，得到一个垂直与向量u和向量v组成的平面， 
        u × v = (u2*v3 - u3*v2      0  -u3  u2     v1
                 u3*v1 - u1*v3   =  u3  0  -u1  *  v2
                 u1*v2 - u2*v1)    -u2  u1  0      v3
        叉积性质 u × v * u = u × v * v  = 0向量，   且  u × v  = -v × u
        u × v  = u' * v, 向量 u 叉乘 上 向量v，可以转化成一个矩阵 与 u的向量乘积
        这个矩阵是一个3×3的实数矩阵，叫做 向量u的反对称矩阵，(我的叫法为叉乘矩阵)
              0  -u3  u2
        u'  = u3  0  -u1 
             -u2  u1  0 
        注意观察这个矩阵，它有一个特殊的性质
                         0   u3  -u2
        u'转置 = - u' = -u3  0    u1
                         u2  -u1  0

        反对称矩阵性质 A转置  = -A
        正交矩阵  性质 B转置  = B逆
## 2.2 旋转矩阵 与 SO3李群 so3李代数
[旋转矩阵理解](http://www.cnblogs.com/caster99/p/4703033.html)    

        旋转矩阵为正交矩阵，
        正交矩阵每一列都是单位向量，并且两两正交，
        正交矩阵的逆（inverse）等于正交矩阵的转置（transpose）
> 以三个欧拉角中的RotX为例（其余两个欧拉角以此类推，标准笛卡尔坐标系绕x轴旋转O角都，逆时针旋转为正方向）
<img src="https://github.com/Ewenwan/MVision/blob/master/vSLAM/lsd_slam/pic/Rx.png" wigth=700>

         1   0   0
    Rx = 0  cos sin
         0 -sin cos

        cos   0   sin
    Ry = 0    1   0
        -sin  0   cos

         cos  -sin  0
    Rz = sin   cos  0
          0     0   1
          
    验证，每一列是一个新的坐标系的向量，第一列为x，第二列为y，第三列为z
    可以验证没一列向量的模长为1 cos^2 + sin^2 =1
    且列与列相乘为0，即 两两正交
    且 转置 = 逆矩阵

>  R * R逆矩阵 = I单位矩阵 =  R * R转置

>  对上式求导得到  R' * R转置 + R * R'转置 = 0

>  R' * R转置 = - R * R'转置 = -(R' * R转置)转置

>  **得到  (R' * R转置)转置 =  - R' * R转置， R' * R转置 满足 反对称矩阵的性质**

>  **所以 R' * R转置 = w 为反对称矩阵**

    式子两边右乘 R，得到
    R' = w * R
    存在一个三维向量，其组成的反对称矩阵为 w
> **一个变量的导数等于它本身再乘以一个系数, exp(a * x)‘ = a * exp(a * x) 指数函数就满足这个性质**

                        1   0   0
    如果 R 为单位矩阵 I = 0   1   0
                        0   0   1
    则 R’  = w
    单位矩阵(标准笛卡尔坐标系空间)的导数，是一个三维向量W3 (3*1) 组成的反对称矩阵w , (3*3)
    这里 3×3 的矩阵表示的是一个空间，空间的 导数 是一个面
    类似的 一个二维曲线在某处的导数是一条切线

    求得导数之后就可以根据 泰勒展开式，进行近似表达
    R(t0 + dt) = R + R'*dt = I + w*dt

    李代数 so3
    我们把所有的 这些反对称矩阵集合起来就组成了一个所谓的 李代数 Lie algebra so3
    李群 SO3 
    把所有的旋转矩阵集合起来呢，就有了一个所谓的李群Lie group SO(3) 

    如果 R是单位矩阵，那它的导数就是一个反对称矩阵，所以只有反对称矩阵组成的空间，即 so(3)
    我们称之为在在单位矩阵处的正切空间tangent space.
    对于三维球面空间，其在某一点的导数，应该是一个切面。

    在非单位矩阵的R处的正切空间(导数空间)就是反对称矩阵乘以R
    R‘ = w * R

## 2.3 旋转矩阵的指数映射
    R(t)' = w * R(t)
    把旋转矩阵R用x替换掉，如下：
      x(t)' = w * x(t), 一个函数的导数 等于一个系数乘以它自身（指数函数 exp(a*t)）
    求解微分方程的到：
      x(t) = exp(w*t)*x(0)
    其中 exp(w*t) 是矩阵的指数映射，可以按照泰勒公式展开：
      exp(w*t) = I + w*t + (w*t)^2/2! + ... + (w*t)^n/n! + ...
    假设 R（0） = I，单位矩阵
      R(t) = exp(w*t)
    R(t)旋转矩阵，是正交矩阵，满足 转置 = 逆矩阵
    可以验证：
     （exp(w*t)）^(-1) = exp(-w*t)
    因为 反对称矩阵w满足 转置 = -本身
     （exp(w*t)）^(-1) = exp(-w*t) = exp(w转置 * t) = exp(w*t)转置

    我们说，可以将 反对称矩阵 w 的 so3李代数  通过 指数映射 （R(t) = exp(w*t)） 转换到 旋转矩阵R 的 SO3李群上去
    也即将 三维曲面空间 R 的导数，某点上的切平面上的一点 通过指数映射到 三维曲面空间上去

### 2.32 指数映射的展开  exp(w*t) = I + w*t + (w*t)^2/2! + ... + (w*t)^n/n! + ...
考虑一种简单情况，当反对称矩阵 对应的 3维向量的模长为1时,||w||=1，旋转速度恒定为1

     w
     w^2 = w*w
     w^3 = w*w*w = -w转置 * w * w = -w
     w^4 = -w^2
     w^5 = w
     w^6 = w^2
     w^7 = -w
     w^8 = -w^2
     w^9 = w
     ...

     exp(w*t)  = I + ( t - t^3/3! + t^5/5! - t^7/7! + t^9/9! - ...)*w + (t^2/2! - t^4/4! + t^6/6! - t^8/8! + t^10/10! - ...) * w^2
     注意括号内的内容分别就是 sin(t) 和 1-cos(t)
> exp(w*t)  = I + sin(t) * w  +  w^2*(1-cos(t))
     
> 当然，对于任意的旋转矩阵，我们也能够找到一组对应的W向量(反对称矩阵w) 和 t：    

    任意：旋转矩阵 R
        r11  r12  r13
    R = r21  r22  r23
        r31  r32  r33
    t = arccos((trace(R) - 1)/2) ， trace为矩阵的 迹，主对角线元素的和 
                                          r32 - r23
    W向量 = (W1, W2, W3) = 1/(2*sin(t)) * r13 - r31   和 反对称矩阵位置有关系  正-负
                                          r21 - r12
                    0   -W3   W2
    反对称矩阵 w  =  W3    0  -W1
                    -W2   W1  0

> 上面推导的是连续时间的，并且假设||w||=1，即旋转速度为单位速度，t是一个时间跨度，

> 联合起来的物理意义就是在单位旋转速度w下，经过时间t后，旋转了多少。

> 可是，我们常常见到的是另一种情况，单位时间t，旋转速度却不为单位模长了，||w||≠1。

> 表达成公式就是如下情况：

    速度不恒定就需要积分类求得？   
    exp(w) = I + w + w^2/2! + ... + w^n/n! + ...
           = I + SUM(w^(2i+1)/(2i+1)!  + w^(2i+2)/(2i+2)!) , 0<= i <无穷大
    w*3 = -(W^T * W)*w
    cet^2  = W^T * W   向量的模长 ||W|| 变量替换  原先为t
    w^(2i+1)  = (-1)^i * cet^(2i) * w
    w^(2i+1)  = (-1)^i * cet^(2i) * w^2

    exp(w) = I + SUM(w^(2i+1)/(2i+1)!  + w^(2i+2)/(2i+2)!)
           = I + SUM((-1)^i * cet^(2i)/(2i+1)! * w + (-1)^i * cet^(2i)/(2i+2)! * w^2)
           = I + (1 - cet^2/3! + cet^4/5! - ... )*w + (1/2! - cet^2/4! + cet^4/6! - ... )*w^2
           = I + sin(cet) / cet * w  + ((1-cos(cet))/cet^2) * w*2
           = I + sin(||W||)/||W|| * w + ((1-cos(||W||))/||W||^2) * w*2
> 这公式就只和W这个三维向量有关了

    t = arccos((trace(R) - 1)/2) ， trace为矩阵的 迹，主对角线元素的和 
                                          r32 - r23
    W向量 = (W1, W2, W3) = 1/(2*sin(t)) * r13 - r31   和 反对称矩阵位置有关系  正-负
                                          r21 - r12
                    0   -W3   W2
    反对称矩阵 w  =  W3    0  -W1
                   -W2   W1  0                                         
> ||W||的计算轻松加随意，三维向量变换成反对称矩阵也是容易，所以整个将三维旋转速度映射到旋转矩阵编程实现是不是也很容易了。        

## 2.4 欧式变换矩阵（刚体变换矩阵） T= [ R t，0 0 0 1] 李群SE(3) 李代数 se(3)
     求导  T导数* T逆 = [ R导数*R转置   t导数 - R导数*R转置*t，0 0 0 0] 
     存在一个反对称矩阵 w = R导数*R转置    3*3 实际有效的有三个量 W
     和一个三维向量     v = t导数 - w*t   实际有效的有三个量

     T导数* T逆  =  [ w   v，0 0 0 0]  = m   实际有效的有六个量

     T导数 =  m * T
     导数知道了
     可以对 T 利用泰勒展开进行近似
     T(t+dt) = T(t) + m(t) * T(t)dt = (I + m(t) * dt) * T(t)

     m 称为 在 曲线 T(t) 处的正切向量

     se(3) 刚体变换李代数 就是 m组成的集合

     T导数 =  m * T
     一个函数的导数为 一个系数乘以其自身
     则此函数为  指数函数      微分方程求解
     T(t) = exp(m*t)*T(0)
     m = [ w   v，
           0  0]

     假设 T(0) = I
     则
                        exp(w*t)  (I-exp(w*t))*w*v + W * W转置*v*t
     T(t) = exp(m*t) =    0                  1

     将 se(3) 通过指数映射 映射到 SE(3)

     去掉 时间t 的版本

     指数映射 
     李群 T = exp(李代数)
     T = exp(m) =  exp(w.)  V*v   =   R   t
                     0       1        0   1
     李代数 m = ln(T) 对数映射

     exp(w.) =  I + sin(||W||)/||W|| * w + ((1-cos(||W||)) / (||W||^2) ) * w*2

     V = I + ((1-cos(||W||))/(||W||^2)) * w  + (||W|| - sin(||W||) )/(||W||^3) * w^2 

     t = t1  t2  t3 平移向量

     v = t导数 - w*t 

     任意：旋转矩阵 R
         r11  r12  r13
     R = r21  r22  r23
         r31  r32  r33
     t = arccos((trace(R) - 1)/2) ， trace为矩阵的 迹，主对角线元素的和 
                                           r32 - r23
     W向量 = (W1, W2, W3) = 1/(2*sin(t)) * r13 - r31   和 反对称矩阵位置有关系  正-负
                                           r21 - r12
                     0   -W3   W2
     反对称矩阵 w  =  W3    0  -W1
                     -W2   W1  0

     各种论文里涉及到的求解位姿矩阵时的非线性最小二乘优化（牛顿法，LM法），
     其中增量都是在单位矩阵处的tangent space se(3)上计算，
     获得的增量(即相邻位姿变换关系)通过指数映射映射回多面体SE(3)上。

     通过这种方式，能够避免奇异点，保证很小的变换矩阵也能够表示出来。

## 2.5  李群 李代数 c++ 库 sophus  Eigen 矩阵运算c++库
### 2.5.1 sophus 库安装 
    * 本库为旧版本 非模板的版本
    * git clone https://github.com//strasdat/Sophus.git
    * git checkout a621ff   版本
    * 再cmake编译
### 2.5.2 变量简介
    SO(n)   特殊正交群   对应 n*n 的旋转矩阵 R  群(集合) 
    SE(n+1) 特殊欧式群   对应 n*n 的旋转矩阵和 n*1的平移向量 组合成的  变换矩阵T  群(集合)
    so(n)   SO(n)对应的李代数 为 so(n) n×n矩阵 3×3时实际有效向量为 3个 4*4是实际有效向量为3+3=6个  
            使得矩阵 和 代数 一一对应  可以使用代数的更新方法来更新 矩阵
    SO(3)  表示三维空间的 旋转矩阵 集合 R 3×3
    SE(3)  表示三维空间的 变换矩阵 集合 T 4×4
    李代数 so3的本质就是个三维向量，直接Eigen::Vector3d定义（简化表示），实际是由这三维向量对应的反对称矩阵
    李代数 se3的本质就是个六维向量，3个旋转 + 3个平移，实际是一个4*4的矩阵，可有效向量数量为6个

    任意：旋转矩阵 R   平移向量 t
        r11  r12  r13
    R = r21  r22  r23
        r31  r32  r33
    cet = arccos((trace(R) - 1)/2) ， trace为矩阵的 迹，主对角线元素的和 
                                                               r32 - r23
    W向量(so3本质的三维向量) = (W1, W2, W3) = 1/(2*sin(cet)) *  r13 - r31   和 反对称矩阵位置有关系  正-负
                                                               r21 - r12
                    0   -W3   W2
    反对称矩阵 w  =  W3    0  -W1    这个反对称矩阵实际为 so3
                    -W2   W1  0
    
     se3 m = [ w  v，    4*4的矩阵 实际有效变量有 3+3 六个
               0  0]
      
     v = t导数 - w*t 
### 2.5.3 Eigen 矩阵运算c++库
     1. 旋转向量   Eigen::AngleAxisd    角度 + 轴   
        Eigen::AngleAxisd rotation_vector ( M_PI/4, Eigen::Vector3d ( 0,0,1 ) );//沿 Z 轴旋转 45 度
     2. 旋转矩阵   Eigen::Matrix3d      3*3  R
        rotation_vector.toRotationMatrix(); //旋转向量转换 到 旋转矩阵
        // 旋转向量 直接转 旋转矩阵 
        Eigen::Matrix3d Eigen::Matrix3d R = Eigen::AngleAxisd(M_PI/2, Eigen::Vector3d(0,0,1)).toRotationMatrix();
     3. 欧拉角向量 Eigen::Vector3d      3*1  r, p, y
        Eigen::Vector3d rotation_matrix.eulerAngles ( 2,1,0 ); 
        // ( 2,1,0 )表示ZYX顺序，即roll pitch yaw顺序  旋转矩阵到 欧拉角转换到欧拉角
     4. 四元素    Eigen::Quaterniond  
        Eigen::Quaterniond q = Eigen::Quaterniond ( rotation_vector );// 旋转向量 定义四元素 
        q = Eigen::Quaterniond ( rotation_matrix );                   //旋转矩阵定义四元素
     5. 欧式变换矩阵 Eigen::Isometry3d   4*4  T 
        Eigen::Isometry3d  T=Eigen::Isometry3d::Identity(); // 虽然称为3d，实质上是4＊4的矩阵   旋转 R+ 平移t 
        T.rotate ( rotation_vector );                       // 按照rotation_vector进行旋转
        也可 Eigen::Isometry3d  T(q)                        // 一步 按四元素表示的旋转 定义 变换矩阵
        T.pretranslate ( Eigen::Vector3d ( 1,3,4 ) );       // 把平移向量设成(1,3,4)
        cout<< T.matrix() <<endl;

### 2.5.4 欧拉角定义：
      1. 旋转向量定义的 李群SO(3) 
         Sophus::SO3 SO3_v( 0, 0, M_PI/2 ); 
         // 亦可从旋转向量构造  这里注意，不是旋转向量的轴，是对应轴 + 轴上的旋转
         相当于把轴 和 旋转角度 放在一起表示了  (0, 0, M_PI/2) 表示的就是 绕 Z轴(0,0,1), 旋转M_PI/2角度

      2. 旋转向量 转 旋转矩阵                     角        Z轴(0,0,1)
         Eigen::Matrix3d R = Eigen::AngleAxisd(M_PI/2, Eigen::Vector3d(0,0,1)).toRotationMatrix();

      3. 旋转矩阵定义的 李群SO(3)
         ophus::SO3 SO3_R(R);     // Sophus::SO(3)可以直接从旋转矩阵构造

      4. 旋转矩阵 转 四元素         
         Eigen::Quaterniond q(R); // 或者四元数(从旋转矩阵构造)

      5. 四元素定义的  李群SO(3)   
         Sophus::SO3 SO3_q( q );

      6. 李代数so3 为李群SO(3) 的对数映射  
         Eigen::Vector3d so3 = SO3_R.log();

      7. 平移  向量表示 
         Eigen::Vector3d t(1,0,0);   // 沿X轴平移1

      8. 从旋转矩阵R 和 平移t 构造 欧式变换矩阵李群 SE3      
         Sophus::SE3 SE3_Rt(R, t);   // 从R,t构造SE(3)

      9. 从四元素q和 平移t 欧式变换矩阵李群 SE3      
         Sophus::SE3 SE3_qt(q,t);   // 从q,t构造SE(3)

      10. 李代数se(3) 是一个6维向量   为李群SE3 的对数映射
          typedef Eigen::Matrix<double,6,1> Vector6d;// Vector6d指代　Eigen::Matrix<double,6,1>
          Vector6d se3 = SE3_Rt.log();
# 3. 优化算法
# 3.1 BA集束优化算法：
[BA集束优化算法](http://www.cnblogs.com/gaoxiang12/p/5304272.html)

	将一个图像中的二维像素点根据相机内参数和深度信息投影到三维空间,
	再通过欧式变换关系[R t]变换到另一个图像的坐标系下，
	再通过相机内参数投影到其像素平面上。
	可以求的的误差，使用优化算法，更新[R t]来使得误差最小

	当把　三维点Pi　也作为优化参数时，可以这样考虑
	图像1 像素点 p = {p1,p2,p3,...,pn}  qi = [u,v]
	图像2 像素点 q = {q1,q2,q3,...,qn}
	物理空间3D点 P = {P1,P2,P3,...,Pn} 坐标系为图像1的坐标系 Pi = [X,Y,Z]
	相机内参数K = [fx, 0,  ux
		   0,  fy, uy
		   0,  0,  1]
	  K_inv = [f1/x, 0,  -ux
		   0,  1/fy, -uy
		   0,  0,    1] 

	则有：
	  3D点 Pi 投影到 图像1的像素坐标系下
	  K*Pi = fx*X + Z*ux   x1              d1*u'
		 fy*Y + Z*uy = y1 = d1 * pi' = d1*v'  d1为3D点在图像1下的深度
		 Z             z1              d1

		      u'
	   投影点pi'    v' =  1/d1 *  K*Pi  值　约接近　pi　误差越小
		       1

	3D点 Pi 投影到 图像2的像素坐标系下
	K*(R*Pi+t)=     x2               
			  y2 = d2 * qi'
			  z2
	  投影点　qi'  = 1/d2 *  K*(R*Pi+t)   d2为3D点在图像2下的深度
     
## 3.2 传统数学方法，可以消去Pi得到只关于　pi,qi,R,t的关系，可以使用对极几何和基础局E进行求解
        　理论上使用 8个匹配点对就可以求解
          可以使用RanSac 随机序列采样一致性方法获取更鲁棒的解　[R,t]
          
## 3.3 最小二乘图优化方法　最小化误差求解
    使用 差平方和作为 误差函数：
       E = sum( (1/d1 *  K*Pi - [pi,1])^2 + (1/d2 *  K*(R*Pi+t) - [qi,1]^2) )
    求解Pi,R,t 使得　E最小
    它叫做最小化重投影误差问题（Minimization of Reprojection error）。
    在实际操作中，我们实际上是在调整每个　Pi，使得它们更符合每一次观测值pi和qi,
    也就是使每个误差项都尽量的小,
    由于这个原因，它也叫做捆集调整（Bundle Adjustment）。

    上述方程是一个非线性函数，该函数的额最优化问题是非凸的优化问题，

    求解E的极值，当E的导数为０时，取得

    那么如何使得　E的导数　E'=0呢?

    对　E'　进行 泰勒展开

      一阶泰勒展开　: E‘(x) =  E’(x0) + E’‘(x0)  * dx 
                    =  J  + H * dx = 0
                     dx = -H逆 * J转置 * E(x0)
                     也可以写成：
                     H * dx = -J转置 * E(x0)

    求解时，需要求得函数 E 对每一个优化变量的　偏导数形成偏导数矩阵(雅克比矩阵)J
    二阶偏导数求解麻烦使用一阶偏导数的平方近似代替
    H = J转置*J

    可以写成如下线性方程：
     J转置*J * dx = -J转置 * E(x0)
     这里　误差E(x0)可能会有不同的执行度　可以在其前面加一个权重　w
     J转置*J * dx = -J转置 * w * E(x0)

     A * dx = b    GS高斯牛顿优化算法

     (A + λI) = b   LM 莱文贝格－马夸特方法 优化算法

     Levenberg-Marquardt方法的好处就是在于可以调节
     如果下降太快，使用较小的λ，使之更接近高斯牛顿法
     如果下降太慢，使用较大的λ，使之更接近梯度下降法  Δ = − J转置 * F(X0)

     这里线性方程组的求解　多使用　矩阵分解的算法　常见 LU分解、LDLT分解和Cholesky分解、SVD奇异值分解等

     所以这里需要：
              1. 系数矩阵求解器，来求解　雅可比矩阵J　和　海塞矩阵H, BlockSolver；
              2. 数值优化算法　GS高斯牛顿优化算法/LM 莱文贝格－马夸特方法 优化算法
                         计算　 A /   (A + λI)   
              3. 线性方程求解器，从 PCG, CSparse, Choldmod中选
	      
## 3.4 g2o 算法库简介  
     g2o 全称 general graph optimization，是一个用来优化非线性误差函数的c++框架。
     稀疏优化 SparseOptimizer 是我们最终要维护的东东。
     它是一个Optimizable Graph，从而也是一个Hyper Graph。
     一个 SparseOptimizer 含有很多个顶点 （都继承自 Base Vertex）和
     很多种边（继承自 BaseUnaryEdge, BaseBinaryEdge或BaseMultiEdge）。
     这些 Base Vertex 和 Base Edge 都是抽象的基类，而实际用的顶点和边，都是它们的派生类。
     我们用 
     SparseOptimizer.addVertex 和 
     SparseOptimizer.addEdge   向一个图中添加顶点和边，
     最后调用 SparseOptimizer.optimize 完成优化。

     在优化之前，需要指定我们用的求解器和迭代算法。
     一个 SparseOptimizer 拥有一个 
         迭代算法 Optimization Algorithm，
         继承自Gauss-Newton,  Levernberg-Marquardt, Powell's dogleg 三者之一（我们常用的是GN或LM）。

     同时，这个 Optimization Algorithm 拥有一个 Solver，它含有两个部分：
        1. 一个是 SparseBlockMatrix ，用于计算稀疏的雅可比和海塞,BlockSolver；
        2. 一个是用于计算迭代过程中最关键的一步，线性方程组求解器；
             H * Δx = −b
             这就需要一个线性方程的求解器。
             而这个求解器，可以从 PCG, CSparse, Choldmod 三者选一。

     综上所述，在g2o中选择优化方法一共需要三个步骤：
       1.  选择一个线性方程求解器，从 PCG, CSparse, Choldmod中选，实际则来自 g2o/solvers 文件夹中定义的东东。
       2.  选择一个 BlockSolver 。
       3.  选择一个迭代优化更新策略，从GN, LM, Doglog中选。
[G2O图优化demo和理论推导](https://github.com/Ewenwan/MVision/blob/master/vSLAM/lsd_slam/g2o%E5%9B%BE%E4%BC%98%E5%8C%96.md)

[卡尔曼滤波推到+应用](https://github.com/Ewenwan/MVision/blob/master/vSLAM/lsd_slam/%E5%8D%A1%E5%B0%94%E6%9B%BC%E6%BB%A4%E6%B3%A2.md)

[ g2o的使用 ](https://www.cnblogs.com/gaoxiang12/p/5304272.html)

[代码双目BA实例 ](https://github.com/gaoxiang12/g2o_ba_example)

# 4. 单目相机  

[单目相机 slam基础 ](https://github.com/Ewenwan/MVision/blob/master/vSLAM/%E5%8D%95%E7%9B%AEslam%E5%9F%BA%E7%A1%80.md)

[lsd_slam 单目直接法](https://github.com/Ewenwan/MVision/blob/master/vSLAM/lsd_slam/readme.md)

[svo 单目半直接](https://github.com/Ewenwan/MVision/blob/master/vSLAM/svo_slam/readme.md)

# 5. 双目相机  

[双目相机 slam基础](https://github.com/Ewenwan/MVision/blob/master/vSLAM/%E5%8F%8C%E7%9B%AEslam%E5%9F%BA%E7%A1%80.md)

[orb_slam2 单目双目rgbd 特征点法]()
