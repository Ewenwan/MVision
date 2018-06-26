# MonoSLAM  扩展卡尔曼滤波 更新 相机特征点法得到的位姿 
    Extended Kalman Filter based Monocular SLAM
![](http://vision.ia.ac.cn/Students/gzp/ekfslamflow.jpg)

[代码](https://github.com/Ewenwan/SceneLib2)

[原理参考 EKF+ Monocular SLAM ](http://vision.ia.ac.cn/Students/gzp/monocularslam.html)

    Davison教授是视觉SLAM研究领域的先驱，
    他在2007年提出的MonoSLAM是第一个实时的单目视觉SLAM系统，被认为是许多工作的发源地。
    MonoSLAM以扩展卡尔曼滤波为后端，追踪前端非常稀疏的特征点。
    由于EKF在早期SLAM中占据着明显主导地位，所以MonoSLAM亦是建立在EKF的基础之上，
    以相机的当前状态和所有路标点为状态量，更新其均值和协方差。
    
    单目相机在一幅图像当中追踪了非常稀疏的特征点（且用到了主动追踪技术）。
    在EKF中，每个特征点的位置服从高斯分布，所以我们能够以一个椭球的形式表达它的均值和不确定性。
    在该图的右半部分，我们可以找到一些在空间中分布着的小球。
    它们在某个方向上显得越长，说明在该方向的位置就越不确定。
    我们可以想象，如果一个特征点收敛，
    我们应该能看到它从一个很长的椭球（相机Z方向上非常不确定）最后变成一个小点的样子。
    
    
    Real-Time Single Camera SLAM  单目摄像头的3D运动轨迹的算法
    本文设计了一种可以实时复现在未知场景里随机运动的单目摄像头的3D运动轨迹的算法。
    我们叫这个系统为MonoSLAM,它是第一个成功地利用一个移动端的不可控制的摄像头获得“纯粹的视觉”的系统
    ，在用移动的方法无法预知接口的情况下可以达到实时且无漂亮的表现。
    这个方法的核心是在一个概率框架内实时在线地创建一些稀疏但持久的自然landmark。 
    关键贡献点包括一种映射和测量的积极的方式，一种为摄像头平滑运动设计的通用移动模型的使用，
    以及单目特征初始化和特征方向估计的解决方案。
    综合以上，我们设计了非常有效和鲁棒的算法，
    在普通PC和摄像头可以运行到30HZ。 这些工作扩展了robotic系统的范围，也开辟了SLAM在其中应用的大门。
    最后我们把MonoSLAM应用于高性能全自由度的仿人机器人和人手持摄像头的增强现实中。

# 扩展卡尔曼滤波
![](http://vision.ia.ac.cn/Students/gzp/ekf.jpg)


# 基于相机状态 特征状态 的扩展卡尔曼系统


# 惯性传感器的校正 IMU-Camera Calibration
