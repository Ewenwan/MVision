# 紧耦合
      紧耦合方式使用 IMU 完成视觉 VO 中的运动估计 , 
      IMU 在图像帧间的积分的误差比较小 , IMU的数据可用于预测帧间运动 , 
      加速完成点匹配 , 完成VO 位姿估计 . 
      相对于松耦合 , 
      紧耦合的另外一个优点是 IMU 的尺度度量信息可以用于辅助视觉中的尺度的估计 .

# 基于滤波器的紧耦合 Filter-based Tightly Coupled method
      紧耦合需要把图像feature进入到特征向量去，
      因此整个系统状态向量的维数会非常高，因此也就需要很高的计算量。
      比较经典的算法是MSCKF，ROVIO
![](https://images2015.cnblogs.com/blog/823608/201701/823608-20170120211824921-442661944.png)
      
## 紧耦合举例-msckf
      据说这也是谷歌tango里面的算法。
      在传统的EKF-SLAM框架中，特征点的信息会加入到特征向量和协方差矩阵里,
      这种方法的缺点是特征点的信息会给一个初始深度和初始协方差，
      如果不正确的话，极容易导致后面不收敛，出现inconsistent的情况。
      
      Msckf维护一个pose的FIFO，按照时间顺序排列，可以称为滑动窗口，
      一个特征点在滑动窗口的几个位姿都被观察到的话，
      就会在这几个位姿间建立约束，从而进行KF的更新。
![](https://images2015.cnblogs.com/blog/823608/201701/823608-20170120211841515-37024958.png)
      
      EKF-SLAM: 多个特征点同时约束一个相机位姿，进行KF更新
      MSCKF   : 一个特征点同时约束多个相机位姿，进行KF更新
[Event-based Visual Inertial Odometry 单目MSCKF视觉惯性里程计 论文](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhu_Event-Based_Visual_Inertial_CVPR_2017_paper.pdf)
[ros节点代码](https://github.com/Ewenwan/msckf_mono)

[Robust Stereo Visual Inertial Odometry for Fast Autonomous Flight 双目MSCKF视觉惯性里程计 论文](https://arxiv.org/pdf/1712.00036.pdf)
[ros节点代码](https://github.com/Ewenwan/msckf_vio)

[MSCKF 中文注释版](https://github.com/NicoChou/MSCKF)

MSCKF算法流程框架:

      1. 初始化
            1.1 摄像机参数、
                噪声方差（图像噪声、IMU噪声、IMU的bias）、
                初始的IMU协方差、
                IMU和摄像机的外参数*、I
                MU和摄像机的时间偏移量*

            1.2 MSCKF参数：
                状态向量里滑动窗口大小的范围、
                空间点三角化误差阈值、
                是否做零空间矩阵构造 和 QR分解

            1.3 构造MSCKF状态向量

      2.读取IMU数据，估计新的MSCKF状态变量和对应的协方差矩阵

      3.图像数据处理

            3.1 MSCKF状态向量 中 增加当前帧的摄像机位姿；
                若位姿数大于滑动窗口大小的范围，
                去除状态变量中最早的视图对应的摄像机位姿.

            3.2提取图像特征并匹配，去除外点.

            3.3 处理所有提取的特征。
                  判断当前特征是否是之前视图中已经观察到的特征
                  3.3.1 如果当前帧还可以观测到该特征，则加入该特征的track列表
                  3.3.2 如果当前帧观测不到该特征(Out_of_View)，
                        将该特征的track加入到featureTracksToResidualize，用于更新MSCKF的状态变量.
                  3.3.3 给该特征分配新的featureID，并加入到当前视图可观测特征的集合

            3.4 循环遍历featureTracksToResidualize中的track，用于更新MSCKF的状态变量
                  3.4.1 计算每个track对应的三维空间点坐标
                       (利用第一幅视图和最后一幅视图计算两视图三角化，使用逆深度参数化和高斯牛顿优化求解)，
                       若三角化误差小于设置的阈值，则加入map集合.
                  3.4.2 计算视觉观测(即图像特征)的估计残差，并计算图像特征的雅克比矩阵.
                  3.4.3 计算图像特征雅克比矩阵的左零空间矩阵和QR分解，构造新的雅克比矩阵.

            3.5 计算新的MSCKF状态向量的协方差矩阵.
                  3.5.1 计算Kalman增益.
                  3.5.2 状态矫正.
                  3.5.3 计算新的协方差矩阵.

            3.6 状态变量管理
                  3.6.1 查找所有无feature track可见的视图集合deleteIdx.
                  3.6.2 将deleteIdx中的视图对应的MSCKF中的状态去除掉.
                  3.6.3 绘制运动轨迹.
