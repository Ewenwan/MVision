# MSCKF  多状态约束卡尔曼滤波器

[论文 MSCKF 1.0 : A Multi-State Constraint Kalman Filter for Vision-aided Inertial Navigation ](http://intra.ece.ucr.edu/~mourikis/papers/MourikisRoumeliotis-ICRA07.pdf)

[论文 MSCKF 2.0 : High-Precision, Consistent EKF-based Visual-Inertial Odometry](http://intra.ece.ucr.edu/~mourikis/papers/Li2013IJRR.pdf)


[双目MSCKF视觉惯性里程计](https://github.com/Ewenwan/msckf_vio)


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
[多状态约束KF MSCKF & 滑动窗口滤波器SWF 论文](https://leeclem.net/assets/docs/crv2015_battle_paper.pdf)

      据说这也是谷歌tango里面的算法。
      在传统的EKF-SLAM框架中，特征点的信息会加入到特征向量和协方差矩阵里,
      这种方法的缺点是特征点的信息会给一个初始深度和初始协方差，
      如果不正确的话，极容易导致后面不收敛，出现inconsistent的情况。
      
      Msckf维护一个pose的FIFO，按照时间顺序排列，可以称为滑动窗口，
      一个特征点在滑动窗口的几个位姿都被观察到的话，
      就会在这几个位姿间建立约束，从而进行KF的更新。
![](https://images2015.cnblogs.com/blog/823608/201701/823608-20170120211841515-37024958.png)
      
      EKF-SLAM: 多个特征点同时约束一个相机位姿，进行KF更新
      MSCKF   : 一个特征点同时约束多个相机位姿(多相机观测同时优化，窗口多帧优化)，进行KF更新

      传统的 EKF-based SLAM 做 IMU 融合时，
      一般是每个时刻的 系统状态向量(state vector) 包含当前的 位姿pose、速度velocity、以及 3D map points 坐标等（
         IMU 融合时一般还会加入 IMU 的 bias（飘逸: 零飘和溫飘）），  
      然后用 IMU 做 预测predict step，
      再用 image frame 中观测 3D map points 的观测误差做 更新update step。

      MSCKF 的 motivation改进 是，EKF的每次 更新(类似优化)update step 是基于 3D map points 在单帧 frame 里观测的，
      如果能基于其在多帧中的观测效果应该会好（有点类似于 local bundle adjustment 的思想）。
      所以 MSCKF 的改进如下：
          预测阶段predict step 跟 EKF 一样，
          而 更新阶段update step 推迟到某一个 3D map point 在多个 frame 中观测之后进行计算，
          在 update 之前每接收到一个 frame，只是将 state vector 扩充并加入当前 frame 的 pose estimate。
      这个思想基本类似于 local bundle adjustment（或者 sliding window smoothing），
      在update step时，相当于基于多次观测同时优化 pose 和 3D map point。

[Event-based Visual Inertial Odometry 单目MSCKF视觉惯性里程计 论文](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhu_Event-Based_Visual_Inertial_CVPR_2017_paper.pdf)

[ros节点代码](https://github.com/Ewenwan/msckf_mono)

[Robust Stereo Visual Inertial Odometry for Fast Autonomous Flight 双目MSCKF视觉惯性里程计 论文](https://arxiv.org/pdf/1712.00036.pdf)

[ros节点代码](https://github.com/Ewenwan/msckf_vio)

[MSCKF 中文注释版 matlab](https://github.com/Ewenwan/MSCKF)

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
# 程序说明
********************程序中主要变量说明************************

## 参数说明：
    numLandmarks                     每帧图像特征点个数(地标数量)
    传感器物理器件参数：
    camera.c_u                       光心横坐标[u pixels]
    camera.c_v                       光心列坐标 [v pixels]
    camera.f_u                       焦距 [u pixels]
    camera.f_v                       焦距 [v pixels]
    传感器安装相关参数：
    camera.b                         双目基线距 [m]
    camera.q_CI                      4x1 IMU坐标系 到 Camera坐标系 位姿变换 T = [R,t]  中旋转矩阵 R，的四元素表示 q
    camera.p_C_I                     3x1 IMU坐标系 到 Camera坐标系 位姿变换 T = [R,t]  中平移变量p = t
    算法相关参数
    msckfParams.minTrackLength        特征点最少 相机 观测数目
    msckfParams.maxTrackLength        特征点最大 相机 观测数目
    msckfParams.maxGNCostNorm         高斯牛顿 优化求解 三角化特征点 的迭代误差上限
    msckfParams.minRCOND              矩阵条件数
    msckfParams.doNullSpaceTrick      是否做零空间映射
    msckfParams.doQRdecomp            是否做QR分解

## IMU状态：
    imuStates{k}.q_IG                 4x1 Global全局坐标系 到 IMU坐标系 的位姿变换 T = [R,t]  中旋转矩阵 R，的四元素表示 q
    imuStates{k}.p_I_G                3x1 IMU 在Global坐标系下位置( Global坐标系 到 IMU坐标系 的平移变量p)
    imuStates{k}.b_g                  3x1 陀螺仪(三轴旋转角速度)零偏
    imuStates{k}.b_v                  3x1 速度(三轴速度)零偏
    imuStates{k}.covar                12x12 IMU 状态协方差

## 相机状态：
    camStates{k}.q_CG                 4x1 Global全局坐标系 到 camera 坐标系 的位姿变换 T = [R,t] 中旋转矩阵 R，的四元素表示 q
    camStates{k}.p_C_G                3x1 Global坐标系下的 camera 位置( Global坐标系 到 camera 坐标系 的平移变量p )
    camStates{k}.trackedFeatureIds    1xM 当前相机可观测到的 特征的ID序列
    camStates{k}.state_k              当前相机ID(系统状态id)

## MSCKF状态：
    msckfState.imuState     IMU 状态
    msckfState.imuCovar     IMU-IMU 协方差矩阵块
    msckfState.camCovar     camera-camera 协方差矩阵块
    msckfState.imuCamCovar  IMU-camera 协方差
    msckfState.camStates    相机状态

## 特征跟踪列表：
    featureTracks        正在追踪的特征点（特征点坐标，能观测到这些特征点的相机，特征点ID）
    trackedFeatureIds    正在追踪特征点的ID号  
    
# 程序框架
********************程序步骤***********************************  

    步骤1：加载数据
    步骤2：初始化MSCKF中  imu测量协方差 与 预测协方差
    步骤3：导入 测量数据 与 参考数据
    步骤4：初始化MSCKF
          a. 将第一个 参考值的 四元数(位姿) 与 位置平移) 初始化为msckf中IMU的状态
          b. 目前MSCKF状态中只有 IMU相关的状态，没有camera相关的状态
          c. msckf中不将特征点作为状态，但也会记录跟踪的特征点。
             初始化时认为 第一帧所有特征点 都被跟踪上。
             
    从步骤5到步骤10循环进行：          
    步骤5：系统状态 与 协方差 预测更新。
          a. 状态预测更新
               a1. IMU状态更新（角速度更新四元数(位姿)，速度积分更新位置，陀螺仪零偏 和 速度零偏不变）
               a2. camera状态更新（保持和上一次相同）
               
          b. 协方差预测更新
               b1. IMU-IMU 状态的协方差更新，并对角化。（imuCover := P * imuCover * P' + G * Q_imu * G'）
               b2. IMU-camera 状态的协方差更新。（imuCamCovar := P * imuCamCovar）
               b3. camera-camera 状态的协方差更新。（camCovar := camCovar）
               
    步骤6：状态增广，在msckf状态 中 增广 相机状态
          a. 由IMU以及 IMU与camera的 固连关系 得到相机的位置和姿态。
          b. 增广雅克比。
             增广状态以后，需要得到 增广状态（相机位置、相机四元数）对 msckf状态（增广前以后的状态）的雅克比。
          c. 增广预测协方差矩阵，更新增广后的协方差矩阵
          
    步骤7：遍历当前帧所有特征点，更新featureTracks
          说明：msckf 中用 featureTracks 记录了 目前 被跟踪到的特征点。
               featureTracks 中包含每个特征点ID 和 观测值（即在所有能观测到该 特征点在相机坐标系下的齐次坐标）
          a. 遍历当前帧上所有特征点，判断是否属于featureTracks
          b. 如果该特征点 在视野范围内：
             将特征点在相机坐标系下的 齐次坐标 添加到 featureTracks 中对应特征的观测中。
          c. 如果该特征点超出视野范围 或者 能够观测到该特征的相机数目大于上限值
                c1. 从所有 相机状态 中 剔除该特征点，并将涉及到的相机状态添加到 状态待优化列表 中。
                c2. 如果待优化的相机状态超过最小跟踪长度（10），则将其添加到列表中用于优化。
                c3. 若已使用完给特征，则从 featureTracks 中剔除该特征点
          d. 如果该帧 检测的特征点在视野范围内，但是不属于featureTracks，则将其添加至featureTracks中。
          
    步骤8：MSCKF 测量更新。
         遍历所有用于优化的特征点, 构造观测模型（特征点 重投影误差），更新 MSCKF状态
          a. 通过特征点 所有 观测 估计出该特征点的 3D空间坐标位置。
          b. 通过 特征3D坐标 与 相机匹配特征点 之间的 重投影残差构造观测模型，包括 重投影误差 对 MSCKF状态量 的 雅克比矩阵的求解。
          c. 计算卡尔曼增益K，更新误差状态.=========================
          d. 根据 误差状态 更新MSCKF状态，x_true := x_nominal + detx
          e. 更新MSCKF 测量协方差 矩阵.
          
    步骤9：历史状态更新。
           从MSCKF状态中更新 IMU 的历史状态，通过相机的 状态 更新对应时刻imu的位姿状态.
           说明：重投影误差 只和 相机状态有关，
                msckf测量更新 只能 更新 相机有关的状态，
                因此在步骤8 测量更新后需要通过相机状态更新IMU状态.
           
    步骤10：剔除MSCKF中 需要被删除的 状态 和 对应的 协方差矩阵块.
           如果相机 不能观测到 featureTracks 中的任一特征（对重投影误差无贡献），则MSCKF状态中剔除该相机状态.
