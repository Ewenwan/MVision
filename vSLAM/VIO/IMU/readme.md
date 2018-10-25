# IMU算法# 惯性传感器（IMU）

[RIDI: Robust IMU Double Integration](https://github.com/Ewenwan/ridi_imu)

[IMUProject](https://github.com/Ewenwan/IMUProject)

[ros节点 MadgwickAHRS 和 MahonyAHRS ](https://github.com/Ewenwan/imu_proc)

[从零开始的 IMU 状态模型推导](https://fzheng.me/2016/11/20/imu_model_eq/)

      能够测量传感器本体的角速度和加速度，被认为与相机传感器具有明显的互补性，
      而且十分有潜力在融合之后得到更完善的SLAM系统。
      
      1、IMU虽然可以测得角速度和加速度，但这些量都存在明显的漂移（Drift），
         使得积分两次得到的位姿数据非常不可靠。
          好比说，我们将IMU放在桌上不动，用它的读数积分得到的位姿也会漂出十万八千里。
          但是，对于短时间内的快速运动，IMU能够提供一些较好的估计。
          这正是相机的弱点。
          当运动过快时，（卷帘快门的）相机会出现运动模糊，
          或者两帧之间重叠区域太少以至于无法进行特征匹配，
          所以纯视觉SLAM非常害怕快速的运动。
          而有了IMU，即使在相机数据无效的那段时间内，
          我们也能保持一个较好的位姿估计，这是纯视觉SLAM无法做到的。
          
      2、相比于IMU，相机数据基本不会有漂移。
         如果相机放在原地固定不动，那么（在静态场景下）视觉SLAM的位姿估计也是固定不动的。
         所以，
         相机数据可以有效地估计并修正IMU读数中的漂移，使得在慢速运动后的位姿估计依然有效。
         
      3、当图像发生变化时，本质上我们没法知道是相机自身发生了运动，
         还是外界条件发生了变化，所以纯视觉SLAM难以处理动态的障碍物。
         而IMU能够感受到自己的运动信息，从某种程度上减轻动态物体的影响。
         
[IMU 互补滤波算法complementary filter 对 Gyroscope Accelerometer magnetometer 融合](http://www.pieter-jan.com/node/11)

[IMU Data Fusing: Complementary, Kalman, and Mahony Filter](http://www.olliw.eu/2013/imu-data-fusing/)

[Mahony&Madgwick 滤波器 Mahony显式互补滤波](http://chenbinpeng.com/2016/10/08/ECF/)

[Mahony 论文](http://chenbinpeng.com/2016/10/08/ECF/A%20Complementary%20Filter%20for%20Attitude%20Estimation%20of%20a%20Fixed-Wing%20UAV.pdf)

[Google Cardboard的九轴融合算法 —— 基于李群的扩展卡尔曼滤波](http://www.cnblogs.com/ilekoaiq/p/8710812.html)

[Madgwick算法详细解读 陀螺仪积分的结果和加速度计磁场计优化的结果加权，就可以得到高精度的融合结果 ](http://www.cnblogs.com/ilekoaiq/p/8849217.html)

      <<Google Cardbord的九轴融合算法>> <<Madgwick算法>>，<<互补滤波算法>>
      讨论的都是在SO3上的传感器融合，
      即，输出的只是纯旋转的姿态。
      只有旋转，而没有位移，也就是目前的一些普通的VR盒子的效果。 
      
      后面 imu+相机的融合是，在SE3上面的传感器融合，在既有旋转又有位移的情况下，该如何对多传感器进行融合。
      
      
[IMU 数据融合](https://blog.csdn.net/haithink/article/details/79975679)

[四元数AHRS姿态解算和IMU姿态解算分析](http://www.bspilot.com/?p=121)
