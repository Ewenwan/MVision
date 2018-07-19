# 5. 基于优化的紧耦合举例-okvis   多目+IMU   使用了ceres solver的优化库。
[代码](https://github.com/Ewenwan/okvis)

[论文Keyframe-Based Visual-Inertial Odometry Using Nonlinear Optimization ](https://spiral.imperial.ac.uk/bitstream/10044/1/23413/2/ijrr2014_revision_1.pdf)

[OKVIS 笔记：位姿变换及其局部参数类](https://fzheng.me/2018/01/23/okvis-transformation/)

[OKVIS IMU 误差公式代码版本](https://blog.csdn.net/fuxingyin/article/details/53449209)

[OKVIS 代码框架](https://blog.csdn.net/fuxingyin/article/details/53428523)

[OKVIS 笔记](https://blog.csdn.net/fuxingyin/article/details/53368649)

![](https://images2015.cnblogs.com/blog/823608/201701/823608-20170120212125265-76552078.png)

      上图左边是纯视觉的odemorty,右边是视觉IMU融合的odemorty结构， 
      这个核心在于Frame通过IMU进行了联合， 
      但是IMU自身测量有一个随机游走的偏置， 
      所以每一次测量又通过这个偏置联合在了一起，
      形成了右边那个结构，对于这个新的结构， 
      我们需要建立一个统一的损失函数进行联合优化.
![](https://pic4.zhimg.com/v2-c00d0a55d9ff7bf23a4ed5249fb1090b_r.png)
      
      相对应于MSCKF的filter-based SLAM派系，OKVIS是keyframe-based SLAM派系做visual-inertial sensor fusion的代表。
      从MSCKF的思想基本可以猜出，OKVIS是将image观测和imu观测显式formulate成优化问题，一起去优化求解pose和3D map point。
      的确如此，OKVIS的优化目标函数包括一个reprojection error term(重投影误差)和一个imu integration error term(imu积分误差)，
      其中已知的观测数据是每两帧之间的feature matching(特征匹配)以及这两帧之间的所有imu采样数据的积分，
      注意imu采样频率一般高于视频frame rate，待求的是camera pose和3D map point，
      优化针对的是一个bounded window内的frames（包括最近的几个frames和几个keyframes）。
      
      需要注意的是，在这个optimization problem中，对uncertainty(不确定性，类似方差)的建模还是蛮复杂的。
      首先是对imu的gyro和accelerometer的bias(漂移)都需要建模，
      并在积分的过程中将uncertainty(不确定性，类似方差)也积分，
      所以推导两帧之间的imu integration error(imu积分误差)时，
      需要用类似于Kalman filter中predict step(预测阶段)里的，
      uncertainty propagation(不确定性传播)方式去计算covariance(协方差矩阵)。
      
      另外，imu的kinematics微分方程也是挺多数学公式，
      这又涉及到捷联惯性导航(strapdown inertial navigation)
      https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-696.pdf
      
      中相关的很多知识，推导起来不是很容易。
      这可以另起一个topic去学习了。
      
      OKVIS使用keyframe的motivation(创新点)是，由于optimization算法速度的限制，
      优化不能针对太多frames一起，所以尽量把一些信息量少的frames给marginalization(滤出)掉，
      只留下一些keyframes之间的constraints。关于marginalization的机制也挺有趣。
      
