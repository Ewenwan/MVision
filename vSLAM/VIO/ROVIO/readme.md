# 2. ROVIO，基于稀疏图像块的EKF滤波实现的VIO
      紧耦合，图像patch的稀疏前端(?)，EKF后端
      
[代码](https://github.com/Ewenwan/rovio)

[北邮的PangFuming去掉ROS后的ROVIO](https://github.com/Ewenwan/Rovio_NoRos)

[Robust Visual Inertial Odometry Using a Direct EKF-Based Approach 论文](https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/155340/eth-48374-01.pdf?sequence=1&isAllowed=y)

[参考](http://jinjaysnow.github.io/blog/2017-07/ROVIO%E8%A7%A3%E6%9E%90.html)


基于扩展卡尔曼滤波:**惯性测量用于滤波器的状态传递过程；视觉信息在滤波器更新阶段使用。**

    他的优点是：计算量小(EKF，稀疏的图像块)，但是对应不同的设备需要调参数，参数对精度很重要。
    没有闭环.
    没有mapping thread。
    经常存在误差会残留到下一时刻。
         
     
多层次patch特征处理：

![](http://jinjaysnow.github.io/images/rovio_feature.png)
      

[rovio_noros_laptop](https://github.com/Ewenwan/rovio_noros_laptop)

## 处理图像和IMU

      图像和IMU的处理沿用了PangFumin的思路。
      只是把输入换成了笔记本摄像头和MPU6000传感器
      。摄像头数据的读取放在了主线程中，IMU数据的读取放在了单独的线程中。
      主线程和IMU线程通过一个Queue共享数据。
## 处理图像和IMU

    图像和IMU的处理沿用了PangFumin的思路。只是把输入换成了笔记本摄像头和MPU6000传感器。
    摄像头数据的读取放在了主线程中，IMU数据的读取放在了单独的线程中。
    主线程和IMU线程通过一个Queue共享数据。
## 测试
    反复实验的经验如下：一开始的时候很关键。最好开始画面里有很多的feature。
    这里有很多办法，我会在黑板上画很多的feature，或者直接把Calibration Targets放在镜头前。
    如果开机没有飘（driftaway），就开始缓慢的小幅移动，让ROVIO自己去调整CameraExtrinsics。
    接着，就可以在房间里走一圈，再回到原点。

    我一开始以为如果有人走动，ROVIO会不准。但是实测结果发现影响有限。
    ROVIO有很多特征点，如果一两个特征点位置变动，ROVIO会抛弃他们。这里，ROVIO相当于在算法层面，解决了移动物体的侦测。
## ROVIO与VR眼镜

    一个是微软的Kinect游戏机，用了RGBD摄像头。
    而HTCvive和Oculus，则使用空间的两点和手中的摄像机定位。
    ROVIO没有深度信息，这也许就是为什么它容易driftaway。

    如果要将ROVIO用于产品上，可能还需要一个特定的房间，在这个房间里，有很多的特征点。
    如果房间里是四面白墙，恐怕ROVIO就无法运算了。

## 有限空间的定位

    但是如果环境是可以控制的，也就不需要ROVIO了。
    可以在房间里的放满国际象棋盘，在每个格子里标上数字，
    这样只需要根据摄像头视野中的四个角上的feature(数字)，就能确定位置了。
    这里不需要什么算法，因为这样的排列组合是有限的。
    只需要一个数据库就可以了。在POC阶段，可能会用数字。当然到了产品阶段会换成别的东西。
    
    
    
