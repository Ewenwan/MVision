# 视觉惯性里程计 VIO - Visual Inertial Odometry

      IO和之前的几种SLAM最大的不同在于两点：
        首先，VIO在硬件上需要传感器的融合，包括相机和六轴陀螺仪，
             相机产生图片，
             六轴陀螺仪产生加速度和角速度。
             相机相对准但相对慢，六轴陀螺仪的原始加速度如果拿来直接积分会在很短的时间飘走（zero-drift），
             但六轴陀螺仪的频率很高，在手机上都有200Hz。
        其次，VIO实现的是一种比较复杂而有效的卡尔曼滤波，比如MSCKF（Multi-State-Constraint-Kalman-Filter），
             侧重的是快速的姿态跟踪，而不花精力来维护全局地图，
             也不做keyframe based SLAM里面的针对地图的全局优化（bundle adjustment）。
             最著名的商业化实现就是Google的Project Tango和已经被苹果收购的Flyby Media，

      其中第二代Project Tango搭载了Nividia TK1并有主动光源的深度摄像头的平板电脑，
      这款硬件可谓每个做算法的小伙伴的梦幻搭档，具体在这里不多阐述。
