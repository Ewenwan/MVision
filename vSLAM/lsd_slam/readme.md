# lsd是一个 大规模的 单目直接法 视觉半稠密 slam系统

[lad源码解析 参考解析](https://blog.csdn.net/lancelot_vim)

[lad算法分析 代码分析 安装 非ros改造](http://www.cnblogs.com/hitcm/category/763753.html)

[算法数学基础](https://blog.csdn.net/xdEddy/article/details/78009748)
 
[tracking  optimizationThreadLoop线程 分析等](https://blog.csdn.net/u013004597)

https://blog.csdn.net/tiandijun/article/details/62226163

[官网:](http://vision.in.tum.de/research/vslam/lsdslam)
[代码:](https://github.com/tum-vision/lsd_slam)

# 运行lsd-slam
[一个来自官方的范例，使用的dataset如下，400+M](http://vmcremers8.informatik.tu-muenchen.de/lsd/LSD_room.bag.zip)

    解压之
    然后运行下面的3个命令，即可看到效果
    rosrun lsd_slam_viewer viewer
    rosrun lsd_slam_core live_slam image:=/image_raw camera_info:=/camera_info
    rosbag play ./LSD_room.bag

    平移，旋转，相似以及投影变换，在lsd-slam中，有个三方开源库叫做Sophus/sophus，封装好了前三个变换。
[库分析  Sophus/sophus ](https://blog.csdn.net/lancelot_vim/article/details/51706832)


# 算法整体框架
    1. Tracking 跟踪线程，当前图像与当前关键帧匹配，获取姿态变换；
    2. 深度图估计线程，    创建新的关键帧/优化当前关键帧，更新关键帧数据库
                                           创建新的关键帧： 传播深度信息到新的关键帧，正则化深度图
                                           优化当前关键帧：近似为小基线长度双目，概率卡尔曼滤波优化更新，正则化深度图
    3. 全局地图优化，        关键帧加入当地图，从地图中匹配最相似的关键帧，估计sim3位姿变换

================================================
## 【1】 Tracking 跟踪 运动变换矩阵求解 对极几何 求解 基本矩阵 F 得到Rt矩阵 两组单目相机 2D图像 
     * (随机采样序列 8点法求解)
     *  2D 点对 求 两相机的 旋转和平移矩阵
     * 空间点 P  两相机 像素点对  p1  p2 两相机 归一化平面上的点对 x1 x2 与P点对应
     * 相机内参数 K  两镜头旋转平移矩阵  R t 或者 变换矩阵 T
     *  p1 = KP  (世界坐标系)     p2 = K( RP + t)  = KTP
     *  而 x1 =  K逆* p1  x2 =  K逆* p2  相机坐标系下 归一化平面上的点     x1= (px -cx)/fx    x2= (py -cy)/fy

> *所以  x1 = P  得到   x2 =  R * x1  + t*

     *  t 外积 x2  = t 外积 R * x1 +  t 外积 t  =  t 外积 R * x1 ； 
        t外积t =0 sin(cet) =0 垂线段投影 方向垂直两个向量
     *  x2转置 *  t 外积  x2 = x2转置 * t 外积 R  x1   = 0 ；
        因为  t 外积  x2 得到的向量垂直 t 也垂直 x2
     *   x2转置 * t 外积 R  x1   = x2转置 * E * x1 =  0 ； 

> 得到       x2转置 * E * x1 =  0 ， E = t 外积 R  为本质矩阵

    * p2转置 * K 转置逆 * t 外积 R * K逆 * p1   = p2转置 * F * p1 =  0 ； 
> 进一步得到  p2转置 * F * p1 =  0 ,     F = K 转置逆 * t 外积 R * K逆 = K 转置逆 * E * K逆 为基础矩阵

    * x2转置 * E * x1 =  0  
    * x1 x2  为 由 像素坐标转化的归一化坐标
    * x1 =  K逆* p1  
    * x2 =  K逆* p2
    * p1 、 p2 为匹配点
    * 一个点对一个约束 ，8点法  可以算出 E的各个元素
    * 再 SVD奇异值分解 E 得到 R t

 =========================
## 【2】深度估计 Depth Estimate 沿极线搜索得到匹配点对，根据运动矩阵得深度
    p2'  =  K × P2 = 
            fx  0  ux       X2      fx * X2 + ux * Z2
            0   fy  uy   ×  Y2    = fy * Y2 + uy * Z2
            0   0   1       Z2      Z2
    P2 = K逆 *  p2'        

    p2 =    u                   fx/Z2 * X2 + ux
            v  = 1/Z * p2'  =   fy/Z2 * Y2 + uy
            1                   1
    这里的Z2就是 点在投影面p2下的深度
    p2'  = Z2 * p2  = d * p2

> 那么   P2 = K逆 * p2'  = Z * K逆 * p2 = d * K逆 * p2 

     P2为相机视角2下的3D坐标， P1为相机视角1下的3D坐标
     P2 = R * P1 + t 
     P1 = R逆 * (P2 - t)    = 
          R逆 * P2 - R逆*t  =  
          R' * P2 +  t'     = 
          d * R' * K逆 * p2  +  t' 
              R'  = R逆
              t'   = - R逆*t
          
> P1 =    d * R' * x2  +  t'    ,  x2    为相机视角2的归一化平面  x2 =  K逆 * p2

     上式  P1 可由 p1求得部分
     x2   可由 p2求得
     R' 、 t' 可由 R、t求得
     R、t可由第一步 Tracking 跟踪 得到
     
     P1为相机视角1下的3D坐标
     p1' = K * P1 = 
                 fx  0  ux       X1      fx * X1 + ux * Z1
                 0   fy  uy   ×  Y1    = fy * Y1 + uy * Z1
                 0   0   1       Z1      Z1

     p1 =  1/Z1 * p1'              
     [P1]1 =     d * [R']1 * x2  +  [t']1   
     [P1]2 =     d * [R']2 * x2  +  [t']2 
     [P1]3 =     d * [R']3 * x2  +  [t']3 

     [P1]1 /  [P1]3  =  [K逆 * p1]1   
     [P1]2 /  [P1]3  =  [K逆 * p1]2 
 
> 在相机视角2下的像素点p2，如果通过极线搜索匹配找到了一个确定的p1， 这里通过 直接法灰度块匹配

    使用p1的横纵坐标均可求得深度d 
    联立可求解d  

    由于SLAM领域经过长期实践发现取深度倒数在计算机中表示并对其进行估计有着更高的健壮性，
    所以如今的SLAM系统一般使用逆深度对三维世界进行表示。

 
 
