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
## 1. 跟踪线程 Tracking Thread
    当前图像与当前关键帧匹配，获取姿态变换,R t；
## 2. 深度图估计线程 Depth Estimation Thread,
    a. 根据极线匹配结果和运动矩阵求得深度；
    b. 创建新的关键帧/优化当前关键帧，更新关键帧数据库；
    c. 创建新的关键帧： 传播深度信息到新的关键帧，正则化深度图；
    d. 优化当前关键帧：近似为小基线长度双目，概率卡尔曼滤波优化更新，正则化深度图；
## 3. 全局地图优化 Map Optimization，      
    关键帧加入当地图，从地图中匹配最相似的关键帧，估计sim3位姿变换，LM优化最小化变换误差.

================================================
## 【1】 Tracking 跟踪 运动变换矩阵求解 对极几何 求解 基本矩阵 F 得到Rt矩阵 两组单目相机 2D图像 
     * (随机采样序列 8点法求解)
     *  2D 点对 求 两相机的 旋转和平移矩阵
     * 空间点 P  两相机 像素点对  p1  p2 两相机 归一化平面上的点对 x1 x2 与P点对应
     * 相机内参数 K  两镜头旋转平移矩阵  R t 或者 变换矩阵 T
     *  p1 = KP  (世界坐标系)     p2 = K( RP + t)  = KTP
     *  而 x1 =  K逆* p1  x2 =  K逆* p2  相机坐标系下 归一化平面上的点     x1= (px -cx)/fx    x2= (py -cy)/fy

> **所以  x1 = P  得到   x2 =  R * x1  + t**

     *  t 外积 x2  = t 外积 R * x1 +  t 外积 t  =  t 外积 R * x1 ； 
        t外积t =0 sin(cet) =0 垂线段投影 方向垂直两个向量
     *  x2转置 *  t 外积  x2 = x2转置 * t 外积 R  x1   = 0 ；
        因为  t 外积  x2 得到的向量垂直 t 也垂直 x2
     *   x2转置 * t 外积 R  x1   = x2转置 * E * x1 =  0 ； 

> **得到       x2转置 * E * x1 =  0 ， E = t 外积 R  为本质矩阵**

    * p2转置 * K 转置逆 * t 外积 R * K逆 * p1   = p2转置 * F * p1 =  0 ； 
> **进一步得到  p2转置 * F * p1 =  0 ,     F = K 转置逆 * t 外积 R * K逆 = K 转置逆 * E * K逆 为基础矩阵**

    * x2转置 * E * x1 =  0  
    * x1 x2  为 由 像素坐标转化的归一化坐标
    * x1 =  K逆* p1  
    * x2 =  K逆* p2
    * p1 、 p2 为匹配点
    * 一个点对一个约束 ，8点法  可以算出 E的各个元素
    * 再 SVD奇异值分解 E 得到 R t
[单应矩阵 基本矩阵 本质矩阵的区别与联系](https://blog.csdn.net/x_r_su/article/details/54813929)


 ====================================================================
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

> **那么   P2 = K逆 * p2'  = Z * K逆 * p2 = d * K逆 * p2**

     P2为相机视角2下的3D坐标， P1为相机视角1下的3D坐标
     P2 = R * P1 + t 
     P1 = R逆 * (P2 - t)    = 
          R逆 * P2 - R逆*t  =  
          R' * P2 +  t'     = 
          d * R' * K逆 * p2  +  t' 
              R'  = R逆
              t'   = - R逆*t
          
> **P1 =    d * R' * x2  +  t'    ,  x2    为相机视角2的归一化平面  x2 =  K逆 * p2**

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
 
> **在相机视角2下的像素点p2，如果通过极线搜索匹配找到了一个确定的p1， 这里通过 直接法灰度块匹配**

    使用p1的横纵坐标均可求得深度d 
    联立可求解d  

    由于SLAM领域经过长期实践发现取深度倒数在计算机中表示并对其进行估计有着更高的健壮性，
    所以如今的SLAM系统一般使用逆深度对三维世界进行表示。
===================================================
# 【3】地图优化 Map Optimization  定义误差函数，最小化误差函数，优化位姿和地图点
    直接法的SLAM中一般采用迭代优化算法求出图像位姿变换，
    此时需要定义误差函数及寻求误差函数对位姿变换的导数，变换到李代数上进行更新后再变换回李群上。
    而求解这个问题的方法就是高斯牛顿迭代法的各种变种。
    
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


# 【4】 李群李代数知识

[李群李代数 反对称矩阵 指数映射 对数 刚体变换群SE3](https://blog.csdn.net/x_r_su/article/details/52749616)

[李群李代数 原版](https://blog.csdn.net/heyijia0327/article/details/50446140)

[李群李代数在计算机视觉中的应用](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6223449)

## 4.1 矢量叉乘 右手定则
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
## 4.2 旋转矩阵 与 SO3李群 so3李代数
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

## 4.4 旋转矩阵的指数映射
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

### 4.42 指数映射的展开  exp(w*t) = I + w*t + (w*t)^2/2! + ... + (w*t)^n/n! + ...
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
           = I + sin(||W||)* w + ((1-cos(||W||))/||W||^2) * w*2
> 这公式就只和W这个三维向量有关了

    t = arccos((trace(R) - 1)/2) ， trace为矩阵的 迹，主对角线元素的和 
                                          r32 - r23
    W向量 = (W1, W2, W3) = 1/(2*sin(t)) * r13 - r31   和 反对称矩阵位置有关系  正-负
                                          r21 - r12
                     0   -W3   W2
    反对称矩阵 w  =  W3    0  -W1
                    -W2   W1  0                                         
> ||W||的计算轻松加随意，三维向量变换成反对称矩阵也是容易，所以整个将三维旋转速度映射到旋转矩阵编程实现是不是也很容易了。        

## 4.5 欧式变换矩阵 T= [ R t]




## 4.6  李群 李代数 c++ 库 sophus 
### sophus 库安装 
    * 本库为旧版本 非模板的版本
    * git clone https://github.com//strasdat/Sophus.git
    * git checkout a621ff   版本
    * 再cmake编译
### 变量简介
    SO(n)   特殊正交群   对应 n*n 的旋转矩阵 R  群(集合) 
    SE(n+1) 特殊欧式群   对应 n*n 的旋转矩阵和 n*1的平移向量 组合成的  变换矩阵T  群(集合)
    so(n)   SO(n)对应的李代数 为 so(n)   n×1列向量 (反对称矩阵对应的向量)使得矩阵 和 代数 一一对应  可以使用代数的更新方法来更新 矩阵
    SO(3)  表示三维空间的 旋转矩阵 集合 R 3×3
    SE(3)  表示三维空间的 变换矩阵 集合 T 4×4
    李代数 so3的本质就是个三维向量，直接Eigen::Vector3d定义。 3个旋转
    李代数 se3的本质就是个六维向量，3个旋转 + 3个平移

    欧拉角定义：
    旋转向量定义的 李群SO(3) Sophus::SO3 SO3_v( 0, 0, M_PI/2 );  // 亦可从旋转向量构造  这里注意，不是旋转向量的三个坐标值，有点像欧拉角构造。
    旋转向量 转 旋转矩阵  Eigen::Matrix3d R = Eigen::AngleAxisd(M_PI/2, Eigen::Vector3d(0,0,1)).toRotationMatrix();
    旋转矩阵定义的 李群SO(3)           Sophus::SO3 SO3_R(R);               // Sophus::SO(3)可以直接从旋转矩阵构造
                     旋转矩阵 转 四元素     Eigen::Quaterniond q(R);            // 或者四元数(从旋转矩阵构造)
    四元素定义的     李群SO(3)   Sophus::SO3 SO3_q( q );
    李代数so3           为李群SO(3)  的对数映射  Eigen::Vector3d so3 = SO3_R.log();

     平移   Eigen::Vector3d t(1,0,0);           // 沿X轴平移1
     从旋转矩阵 和 平移t 构造  SE3      Sophus::SE3 SE3_Rt(R, t);           // 从R,t构造SE(3)
     从四元素     和 平移t 构造  SE3      Sophus::SE3 SE3_qt(q,t);            // 从q,t构造SE(3)
     李代数se(3) 是一个6维向量   为李群SE3 的对数映射
     typedef Eigen::Matrix<double,6,1> Vector6d;// Vector6d指代　Eigen::Matrix<double,6,1>
     Vector6d se3 = SE3_Rt.log();
  
  
