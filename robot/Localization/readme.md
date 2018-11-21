# 定位 
# 1. 蒙特卡洛定位   
    贝叶斯概率滤波  
    均匀直方图分布 + 后验测量  估计位置
    离散数据   多峰值数据（均匀分布）
# 2. 卡尔曼滤波定位                
    高斯分布   状态转移  测量  反馈补偿 协方差矩阵
    连续状态   单峰值数据（高斯分布）
    
[Kalman滤波的公式解析](https://wenku.baidu.com/view/187eecec856a561252d36f5b.html)

[kalman滤波以及EKF 博文参考](https://blog.csdn.net/lilynothing/article/details/66967744)

[机器人学 —— 机器人感知（Kalman Filter）](https://www.cnblogs.com/ironstark/p/5537219.html)

[EKF-SLAM 算法实现，以ArUco码为路标](https://github.com/Ewenwan/aruco_ekf_slam)

# 3. 粒子滤波定位
    连续状态   多峰值数据 
[机器人学 —— 机器人感知（Location）  粒子滤波器 ](https://www.cnblogs.com/ironstark/p/5570071.html)

[[PR-2] PF 粒子滤波/蒙特卡罗定位](https://github.com/Ewenwan/particle_filter_localization)

# 一. 贝叶斯概率滤波  蒙特卡洛滤波

## 1. 均匀概率测试
    机器人在所有位置上所存在的概率相等
    且概率之和为1

    假设 5个格子
    p = [0.2 0.2 0.2 0.2 0.2]
    print p

## 2. 广义均匀分布
     n = 5  # 格子数量 
     p = []
     for i in range(n):
         p.append(1.0/n)
     print p

## 3. 测量后的概率

    由于测量对于不同的颜色特征有不同的测量可信度，
    假设传感器检测出的红色 可信度 0.6
                     蓝色 可信度 0.2
    第2、3个格子为红色，其余为蓝色：

    则测量之后 概率分部为：
    p = [ 0.04 0.12 0.12 0.04 0.04 ] 

### 3.2 求和
    sum(p) = 0.36
### 3.3 分布归一化 p(i)/sum(p)
    [ 1/9 1/3 1/3 1/9 1/9]

## 4.编程实现 原有均匀分布
    在 测量之后 的概率分部
    pHit pMiss 错过/碰上

    p = [0.2 0.2 0.2 0.2 0.2]# 先验概率分部
    pHit = 0.6  # 红色
    pMiss = 0.2 #其他颜色

    p[0] = p[0]*pMiss
    p[3] = p[3]*pHit
    p[4] = p[4]*pHit
    p[3] = p[3]*pMiss
    p[4] = p[4]*pMiss

    print p      # 后验概率分布
    print sum(p) # 概率总和

## 5. 测量函数

    p = [0.2 0.2 0.2 0.2 0.2] # 先验概率分部
    world = ['green','red','red','green','green']
    Z = 'red'   # 传感器 感知的颜色
    pHit = 0.6  # 红色
    pMiss = 0.2 # 其他颜色

    # 传感器检测函数
    def sense(p,Z):
       q=[]#新建一个列表
       for i in range(len(p)):
           hit = (p[i] == Z)
           q.append(p[i]*(pHit*hit + (1-hit)*pMiss)) 
       return q

### 5.2 归一化测量函数

    # 传感器检测函数
    def sense(p,Z):
       q=[]# 新建一个列表
       sum1 = 0
       tem = 0
       for i in range(len(p)):
           hit = (p[i] == Z)
           tem = p[i]*(pHit*hit + (1-hit)*pMis)
           sum1 += tem
           q.append(p[i]*tem)
       # sum1 = sum(q) 
       for i in range(len(q)): 
           q[i] = q[i]/sum1
       return q


## 6. 多个测量值
    p = [0.2 0.2 0.2 0.2 0.2] # 先验概率分部
    world = ['green','red','red','green','green']
    # Z = 'red'   # 传感器 感知的颜色

    measurement = ['red','green']

    pHit = 0.6  # 红色
    pMiss = 0.2 # 其他颜色

    # 传感器检测函数
    def sense(p,Z):
       q=[]# 新建一个列表
       sum1 = 0
       tem = 0
       for i in range(len(p)):
           hit = (p[i] == Z)
           tem = p[i]*(pHit*hit + (1-hit)*pMis)
           sum1 += tem
           q.append(tem)
       # sum1 = sum(q) 
       for i in range(len(q)): 
           q[i] = q[i]/sum1
       return q

    for i in range(len(measurement)):
        p  = sense(p,measurement[i])

    print p


## 7. 精确移动  原有概率 和移动一起变动
    # 移动函数
    def move(p,U):
       q = [] # 新定义一个列表
       for i in range(len(p))：
           q.append(p[ (i-U)%len(p)])# i-U位置上的元素移动到 i位置上
                                     # %len(p) 世界是循环的
       return q

## 8. 非精确移动 
    U = 2  向前移动两格时
    p(xi+2|xi) = 0.8# 移动到指定格的概率最大
    p(xi+1|xi) = 0.1# 指定格值前后的格子 也有少部分概率
    p(xi+3|xi) = 0.1#

    p(xi+U|xi)   = 0.8# 移动到指定格的概率最大
    p(xi+U-1|xi) = 0.1# 指定格值前后的格子 也有少部分概率
    p(xi+U+1|xi) = 0.1#

## 8.2 非精确移动函数
    p = [0.2 0.2 0.2 0.2 0.2] # 先验概率分部
    world = ['green','red','red','green','green']
    pExact     = 0.8 # 走到指定位置的概率
    pOvershot  = 0.1 # 走过一格的概率
    pUndershot = 0.1 #少走一格的概率
    def move(p,U):
       q = []
       for i in range(len(p))：
           s =  pExact * p[ (i-U)%len(p)]      # 按指定位置走过来的
           s += pOvershot  * p[(i-U-1)%len(p)] # 走过一步到这里的
           s += pUndershot * p[(i-U+1)%len(p)] # 少走一步到这里的
           q.append(s)#

       return q

    每走一次都会丢失一些确定信息，极限移动后，变成 均匀分布
    移动两次：
      p = move(p,1)
      p = move(p,1)
      print p

    移动 1000次：
      for i in range(1000):
          p = move(p,1)
      print p

## 9. 测量 移动 测量 移动

    p = [0.2 0.2 0.2 0.2 0.2] # 先验概率分部
    world = ['green','red','red','green','green']
    # Z = 'red'   # 传感器 感知的颜色

    measurement = ['red','green']# 测量值
    motion = [1, 1]# 移动值

    pHit = 0.6  # 红色
    pMiss = 0.2 # 其他颜色

    # 传感器检测函数
    def sense(p,Z):
       q=[]# 新建一个列表
       for i in range(len(p)):
           hit = (p[i] == Z)
           q.append(p[i]*( pHit*hit + (1-hit) * pMis))
       sum1 = sum(q) 
       for i in range(len(q)): 
           q[i] = q[i]/sum1
       return q

    pExact     = 0.8 # 走到指定位置的概率
    pOvershot  = 0.1 # 走过一格的概率
    pUndershot = 0.1 #少走一格的概率

    # 机器人移动函数
    def move(p,U):
       q = []
       for i in range(len(p))：
           s =  pExact * p[ (i-U)%len(p)]      # 按指定位置走过来的
           s += pOvershot  * p[(i-U-1)%len(p)] # 走过一步到这里的
           s += pUndershot * p[(i-U+1)%len(p)] # 少走一步到这里的
           q.append(s)#
       return q

    for i in range(len(measurement)):
        p = sense(p,measurement[i])
        p = move(p,motion[i]) 

    print p


## 10. 二维

    def localize(colors,measurements,motions,sensor_right,p_move):
        ##### 均匀分布 4*5矩阵，每个格子的概率为 1.0/4/5 = 0.05######
        # len(colors) 行数  len(colors[0]) 列数 
        pinit = 1.0 / float(len(colors)) / float(len(colors[0]))
        p = [[pinit for row in range(len(colors[0]))] for col in range(len(colors))]

        sensor_error = 1.0 - sensor_right# 检测错误概率
        p_still = 1.0 - p_move# 保持不动概率 

        ###### 2维移动 全概率公式###################
        def motion_2d(p,motion):
            aux = [[0.0 for row in range(len(p[0]))] for col in range(len(p))] # 移动后的概率分布
            for i in range(len(p)):# 每一行
                for j in range(len(p[i])):# 每一列
                    # motion[0] 表示上下移动（跨行） motion[1] 表示左右移动（跨列）
                    # 由移动过来的 + 当前静止不动的
                    aux[i][j] = p_move * p[(i-motion[0])%len(p)][(j-motion[1])%len(p[i])] + p_still * p[i][j]
            return aux

        ###### 2维 感知 传感 检测 贝叶斯  乘法#####################
        def sense_2d(p,colors,measurement):
            aux = [[0.0 for row in range(len(p[0]))] for col in range(len(p))] # 初始化 概率矩阵
            s = 0.0#和
            # 求后验概率
            for i in range(len(p)):#每一行
                for j in range(len(p[i])):#每一列
                    hit = (measurement == colors[i][j]) # 检测
                    aux[i][j] = p[i][j] * (hit * sensor_right + (1-hit) * sensor_error)
                    s += aux[i][j]
            # 归一化
            for i in range(len(aux)):#每一行
                for j in range(len(aux[i])):#每一列
                    aux[i][j] =  aux[i][j] / s
            return aux

        # >>> Insert your code here <<<
        if len(measurements)!=len(motions):
            print 'error! the length of measurements mush equals to motions'

        for i in range(len(motions)):#每一次移动
            p=motion_2d(p,motions[i])
            p=sense_2d(p,colors,measurements[i])
        return p

    def show(p):
        rows = ['[' + ','.join(map(lambda x: '{0:.5f}'.format(x),r)) + ']' for r in p]
        print '[' + ',\n '.join(rows) + ']'

    #############################################################
    # For the following test case, your output should be 
    # [[0.01105, 0.02464, 0.06799, 0.04472, 0.02465],
    #  [0.00715, 0.01017, 0.08696, 0.07988, 0.00935],
    #  [0.00739, 0.00894, 0.11272, 0.35350, 0.04065],
    #  [0.00910, 0.00715, 0.01434, 0.04313, 0.03642]]
    # (within a tolerance of +/- 0.001 for each entry)

    # 世界 标记点
    colors = [['R','G','G','R','R'],
              ['R','R','G','R','R'],
              ['R','R','G','G','R'],
              ['R','R','R','R','R']]
    # 测量值
    measurements = ['G','G','G','G','G']
    # 移动
    motions = [[0,0],[0,1],[1,0],[1,0],[0,1]]
    # [0,0]  原地不动
    # [0,1]  右移动一位  
    # [0,-1] 左移动一位
    # [1, 0] 下移动一位
    # [-1,0] 上移动一位

    sensor_right = 0.7#传感器测量正确的概率
    p_move = 0.8# 移动指令正确执行的概率
    p = localize(colors,measurements,motions,sensor_right, p_move)
    show(p) # displays your answer
    
# 二、 卡尔曼滤波  定位 连续状态  单峰值数据 
## 1. 高斯分布
    f(x) = 1/sqrt(2*pi*西格玛^2) * exp(-1/2 * (x-u)^2/西格玛^2)
    u:     x的均值
    西格玛: 标准差 x-u 偏离均值的程度
    西格玛大，变量x偏离中心均值的程度大，高斯曲线 矮胖
    西格玛小，变量x偏离中心均值的程度小，高斯曲线 瘦高

    # python 实现
    from math import *
    def gs(mn, sigma2, x):
        return 1.0/sqrt(2.0*pi*sigma2)* exp(-0.5 * (x-mn)**2 / sigma2)

    print gs(10, 4., 10)  中心最大值
## 2. 两个1维高斯分布的 混合分布  测量过程  先验值是高斯分布  测量过程也是高斯 测量结果
    x ：(u, t^2)
    y : (v, r^2)
    z= x+y : (m, q^2 )

    m   = ( r^2 * u + t^2 * v ) / ( t^2 + r^2)
    q^2 = 1 / (  1/t^2 + 1/r^2)   和两个电阻并联后的电阻公式 有点类似

    可以看到 混合后的分布 的均值在 两个均值之间  min(u,v) < m < max(u,v)
    方差 比钱两个都小  q^2 < min(t^2 , r^2) 混合曲线 变得更瘦高

    # python 实现
    from math import *
    def update(mean1,var1,mean2,var2):
        new_mean = (mean1*var2 + mean2*var1)/(var1+var2)
        new_var  = 1.0/(1.0/var1 + 1.0/var2)
        return  [new_mean, new_var]
    print update(10.,8.,13.,2.)

## 3. 运动高斯叠加 先验值是高斯分布  运动过程也是高斯
    x ：(u, t^2)
    y : (v, r^2)
    x -> 运动y ->  z : (v, p^2 )

    v   = u + v
    p^2 = t^2 + r^2

    # python 实现
    from math import *
    def perdect(mean1,var1,mean2,var2):
        def update(mean1,var1,mean2,var2):
        new_mean = mean1 + mean2 
        new_var  = var1  + var2
        return  [new_mean, new_var]
    print update(10.,8.,13.,2.)

## 4. 高斯分布 先验值  高斯分布分量过程  高斯分布运动过程

    from math import *
    # 测量过程  两个 高地分布叠加
    def update(mean1,var1,mean2,var2):
        new_mean = (mean1*var2 + mean2*var1)/(var1+var2)
        new_var  = 1.0/(1.0/var1 + 1.0/var2)
        return  [new_mean, new_var]
    # 运动过程 两个高斯分布 简单线性 叠加
    def perdect(mean1,var1,mean2,var2):
        def update(mean1,var1,mean2,var2):
        new_mean = mean1 + mean2 
        new_var  = var1  + var2
        return  [new_mean, new_var]
    # 数据
    measurements = [5., 6., 7., 9., 10.] # 测量过程 均值
    motion = [1., 1., 2., 1., 1.]        # 运动过程 均值
    measurement_sig = 4.                 # 测量过程 方差
    motion_sig = 2.                      # 运动过程 方差
    mu = 0.                              # 系统初始 
    sig = 10000.                         # 系统初始方差

    for i in range(len(measurements)):
        [mu, sig] = update(mu, sig, measurements[i], measurement_sig)
        # print 'update: ', [mu, sig]
        [mu, sig] = predict(mu, sig, motion[i], motion_sig)
        # print 'predict: ', [mu, sig]
    print [mu, sig]
## 5. 多维高斯分布
    均值        u = [u1;u2;...;un]
    协方差矩阵   P = [n, n] n*n的矩阵 
    二维高斯分布 类似山峰
###  5.1 一维高斯分布
![](https://github.com/Ewenwan/Mathematics/blob/master/pic/34.png)

> 均值u, 标准差 西格玛，方差 西格玛平方

![](https://github.com/Ewenwan/Mathematics/blob/master/pic/31.png)

    均值u 决定了 曲线在 坐标轴上的位置, 方差 决定了曲线的形状，
    方差越大，数据间差异越大，分布越广泛，曲线矮胖，
    反之，数据集中分布，曲线瘦高。

### 5.2 多维高斯分布
![](https://github.com/Ewenwan/Mathematics/blob/master/pic/35.png)

> 均值u 为n * 1的向量，n为维度数, 一维的方差 变成了多维度向量之间的协方差矩阵,

    协方差求逆，代替了 一维 除以方差， 
    多维时 x-u 为矩阵形式(其实为向量)，一维时直接平方即可，
    多维时，需要 矩阵转置 * 矩阵。

![](https://github.com/Ewenwan/Mathematics/blob/master/pic/32.png)

> 对应的协方差矩阵如下：

![](https://github.com/Ewenwan/Mathematics/blob/master/pic/33.png)

    协方差矩阵的主对角线，是各个变量本身的 方差，
    其余反应 各个变量之间的相关关系, 
    就上面的二元高斯分布而言，
    协方差越大，图像越扁，
    也就是说两个维度之间越有联系。
## 6. 卡尔曼滤波

        # 运动预测 prediction
        x = (F * x) + u # 状态转移
        P = F * P * F.transpose()# 状态协方差传递
        
        # 测量过程 后验 补偿 measurement update
        Z = matrix([measurements[n]])# 测量值
        y = Z.transpose() - (H * x)  # 误差
        S = H * P * H.transpose() + R#
        K = P * H.transpose() * S.inverse()# 卡尔曼增益
        x = x + (K * y)# 状态值 补偿
        P = (I - (K * H)) * P# 状态协方差矩阵更新




# 三、  粒子滤波定位 连续状态   多峰值数据
[粒子滤波](http://blog.csdn.net/heyijia0327/article/details/40899819)

[粒子滤波通俗解释](https://blog.csdn.net/x_r_su/article/details/53083438)

[Particle Filter Tutorial 粒子滤波：从推导到应用（一）](https://blog.csdn.net/heyijia0327/article/details/40899819)

[粒子滤波pdf](https://github.com/Ewenwan/Mathematics/blob/master/pdf/%E7%B2%92%E5%AD%90%E6%BB%A4%E6%B3%A2.pdf)

[维基百科](https://en.wikipedia.org/wiki/Particle_filter)

## 3.1 贝叶斯滤波
### 3.1.1 假设一个系统，我们知道他的状态方程xk 和 测量方程 yk 如下：

      xk = fk(xk-1,vk), 如 xk = xk-1 / 2 + 25 * xk-1 / ( 1 + xk-1 * xk-1) + 8 * cos(1.2*(k-1)) + vk 
      yk = hk(xk, nk),  如 yk = xk * xk / 20 + nk 

      x 为 系统的状态， y为对系统状态x的测量值，
      f是系统状态转移函数，h为系统测量函数，
      v是系统过程噪声， n是系统测量噪声

      从贝叶斯理论的观点来看，状态估计问题（目标跟踪、信号滤波）
      就是根据之前一系列的已有数据 y1：yk（后验知识）递推的计算出当前状态 xk的可信度，
      这个可信度就是概率公式 
      p(xk|y1:yk)，
      它需要通过预测和更新两个步奏来递推的计算。
### 3.1.2 预测过程  先验概率
      通过已有的先验知识对未来的状态进行猜测，即
      先验概率  p( x(k)|x(k-1) )

### 3.1.3 更新过程 后验概率
      利用最新的测量值对先验概率密度进行修正，
      得到后验概率密度，也就是对之前的猜测进行修正。
### 3.1.4 假设系统的状态转移服从一阶马尔科夫模型
      当前时刻的状态x(k)只与上一个时刻的状态x(k-1)有关
      例如 掷筛子，前进几步
      
      同时，假设k时刻测量到的数据y(k)只与当前的状态x(k)有关，
### 3.1.5 推导
      已知：
            已知k-1时刻的概率密度函数 , p(xk-1|y1:yk-1)
      预测：
           p(xk|y1:yk-1) = 积分(p(xk,xk-1|y1:yk-1)*dxk-1)
                         = 积分( p(xk|xk-1,y1:yk-1) * p(xk-1|y1:yk-1) * dxk-1 )
                           (一阶马尔科夫过程的假设，状态x(k)只由x(k-1)决定)
                         = 积分( p(xk|xk-1) * p(xk-1|y1:yk-1) * dxk-1 )
           要采样x(k)，直接采样一个过程噪声，再叠加上 f(x(k-1)) 这个常数就行了。
           
      更新：
          p(xk|y1:yk) = p(yk|xk,y1:yk-1) * p(xk|y1:yk-1)/ p(yk|y1:yk-1)
          
 ## 3.2 蒙特卡洛采样     
      蒙特卡洛采样的思想就是用平均值来代替积分
      抛硬币的例子一样，抛的次数足够多就可以用来估计正面朝上或反面朝上的概率了。
      
      其实就是想知道当前状态的期望值：
      E(f) = 1/N * sum(f(xi)) 
      就是用这些采样的粒子的状态值直接平均就得到了期望值，
      也就是滤波后的值，这里的 f(x) 就是每个粒子的状态函数。
      这就是粒子滤波了，只要从后验概率中采样很多粒子，用它们的状态求平均就得到了滤波结果。
      
       思路看似简单，但是要命的是，后验概率不知道啊，怎么从后验概率分布中采样！
       所以这样直接去应用是行不通的，这时候得引入重要性采样这个方法来解决这个问题。
       
## 3.3 重要性采样    
      无法从目标分布中采样，
      就从一个已知的可以采样的分布里去采样如 q(x|y)，
      这样上面的求期望问题就变成了：  
      E(f) = Wi‘ * sum(f(xi)) 
      Wi’ =  Wi/sum(Wi) 归一化的权重
      而是一种加权和的形式。
      不同的粒子都有它们相应的权重，如果粒子权重大，说明信任该粒子比较多。

      到这里已经解决了不能从后验概率直接采样的问题，
      但是上面这种每个粒子的权重都直接计算的方法，效率低。

      最佳的形式是能够以递推的方式去计算权重，这就是所谓的序贯重要性采样（SIS），粒子滤波的原型。

## 3.4 序贯重要性采样（SIS）  Sequential Importance Sampling (SIS) Filter 
      1、 采样
      2、 递推计算各粒子权重
      3、 粒子权值归一化
      4、 对每个粒子的状态进行加权去估计目标的状态了

## 3.5 重采样
      在应用SIS 滤波的过程中，存在一个退化的问题。
      就是经过几次迭代以后，很多粒子的权重都变得很小，可以忽略了，只有少数粒子的权重比较大。
      并且粒子权值的方差随着时间增大，状态空间中的有效粒子数较少。
      随着无效采样粒子数目的增加，
      使得大量的计算浪费在对估计后验滤波概率分布几乎不起作用的粒子上，使得估计性能下降，
      克服序贯重要性采样算法权值退化现象最直接的方法是增加粒子数，
      而这会造成计算量的相应增加，影响计算的实时性。
      因此，一般采用以下两种途径：
           1)选择合适的重要性概率密度函数；
           2)在序贯重要性采样之后，采用重采样方法。
           
      重采样的思路是：
          既然那些权重小的不起作用了，那就不要了。
          要保持粒子数目不变，得用一些新的粒子来取代它们。
          找新粒子最简单的方法就是将权重大的粒子多复制几个出来，至于复制几个？
          那就在权重大的粒子里面让它们根据自己权重所占的比例去分配，
          也就是老大分身分得最多，
          老二分得次多，以此类推。
          
          类似遗传算法的 淘汰 和 变异产生新个体
          
      假设有3个粒子，在第k时刻的时候，他们的权重分别是0.1, 0.1 ,0.8, 
      然后计算他们的概率累计和(matlab 中为cumsum() )得到： [0.1, 0.2, 1]。
      接着，我们用服从[0,1]之间的均匀分布随机采样3个值，
      假设为0.15 , 0.38 和 0.54。
      也就是说，第二个粒子复制一次，第三个粒子复制两次。 
      
### 3.6 基本粒子滤波算法 
      1、 粒子采样初始化，均匀采样/高斯分布采样
      2、 重要性采样，递推计算各粒子权重，并归一化 粒子权值
      3、 对权重值较小的粒子进行更新，重采样，用权重大的复制替换，(可能会逐渐丢失个体特性，可尝试使用正则粒子滤波，遗传算法，会使用交叉变异)
      4、 对每个粒子的状态进行加权去估计目标的状态 

      通俗解释：
      1、初始化阶段——计算目标特征，比如人体跟踪，就是人体区域的颜色直方图等，n*1的向量；
      2、搜索化阶段——放警犬(采样大量粒子，用来发现目标)，可以 均匀的放置/按目标中心高斯分布放置；
                  ——狗鼻子发现目标，按照相似度信息，计算警犬距离目标的权重，并归一化；
      3、决策化阶段——收集信息，综合信息，每条小狗有一个位置信息和一个权重信息，我们进行 加权求和得到目标坐标；
      4、重采样阶段——去掉一些跑偏的警犬，再放入一些警犬，根据权重信息，将权重低的警犬调回来，重新放置在权重高的地方；
                     根据重要性重新放狗 
      2-> 3-> 4-> 2

      思考1：

      粒子滤波的核心思想是随机采样+重要性重采样。
      既然我不知道目标在哪里，那我就随机的撒粒子吧。
      撒完粒子后，根据特征相似度计算每个粒子的重要性，然后在重要的地方多撒粒子，不重要的地方少撒粒子。
      根据 粒子重要性 和粒子的信息，加权求和得到目标物体。
      所以说粒子滤波较之蒙特卡洛滤波，计算量较小。
      这个思想和RANSAC（随机采样序列一致性）算法真是不谋而合。
      RANSAC的思想也是(比如用在最简单的直线拟合上)，既然我不知道直线方程是什么，
      那我就随机的取两个点先算个直线出来，然后再看有多少点符合我的这条直线（inline 内点数量）。
      哪条直线能获得最多的点的支持，哪条直线就是目标直线。想
      法非常简单，但效果很好。

      思考2：
      感觉粒子滤波和遗传算法真是像极了。
      同时，如果你觉得这种用很多粒子来计算的方式效率低，
      在工程应用中不好接受，推荐看看无味卡尔曼滤波（UKF）,
      他是有选择的产生粒子，而不是盲目的随机产生。


### 3.7 SIR粒子滤波的应用     
      %% SIR粒子滤波的应用，算法流程参见博客http://blog.csdn.net/heyijia0327/article/details/40899819  
      clear all  
      close all  
      clc  
      %% initialize the variables  
      x = 0.1; % 系统初始状态 initial actual state  
      x_N = 1; % 系统过程噪声的协方差  (由于是一维的，这里就是方差)    vk
      x_R = 1; % 测量的协方差                                       nk
      T = 75;  % 共进行75次  
      N = 100; % 粒子数，越大效果越好，计算量也越大  

      %initilize our initial, prior particle distribution as a gaussian around  
      %the true initial value  

      V = 2;    % 初始分布的方差  
      x_P = []; % 粒子  100个
      % 用一个高斯分布随机的产生初始的粒子  
      for i = 1:N  
          x_P(i) = x + sqrt(V) * randn; % 生成初始值 
      end  
       %% xk = fk(xk-1,vk), 如 xk = xk-1 / 2 + 25 * xk-1 / ( 1 + xk-1 * xk-1) + 8 * cos(1.2*(k-1)) + vk 
       %% yk = hk(xk, nk),  如 yk = xk * xk / 20 + nk  测量值
      z_out = [x^2 / 20 + sqrt(x_R) * randn];  % 实际测量值  
      x_out = [x];  % 系统the actual output vector for measurement values.  
      x_est = [x];  % 状态估计值time by time output of the particle filters estimate  
      x_est_out = [x_est]; % the vector of particle filter estimates.  

      for t = 1:T %迭代次数 
          x = 0.5*x + 25*x/(1 + x^2) + 8*cos(1.2*(t-1)) +  sqrt(x_N)*randn; % 系统状态 传递
          z = x^2/20 + sqrt(x_R)*randn;  % 测量值
          for i = 1:N %计算 每一个粒子的权重
                % 从先验p(x(k)|x(k-1))中采样  
              x_P_update(i) = 0.5*x_P(i) + 25*x_P(i)/(1 + x_P(i)^2) + 8*cos(1.2*(t-1)) + sqrt(x_N)*randn;  
                % 计算采样粒子的值，为后面根据似然去计算权重做铺垫  
              z_update(i) = x_P_update(i)^2/20;% 采样粒子的 测量值  
                % 对每个粒子计算其权重，这里假设量测噪声是高斯分布。所以 w = p(y|x)对应下面的计算公式  
              P_w(i) = (1/sqrt(2*pi*x_R)) * exp(-(z - z_update(i))^2/(2*x_R));  
          end  
          % 归一化.  
          P_w = P_w./sum(P_w);  

          %% Resampling这里没有用博客里之前说的histc函数，不过目的和效果是一样的  
          for i = 1 : N  
              x_P(i) = x_P_update(find(rand <= cumsum(P_w),1));   % 粒子权重大的将多得到后代  
          end                                                     % find( ,1) 返回第一个 符合前面条件的数的 下标  

          %状态估计，重采样以后，每个粒子的权重都变成了1/N  
          x_est = mean(x_P); % 均值为估计值 

          % Save data in arrays for later plotting  
          x_out = [x_out x]; % 系统状态 
          z_out = [z_out z]; % 测量值 
          x_est_out = [x_est_out x_est];  % 系统粒子滤波估计值

      end  

      t = 0:T;  
      figure(1);  
      clf  
      plot(t, x_out, '.-b', t, x_est_out, '-.r','linewidth',3);  
      set(gca,'FontSize',12); set(gcf,'Color','White');  
      xlabel('time step'); ylabel('flight position');  
      legend('True flight position', 'Particle filter estimate'); 

### pf 实例2
```c
% 参数设置
N = 200;   %粒子总数
Q = 5;      %过程噪声
R = 5;      %测量噪声
T = 10;     %测量时间
theta = pi/T;       %旋转角度
distance = 80/T;    %每次走的距离
WorldSize = 100;    %世界大小
X = zeros(2, T);    %存储系统状态
Z = zeros(2, T);    %存储系统的观测状态
P = zeros(2, N);    %建立粒子群
PCenter = zeros(2, T);  %所有粒子的中心位置
w = zeros(N, 1);         %每个粒子的权重
err = zeros(1,T);     %误差
X(:, 1) = [50; 20];     %初始系统状态
Z(:, 1) = [50; 20] + wgn(2, 1, 10*log10(R));    %初始系统的观测状态

 

%初始化粒子群
for i = 1 : N
    P(:, i) = [WorldSize*rand; WorldSize*rand];
    dist = norm(P(:, i)-Z(:, 1));     %与测量位置相差的距离
    w(i) = (1 / sqrt(R) / sqrt(2 * pi)) * exp(-(dist)^2 / 2 / R);   %求权重
end

PCenter(:, 1) = sum(P, 2) / N;      %所有粒子的几何中心位置

 

%%

err(1) = norm(X(:, 1) - PCenter(:, 1));     %粒子几何中心与系统真实状态的误差

figure(1);
set(gca,'FontSize',12);
hold on
plot(X(1, 1), X(2, 1), 'r.', 'markersize',30)   %系统状态位置
axis([0 100 0 100]);
plot(P(1, :), P(2, :), 'k.', 'markersize',5);   %各个粒子位置
plot(PCenter(1, 1), PCenter(2, 1), 'b.', 'markersize',25); %所有粒子的中心位置
legend('True State', 'Particles', 'The Center of Particles');
title('Initial State');
hold off
%%

%开始运动
for k = 2 : T
       
    %模拟一个弧线运动的状态
    X(:, k) = X(:, k-1) + distance * [(-cos(k * theta)); sin(k * theta)] + wgn(2, 1, 10*log10(Q));     %状态方程
    Z(:, k) = X(:, k) + wgn(2, 1, 10*log10(R));     %观测方程 
   
    %粒子滤波
    %预测
    for i = 1 : N
        P(:, i) = P(:, i) + distance * [-cos(k * theta); sin(k * theta)] + wgn(2, 1, 10*log10(Q));
        dist = norm(P(:, i)-Z(:, k));     %与测量位置相差的距离
        w(i) = (1 / sqrt(R) / sqrt(2 * pi)) * exp(-(dist)^2 / 2 / R);   %求权重
    end

%归一化权重
    wsum = sum(w);
    for i = 1 : N
        w(i) = w(i) / wsum;
    end
   
    %重采样（更新）
    for i = 1 : N
        wmax = 2 * max(w) * rand;  %另一种重采样规则
        index = randi(N, 1);
        while(wmax > w(index))
            wmax = wmax - w(index);
            index = index + 1;
            if index > N
                index = 1;
            end          
        end
        P(:, i) = P(:, index);     %得到新粒子
    end
   
    PCenter(:, k) = sum(P, 2) / N;      %所有粒子的中心位置
   
    %计算误差
    err(k) = norm(X(:, k) - PCenter(:, k));     %粒子几何中心与系统真实状态的误差
   
    figure(2);
    set(gca,'FontSize',12);
    clf;
    hold on
    plot(X(1, k), X(2, k), 'r.', 'markersize',50);  %系统状态位置
    axis([0 100 0 100]);
    plot(P(1, :), P(2, :), 'k.', 'markersize',5);   %各个粒子位置
    plot(PCenter(1, k), PCenter(2, k), 'b.', 'markersize',25); %所有粒子的中心位置
    legend('True State', 'Particle', 'The Center of Particles');
    hold off
    pause(0.1);
end

 

%%

figure(3);
set(gca,'FontSize',12);
plot(X(1,:), X(2,:), 'r', Z(1,:), Z(2,:), 'g', PCenter(1,:), PCenter(2,:), 'b-');
axis([0 100 0 100]);
legend('True State', 'Measurement', 'Particle Filter');
xlabel('x', 'FontSize', 20); ylabel('y', 'FontSize', 20);
%%

figure(4);
set(gca,'FontSize',12);
plot(err,'.-');
xlabel('t', 'FontSize', 20);
title('The err');

```
## pf 实例3
```c
function [] = particle_filter_localization()
%PARTICLE_FILTER_LOCALIZATION Summary of this function goes here
%   Detailed explanation goes here

% -------------------------------------------------------------------------
% TASK for particle filter localization
% for robotic class in 2018 of ZJU

% Preparartion: 
% 1. you need to know how to code and debug in matlab
% 2. understand the theory of Monte Carlo

% Then complete the code by YOURSELF!
% -------------------------------------------------------------------------

close all;
clear all;

disp('Particle Filter program start!!')

%% initialization

time = 0;
endTime = 60; % second
global dt;
dt = 0.1; % second
 
nSteps = ceil((endTime - time)/dt);
  
localizer.time = [];
localizer.xEst = [];
localizer.xGnd = [];
localizer.xOdom = [];
localizer.z = [];
localizer.PEst=[];
localizer.u=[];

% Estimated State [x y yaw]'ä¼°è?¡å??
xEst=[0 0 0]';
% GroundTruth State????
xGnd = xEst;
% Odometry-only = Dead Reckoning  
xOdom = xGnd;

% Covariance Matrix for predict
Q=diag([0.1 0.1 toRadian(3)]).^2;
Q
% Covariance Matrix for observation
R=diag([1]).^2;% range:meter
R
% Simulation parameter??
global Qsigma
Qsigma=diag([0.1 toRadian(5)]).^2;
global Rsigma
Rsigma=diag([0.1]).^2;

% landmark position
landMarks=[10 0; 10 10; 0 15; -5 20];

% longest observation confined
MAX_RANGE=20;
% Num of particles, initialized
NP=50;
% Used in Resampling Step, a threshold
NTh=NP/2.0;

% particles produced
px=repmat(xEst,1,NP);
% weights of particles produced
pw=zeros(1,NP)+1/NP;


%% Main Loop 

for i=1 : nSteps
    
    time = time + dt;
    u=doControl(time);
    
    % do observation
    [z,xGnd,xOdom,u]=doObservation(xGnd, xOdom, u, landMarks, MAX_RANGE);
    
    for ip=1:NP
        
        % process every particle
        x=px(:,ip);
        w=pw(ip);
        
        % do motion model and random sampling
        x=doMotion(x, u)+sqrt(Q)*randn(3,1);
    
         % calculate inportance weight
        for iz=1:length(z(:,1))
            pz=norm(x(1:2)'-z(iz,2:3));
            dz=pz-z(iz,1);
            w=w*Gaussian(dz,0,sqrt(R));
        end
        px(:,ip)=x;
        pw(ip)=w;
        
    end
    
    pw=Normalization(pw,NP);
    [px,pw]=ResamplingStep(px,pw,NTh,NP);
    xEst=px*pw';
    
    % Simulation Result
    localizer.time=[localizer.time; time];
    localizer.xGnd=[localizer.xGnd; xGnd'];
    localizer.xOdom=[localizer.xOdom; xOdom'];
    localizer.xEst=[localizer.xEst;xEst'];
    localizer.u=[localizer.u; u'];
    
    % Animation (remove some flames)
    if rem(i,10)==0 
        hold off;
        arrow=0.5;
        for ip=1:NP
            quiver(px(1,ip),px(2,ip),arrow*cos(px(3,ip)),arrow*sin(px(3,ip)),'ok');hold on;
        end
        plot(localizer.xGnd(:,1),localizer.xGnd(:,2),'.b');hold on;
        plot(landMarks(:,1),landMarks(:,2),'pk','MarkerSize',10);hold on;
        if~isempty(z)
            for iz=1:length(z(:,1))
                ray=[xGnd(1:2)';z(iz,2:3)];
                plot(ray(:,1),ray(:,2),'-r');hold on;
            end
        end
        plot(localizer.xOdom(:,1),localizer.xOdom(:,2),'.k');hold on;
        plot(localizer.xEst(:,1),localizer.xEst(:,2),'.r');hold on;
        axis equal;
        grid on;
        drawnow;
    end
    
end

% draw the final results of localizer, compared to odometry & ground truth
drawResults(localizer);


end









%% Other functions

% degree to radian
function radian = toRadian(degree)
    radian = degree/180*pi;
end

function []=drawResults(localizer)
%Plot Result
 
    figure(1);
    hold off;
    x=[ localizer.xGnd(:,1:2) localizer.xEst(:,1:2)];
    set(gca, 'fontsize', 12, 'fontname', 'times');
    plot(x(:,1), x(:,2),'-.b','linewidth', 4); hold on;
    plot(x(:,3), x(:,4),'r','linewidth', 4); hold on;
    plot(localizer.xOdom(:,1), localizer.xOdom(:,2),'--k','linewidth', 4); hold on;

    title('Localization Result', 'fontsize', 12, 'fontname', 'times');
    xlabel('X (m)', 'fontsize', 12, 'fontname', 'times');
    ylabel('Y (m)', 'fontsize', 12, 'fontname', 'times');
    legend('Ground Truth','Particle Filter','Odometry Only');
    grid on;
    axis equal;

end

function [ u ] = doControl( time )
%DOCONTROL Summary of this function goes here
%   Detailed explanation goes here

    %Calc Input Parameter
    T=10; % [sec]

    % [V yawrate]
    V=1.0; % [m/s]
    yawrate = 5; % [deg/s]

    u =[ V*(1-exp(-time/T)) toRadian(yawrate)*(1-exp(-time/T))]';


end


%%  you need to complete

%% do Observation model 
function [z, xGnd, xOdom, u] = doObservation(xGnd, xOdom, u, landMarks, MAX_RANGE)
    global Qsigma;
    global Rsigma;
    
    % Gnd Truth and Odometry
    xGnd=doMotion(xGnd, u);% Ground Truth
    u=u+sqrt(Qsigma)*randn(2,1); % add noise randomly
    xOdom=doMotion(xOdom, u); % odometry only
    
    %Simulate Observation
    z=[];
    for iz=1:length(landMarks(:,1))
        d = norm(xGnd(1:2)' - landMarks(iz,:));%d = norm( -landMarks(:,1) )

        if d<MAX_RANGE 
            z=[z;[d+sqrt(Rsigma)*randn(1,1) landMarks(iz,:)]];   % add observation noise randomly
        end
    end
end


%% do Motion Model
function [ x ] = doMotion( x, u)
    global dt;

    Delta = [ dt*cos(x(3)) 0
              dt*sin(x(3)) 0
              0 dt];
    Identity = eye(3)
    x = Identity*x + Delta*u
      
end

%% Gauss function
function g = Gaussian(x,u,sigma)
    g= gaussmf(x,[sigma u])
end

%% Normalization 
function pw=Normalization(pw,NP)
    pwsum = sum(pw)
    for i = 1 : NP
        pw(i) = pw(i) / pwsum 
    end
end

%% Resampling
function [px,pw]=ResamplingStep(px,pw,NTh,NP)
    N_eff= 1/(pw*pw');
    if N_eff < NTh
        ppx=px
    for i=1:NP
      u=rand;
      pw_sum=0;
      for j=1:NP
          pw_sum = pw_sum + pw(j);
          if pw_sum >= u
              px(:,i) = ppx(:,j);
              break;
          end
      end
    end
    pw = zeros(1,NP)+1/NP;
    end
end

```
