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


# 3. 粒子滤波定位
    连续状态   多峰值数据 

# 一. 贝叶斯概率滤波   蒙特卡洛滤波

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
