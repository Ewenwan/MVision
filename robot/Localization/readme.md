# 定位 
# 1. 贝叶斯概率滤波  均匀直方图分布 + 后验测量  估计位置

# 2. 卡尔曼滤波定位 

# 3. 粒子滤波定位


# 一. 贝叶斯概率滤波 

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


