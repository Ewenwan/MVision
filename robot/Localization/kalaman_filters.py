#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 卡尔曼滤波 定位 连续状态   单峰值数据 
# 蒙特卡洛定位    离散数据   多峰值数据
# 粒子滤波定位    连续状态   多峰值数据
# 一维坐标+速度  x ， Vx  
# Write a function 'kalman_filter' that implements a 
# multi-dimensional Kalman Filter for the example given

#####  单维度高斯分布 就一个横坐标
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
measurements = [5., 6., 7., 9., 10.] # 测量过程 均值   x轴 坐标值
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
 
 

##### 多维高斯分布  #########################
## 一维坐标+速度  x , Vx
# 矩阵运算 类
class matrix:   
    # implements basic operations of a matrix class
	# 类初始化函数 __inin__()#############
    def __init__(self, value):
        self.value = value
        self.dimx = len(value)#行
        self.dimy = len(value[0])#列
        if value == [[]]:
            self.dimx = 0
    # 0矩阵###############################
    def zero(self, dimx, dimy):
        # check if valid dimensions
        if dimx < 1 or dimy < 1:
            raise ValueError, "Invalid size of matrix"
        else:
            self.dimx = dimx
            self.dimy = dimy
            self.value = [[0 for row in range(dimy)] for col in range(dimx)]
    # 单位阵 对角矩阵#####################
    def identity(self, dim):
        # check if valid dimension
        if dim < 1:
            raise ValueError, "Invalid size of matrix"
        else:
            self.dimx = dim
            self.dimy = dim
            self.value = [[0 for row in range(dim)] for col in range(dim)]
            for i in range(dim):
                self.value[i][i] = 1#主对角线上元素为1 其余为0
    # 打印矩阵############################
    def show(self):
        for i in range(self.dimx):#每一行
            print self.value[i]#打印每一行
        print ' '
		
    # 矩阵相加############################
    def __add__(self, other):
        # check if correct dimensions
        if self.dimx != other.dimx or self.dimy != other.dimy:
            raise ValueError, "Matrices must be of equal dimensions to add"
        else:
            # add if correct dimensions
            res = matrix([[]])#结果矩阵
            res.zero(self.dimx, self.dimy)#新建大小一致的0矩阵
            for i in range(self.dimx):#每一行
                for j in range(self.dimy):#每一列
                    res.value[i][j] = self.value[i][j] + other.value[i][j]#对于位置相加后 赋给 新矩阵
            return res#返回 新矩阵
    # 矩阵相减 ###########################
    def __sub__(self, other):
        # check if correct dimensions
        if self.dimx != other.dimx or self.dimy != other.dimy:
            raise ValueError, "Matrices must be of equal dimensions to subtract"
        else:
            # subtract if correct dimensions
            res = matrix([[]])#结果矩阵
            res.zero(self.dimx, self.dimy)#新建大小一致的0矩阵
            for i in range(self.dimx):#每一行
                for j in range(self.dimy):#每一列
                    res.value[i][j] = self.value[i][j] - other.value[i][j]#对于位置相减后 赋给 新矩阵
            return res
    # 矩阵相乘 ###########################
    def __mul__(self, other):
        # check if correct dimensions
        if self.dimy != other.dimx:# 前者的列数 需要等于 后者的 行数
            raise ValueError, "Matrices must be m*n and n*p to multiply"
        else:
            # subtract if correct dimensions
            res = matrix([[]])#结果矩阵
            res.zero(self.dimx, other.dimy)#新建大小一致的0矩阵 [b_x,b_y] * [l_x,l_y] --> [b_x,l_y]
            for i in range(self.dimx):# 前者的每一行
                for j in range(other.dimy):#后者的每一列
                    for k in range(self.dimy):# 每个元素
                        res.value[i][j] += self.value[i][k] * other.value[k][j]# 对于一行*一列 相加  赋给 新矩阵
            return res
    # 矩阵转置 行变列#####################################
    def transpose(self):
        # compute transpose
        res = matrix([[]])#结果矩阵
        res.zero(self.dimy, self.dimx)# 行列数 交换 新建0矩阵
        for i in range(self.dimx):# 原矩阵的每一行
            for j in range(self.dimy):# 每一列
                res.value[j][i] = self.value[i][j]#  交换 行列下标
        return res
    
    # Thanks to Ernesto P. Adorio for use of Cholesky and CholeskyInverse functions
    ###### cholesky分解，则A = R转置 * R   R是上三角阵#############################3
  	## A-¹=（R转置 * R）-¹ = R-¹ （R转置）-¹ =R-¹ *（R-¹）转置 
    def Cholesky(self, ztol=1.0e-5):
        # Computes the upper triangular Cholesky factorization of
        # a positive definite matrix.
        res = matrix([[]])
        res.zero(self.dimx, self.dimx)
        
        for i in range(self.dimx):
            S = sum([(res.value[k][i])**2 for k in range(i)])
            d = self.value[i][i] - S
            if abs(d) < ztol:
                res.value[i][i] = 0.0
            else:
                if d < 0.0:
                    raise ValueError, "Matrix not positive-definite"
                res.value[i][i] = sqrt(d)
            for j in range(i+1, self.dimx):
                S = sum([res.value[k][i] * res.value[k][j] for k in range(self.dimx)])
                if abs(S) < ztol:
                    S = 0.0
                res.value[i][j] = (self.value[i][j] - S)/res.value[i][i]
        return res
    
    def CholeskyInverse(self):
        # Computes inverse of matrix given its Cholesky upper Triangular
        # decomposition of matrix.
        res = matrix([[]])
        res.zero(self.dimx, self.dimx)
        
        # Backward step for inverse.
        for j in reversed(range(self.dimx)):
            tjj = self.value[j][j]
            S = sum([self.value[j][k]*res.value[j][k] for k in range(j+1, self.dimx)])
            res.value[j][j] = 1.0/tjj**2 - S/tjj
            for i in reversed(range(j)):
                res.value[j][i] = res.value[i][j] = -sum([self.value[i][k]*res.value[k][j] for k in range(i+1, self.dimx)])/self.value[i][i]
        return res		
    # 矩阵的逆 ###################################################
    def inverse(self):
        aux = self.Cholesky()
        res = aux.CholeskyInverse()
        return res
    
    def __repr__(self):
        return repr(self.value)


########################################
###### 两维数据 一维坐标+速度  x ， Vx  #########
# Implement the filter function below

def kalman_filter(x, P):
    for n in range(len(measurements)):#对每一个测量数据
        
        # 测量过程 更新 类似后验测量值更新 measurement update
        Z = matrix([[measurements[n]]])  # 实际测量值
        y = Z - (H*x)                    # 类似 反馈 ERROR 误差
        S = H * P * H.transpose() + R    # UN
        K = P * H.transpose() * S.inverse()# 卡尔曼增益
        x = x + K*y     #  更新 系统状态, 原值 + 比例放大*误差
        P = (I - K*H)*P #  更新 系统状态的协方差矩阵
        
        # 运动过程 预测 prediction
        x = (F * x) + u #  状态转移 
		# 变量组X的线性变换， f(X) = AX，
        # 假设X的协方差矩阵为C
        # 则f(X)的协方差为 A * C * A转置
		# (相当于乘以系数的平方， 方差为 误差的平方, (x-均值)^2 , x放大了a倍，则(x-均值)^2放大a^2)
        P = F * P *F.transpose()# 协方差矩阵的传播 
    return x,P

############################################
### use the code below to test your filter!
############################################

measurements = [1, 2, 3]# 测量值 只有一个位置数据

x = matrix([[0.], [0.]])# 初始状态 (位置location 和 速度 velocity) 
P = matrix([[1000., 0.], [0., 1000.]]) # 初始系统状态协方差矩阵 initial uncertainty
u = matrix([[0.], [0.]])        # 系统输入 加速度 没有 external motion

F = matrix([[1., 1.], [0, 1.]]) # 状态转移矩阵 A next state function
#x  = x + vx*1
#vx = vx       匀加速运动
#[x;vx] = [1,1;0,1]*[x,vx]

H = matrix([[1., 0.]])          # 测量变换矩阵 状态到测量值 measurement function
# z = [1 , 0] * [x , vx]  只有一个位置测量值


R = matrix([[1.]])              # 测量过程方差 measurement uncertainty
I = matrix([[1., 0.], [0., 1.]])# 单位阵 identity matrix

print kalman_filter(x, P)
# output should be:
# x: [[3.9996664447958645], [0.9999998335552873]]
# P: [[2.3318904241194827, 0.9991676099921091], [0.9991676099921067, 0.49950058263974184]]


#############################################################################
####### 4维数据  两维围标+速度  x，y，Vx，Vy################
########################################

def filter(x, P):
    for n in range(len(measurements)):
        
        # 运动预测 prediction
        x = (F * x) + u# 状态转移
        P = F * P * F.transpose()# 状态协方差传递
        
        # 测量过程 后验 补偿 measurement update
        Z = matrix([measurements[n]])#测量值
        y = Z.transpose() - (H * x)  #误差
        S = H * P * H.transpose() + R#
        K = P * H.transpose() * S.inverse()#卡尔曼增益
        x = x + (K * y)#状态值 补偿
        P = (I - (K * H)) * P# 状态协方差矩阵更新
    
    print 'x= '
    x.show()
    print 'P= '
    P.show()

########################################

print "### 4-dimensional example ###"

# 这里我们只对 x和y轴坐标值进行测量
measurements = [[5., 10.], [6., 8.], [7., 6.], [8., 4.], [9., 2.], [10., 0.]]# x,y坐标测量值
initial_xy = [4., 12.]#系统初始起点 位置坐标

# measurements = [[1., 4.], [6., 0.], [11., -4.], [16., -8.]]
# initial_xy = [-4., 8.]

# measurements = [[1., 17.], [1., 15.], [1., 13.], [1., 11.]]
# initial_xy = [1., 19.]

dt = 0.1#时间增量
# 系统状态 x =[x，y，vx，vy]
x = matrix([[initial_xy[0]], [initial_xy[1]], [0.], [0.]]) # initial state (location and velocity)
u = matrix([[0.], [0.], [0.], [0.]]) # 匀速运动 系统无输入 external motion

#### DO NOT MODIFY ANYTHING ABOVE HERE ####
#### fill this in, remember to use the matrix() function!: ####
# 系统状态 协方差矩阵   初始速度方差 1000
P = matrix([
[0, 0, 0,     0],
[0, 0, 0,     0],
[0, 0, 1000., 0],
[0, 0, 0,     1000.]])
# initial uncertainty: 0 for positions x and y, 1000 for the two velocities

# 状态转移矩阵  
F = matrix([
[1., 0, dt, 0],
[0, 1., 0, dt],
[0, 0, 1., 0],
[0, 0, 0, 1.]]) 
# x = x + 0*y + dt*vx + 0*vy
# y = 0 + y   + 0*vx  + dt*vy
# vx= 0 + 0   + vx    + 0
# vy= 0 + 0   + 0     + vy
# next state function: generalize the 2d version to 4d

# 系统状态 到 测量值 转换 矩阵  这里我们只对 x和y轴坐标值进行测量
H = matrix([
[1., 0, 0, 0],
[0, 1., 0, 0]]) 
# x = x + 0 + 0 + 0
# y = 0 + y + 0 + 0
# measurement function: reflect the fact that we observe x and y but not the two velocities

# 测量过程协方差矩阵   只对 x和y轴坐标值进行测量
R = matrix([
[0.1, 0],
[0, 0.1]])
# measurement uncertainty: use 2x2 matrix with 0.1 as main diagonal

# 系统状态独立性 单位矩阵
I = matrix([
[1., 0, 0, 0],
[0, 1., 0, 0],
[0, 0, 1., 0],
[0, 0, 0, 1.]])  # 4d identity matrix

###### DO NOT MODIFY ANYTHING HERE #######
filter(x, P)
