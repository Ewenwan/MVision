# -*- coding: utf-8 -*-
#!/usr/bin/env python3

# 1维 2维 贝叶斯滤波  
# 均匀分布 先验值
# 测量     后验
# 移动     噪声

##########################################################
#### 1维 #####################################

p = [0.2 0.2 0.2 0.2 0.2] # 先验概率分部
world = ['green','red','red','green','green']# 世界标志
# Z = 'red'   # 传感器 感知的颜色

measurement = ['red','green']# 测量值
motion = [1, 1]# 移动值 为正值 前移  负值  后移

pHit = 0.6  # 红色
pMiss = 0.2 # 其他颜色

# 传感器检测函数
def sense(p,Z):
   q=[]# 新建一个列表
   for i in range(len(p)):
       hit = (p[i] == Z)
       q.append(p[i]*(pHit*hit + (1-hit) * pMis))
   sum1 = sum(q) #求和
   for i in range(len(q)): 
       q[i] = q[i]/sum1#归一化
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
    p = sense(p,measurement[i])#测量
    p = move(p,motion[i]) #移动

print p


########################################################################
#### 2维 #####################


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
