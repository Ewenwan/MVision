# 路径规划
[一本圣经《Planning Algorithms》伊利诺伊大学教授Lavalle于2006年写的这本书](http://planning.cs.uiuc.edu/booka4.pdf)

[Filtering and Planning in Information Spaces](http://msl.cs.uiuc.edu/~lavalle/iros09/paper.pdf)

[关于寻路算法的一些思考](http://blog.jobbole.com/71044/)

[路径规划算法初步认识](http://www.cnblogs.com/shhu1993/p/6942484.html)

[A Literature Review of UAV 3D Path Planning 综述论文](https://github.com/Ewenwan/MVision/blob/master/robot/Search/A%20Literature%20Review%20of%20UAV%203D%20Path%20Planning.pdf)

# 上面那个论文把uav的路径规划分为以下5类:

    sampling-based algorithms  基于采样的算法
    node-based algorithms      基于节点的算法
    mathematical model based algorithms 基于数学模型的算法
    Bio-inspired algorithms             生物启发式算法
    multi-fusion based algorithms       多数据融合算法

# 知乎 知乎移动机器人路径规划
[知乎移动机器人路径规划](https://www.zhihu.com/question/26342064)
    
    比较传统的如栅格法、人工势场法、模糊规则等，以及它们相应的改进算法。
    
    路径规划算法发展的历程:
![](https://images2015.cnblogs.com/blog/542140/201706/542140-20170604231007086-267952544.png)
    
    这幅图上的算法罗列的还是很全面的，体现了各个算法的出生顺序。但是并不能很好的对他们进行一个本质的分类。
    
分类，两大类：

    1. 完备的（complete)，
       (有解是可以求出来的)，主要应用于二维三维的grid（栅格），多维的计算量就大了；
       
    2. 基于采样的（sampling-based）又称为概率完备的，
       (有解不一定能求出来的,可能经过足够多的采样可以得到解，是概率上的可能能得到解)；
#  一、完备的规划算法
  Dijstra算法、A*、Theta*
    
    所谓完备就是要达到一个systematic的标准，
    即：如果在起始点和目标点间有路径解存在那么一定可以得到解，如果得不到解那么一定说明没有解存在。
    这一大类算法在移动机器人领域通常直接在 occupancy grid网格地图 上进行规划（可以简单理解成二值地图的像素矩阵），
    
    以深度优先寻路算法、广度优先寻路算法、Dijkstra(迪杰斯特拉)算法为始祖，
    以A*算法(Dijstra算法上以减少计算量为目的加上了一个启发式代价)最为常用，
    近期的Theta*算法是在A*算法的基础上增加了line-of-sight优化使得规划出来的路径不完全依赖于单步的栅格形状，
    答主以为这个算法意义不大，不就是规划了一条路径再简单平滑了一下么
    
    完备的算法的优势在与它对于解的捕获能力是完全的，但是由此产生的缺点就是算法复杂度较大。
    这种缺点在二维小尺度栅格地图上并不明显，但是在大尺度，尤其是多维度规划问题上，
    比如机械臂、蛇形机器人的规划问题将带来巨大的计算代价。
    这样也直接促使了第二大类算法的产生。
    
# 二、基于采样(概率逼近思想)的规划算法
    这种算法一般是不直接在grid地图进行最小栅格分辨率的规划，
    它们采用在地图上随机撒一定密度的粒子来抽象实际地图辅助规划。
    如PRM算法及其变种就是在原始地图上进行撒点，抽取roadmap在这样一个拓扑地图上进行规划；
    RRT以及其优秀的变种RRT-connect则是在地图上每步随机撒一个点，
    迭代生长树的方式，连接起止点为目的，最后在连接的图上进行规划。
    这些基于采样的算法速度较快，但是生成的路径代价（可理解为长度）较完备的算法高，
    而且会产生“有解求不出”的情况（PRM的逢Narrow space卒的情况）。
    这样的算法一般在高维度的规划问题中广泛运用。
    
# 三、其他规划算法 
    除了这两类之外还有 
    间接的规划算法：Experience-based（Experience Graph经验图算法）算法：基于经验的规划算法，
    这是一种存储之前规划路径，建立知识库，依赖之进行规划的方法，题主有兴趣可以阅读相关文献。
    这种方法牺牲了一定的空间代价达到了速度与完备兼得的优势。
    此外还有基于广义Voronoi图的方法进行的Fast-marching规划，类似dijkstra规划和势场的融合，
    该方法能够完备地规划出位于道路中央，远离障碍物的路径。
    
    APF（人工势场）算法,
    至于D* 、势场法、DWA(动态窗口法)、SR-PRM,
    属于在动态环境下为躲避动态障碍物、考虑机器人动力学模型设计的规划算法。
    最后推荐一本圣经《Planning Algorithms》伊利诺伊大学教授Lavalle于2006年写的这本书，可以网上看看英文原版的
    
     http://planning.cs.uiuc.edu/booka4.pdf

# 1. 迪杰斯特拉 规划路径
# 2. A星算法   规划路径
# 3. 动态规划  搜索路径
