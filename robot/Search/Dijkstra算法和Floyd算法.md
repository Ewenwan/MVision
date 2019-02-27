Dijkstra算法和Floyd算法
======================

# 一. 最短路径问题介绍

> 从图中的某个顶点出发到达另外一个顶点的所经过的边的权重和最小的一条路径，称为最短路径。解决最短路径问题的算法有Dijkstra算法和Floyd算法。

# 二. Dijkstra算法

## (一) 基本思想

> Dijkstra算法（单源点路径算法，要求：图中不存在负权值边），Dijkstra算法使用了广度优先搜索解决赋权有向图或者无向图的单源最短路径问题，算法最终得到一个最短路径树。 Dijkstra(迪杰斯特拉)算法是典型的最短路径路由算法，用于计算一个节点到其他所有节点的最短路径。主要特点是以起始点为中心向外层层扩展，直到扩展到终点为止。Dijkstra算法能得出最短路径的最优解，但由于它遍历计算的节点很多，所以效率低。

## (二)算法流程

> 1. 设置两个集合S和V。其中，S是已求出最短路径的顶点，V是没有求出最短路径的顶点，初始状态时，S中只有节点0———即起点，V中是节点1-n。设置最短路径数组dist[n+1]，dist代表起点0到节点1-n的最短距离，初始状态时，dist[i]为起点0到节点i的距离，当起点0与节点i有边连接时，dist[i]=边的权值，当起点0与节点i没有边连接时，dist[i]=无穷大。

> 2. 从V中寻找与S(S中最后一个节点)距离最短的节点k，将其加入S中，同时，从V中移除k

> 3. 以k为中间点，更新dist中各节点j的距离。如果: 起点0—>j(经过k)的距离 < 起点0—>j(不经过k)的距离即dist[j]，则dist[j]= 0—>j(经过k)的距离 = 0->k的距离即dist[k] + k->j的距离( <k,j>的权值 )

> 重复步骤2和3，直到所有节点都在S中。

## (三) 图解过程

> 求从A到F的最短路径，设置集合S、V和dist，并初始化，如图1所示：

![image](https://github.com/ShaoQiBNU/The-shortest-path/blob/master/images/1.png)

> 遍历集合V中与A直接相邻的顶点，找出当前与A距离最短的顶点。发现： A-->B 6   A-->C 3，于是将C加入S，并将C从V中移除。以C为中间点，更新dist中各节点的距离如下：

```python
节点  经过C的距离   不经过C的距离   dist
 B     3+2=5          6          5
 C       -            -          3
 D     3+3=6          ∞          6
 E     3+4=7          ∞          7
 F       ∞            ∞          ∞
```

![image](https://github.com/ShaoQiBNU/The-shortest-path/blob/master/images/2.png)

> 遍历集合V中与C直接相邻的顶点，找出当前与C距离最短的顶点。发现： C-->B 2   C-->D 3   C-->E 4，于是将B加入S，并将B从V中移除。以B为中间点，更新dist中各节点的距离如下：
```python
节点  经过B的距离   不经过B的距离   dist
 B       -            -          5
 C       -            -          3
 D     6+5=11         6          6
 E       ∞            7          7
 F       ∞            ∞          ∞
```

![image](https://github.com/ShaoQiBNU/The-shortest-path/blob/master/images/3.png)

> 遍历集合V中与B直接相邻的顶点，找出当前与B距离最短的顶点。发现： B-->D 5 ，于是将D加入S，并将D从V中移除。以D为中间点，更新dist中各节点的距离如下：
```python
节点  经过D的距离   不经过D的距离   dist
 B       -            -          5
 C       -            -          3
 D       -            -          6
 E     6+2=8          7          7
 F     6+3=9          ∞          9
```

![image](https://github.com/ShaoQiBNU/The-shortest-path/blob/master/images/4.png)


> 遍历集合V中与D直接相邻的顶点，找出当前与D距离最短的顶点。发现： D-->E 2  D-->F 3，于是将E加入S，并将E从V中移除。以E为中间点，更新dist中各节点的距离如下：
```python
节点  经过E的距离   不经过E的距离   dist
 B       -            -          5
 C       -            -          3
 D       -            -          6
 E       -            -          7
 F     7+5=12         9          9
```

![image](https://github.com/ShaoQiBNU/The-shortest-path/blob/master/images/5.png)

> 遍历集合V中与E直接相邻的顶点，找出当前与E距离最短的顶点。发现： E-->F 5，于是将F加入S，并将F从V中移除。以F为中间点，更新dist中各节点的距离如下：
```python
节点  经过E的距离   不经过E的距离   dist
 B       -            -          5
 C       -            -          3
 D       -            -          6
 E       -            -          7
 F       -            -          9
```

![image](https://github.com/ShaoQiBNU/The-shortest-path/blob/master/images/6.png)

## (四) python实现(有向图)

```python
def minDist(mdist, visit, V):
    minVal = float('inf')
    minInd = -1
    for i in range(V):
        if (not visit[i]) and mdist[i] < minVal :
            minInd = i
            minVal = mdist[i]
    return minInd 

def Dijkstra(graph, V, startP,endP):
    # 初始化 mdist
    mdist=[float('inf') for _ in range(V)]
    # 被访问的点
    visit = [False for _ in range(V)]
    mdist[startP-1] = 0.0 # 起始点设距离为0
    
    # V个顶点需要做V-1次循环
    for i in range(V-1):
        # 更新每次起点到下一点的位置
        u = minDist(mdist, visit, V) 
        visit[u] = True # 位置被访问
        # 循环遍历所以顶点
        for v in range(V):
            if (not visit[v]) and graph[u][v]!=float('inf') and mdist[u] + graph[u][v] < mdist[v]:
                # 更新mdist
                mdist[v] = mdist[u] + graph[u][v] 
    
    # 返回起始点到其他所有点的最近距离,到终点的距离
    return mdist,mdist[endP-1]

if __name__ == '__main__':
    V = int(input("Enter number of vertices: "))

    graph = [[float('inf') for i in range(V)] for j in range(V)]

    for i in range(V):
        graph[i][i] = 0.0

    graph[0][1] = 6
    graph[0][2] = 3
    graph[1][2] = 2
    graph[1][3] = 5
    graph[2][3] = 3
    graph[2][4] = 4
    graph[3][4] = 2
    graph[3][5] = 3
    graph[4][5] = 5

    startP = int(input("起点:"))
    endP = int(input("终点:"))
    print(Dijkstra(graph, V, startP,endP))


#output:
Enter number of vertices: 6
起点:1
终点:5
([0.0, 6.0, 3.0, 6.0, 7.0, 9.0], 7.0)
```


# 三. Floyd算法

## (一) 算法原理

>  Floyd算法是一个经典的**动态规划算法**。用通俗的语言来描述的话，首先我们的目标是寻找从点i到点j的最短路径。从动态规划的角度看问题，我们需要为这个目标重新做一个诠释（这个诠释正是动态规划最富创造力的精华所在），从任意节点i到任意节点j的最短路径不外乎2种可能，1是直接从i到j，2是从i经过若干个节点k到j。所以，我们假设Dis(i,j)为节点u到节点v的最短路径的距离，对于每一个节点k，我们检查Dis(i,k) + Dis(k,j) < Dis(i,j)是否成立，如果成立，证明从i到k再到j的路径比i直接到j的路径短，我们便设置Dis(i,j) = Dis(i,k) + Dis(k,j)，状态转移方程如下：map[i,j]=min{map[i,k]+map[k,j],map[i,j]}，这样一来，当我们遍历完所有节点k，Dis(i,j)中记录的便是i到j的最短路径的距离。

## (二) 算法描述

> 设置矩阵map和path，map为邻接矩阵，path为路径矩阵，初始状态时，当i和j之间有边时，map[i][j]=<i,j>权重，否则map[i][j]=∞；path矩阵初始状态为i和j可直达，path[i][j]=j。

> 对每一对顶点i和j，看看是否存在k，使得 map[i][k] + map[k][j] < map[i][j]，如果有，则更新map[i][j]；同时更新路径矩阵path[i][j]=path[i][k]。

## (三) 图解过程

> 初始化map和path，如图所示

![image](https://github.com/ShaoQiBNU/The-shortest-path/blob/master/images/7.png)

> 以A为中间节点，更新map和path，此时没有更新项。以B为中间节点，更新map和path

![image](https://github.com/ShaoQiBNU/The-shortest-path/blob/master/images/8.png)

> 以C为中间节点，更新map和path

![image](https://github.com/ShaoQiBNU/The-shortest-path/blob/master/images/9.png)

> 以D为中间节点，更新map和path

![image](https://github.com/ShaoQiBNU/The-shortest-path/blob/master/images/10.png)

> 以E为中间节点，更新map和path，此时没有更新项。以F为中间节点，更新map和path，此时没有更新项。

![image](https://github.com/ShaoQiBNU/The-shortest-path/blob/master/images/11.png)

## (四) python代码实现(无向图)

```python
def floyd(arr,n,startP,endP):
    # 中间节点k
    for k in range(n):
        # 起点i
        for i in range(n):
            # 终点j
            for j in range(n):
                # 更新距离
                if arr[i][j]>(arr[i][k]+arr[k][j]) and i !=j:
                    arr[i][j] = arr[i][k] + arr[k][j]
                    #path[i][j] = j
    return arr[startP-1][endP-1],arr

if __name__ == '__main__':
    n = 6
    arr = [[float('inf')] * n for _ in range(n)]
    arr[0][1] = 6
    arr[0][2] = 3
    arr[1][2] = 2
    arr[1][3] = 5
    arr[2][3] = 3
    arr[2][4] = 4
    arr[3][4] = 2
    arr[3][5] = 3
    arr[4][5] = 5

    for i in range(n):
        arr[i][i] = 0.0
        for j in range(n):
            if arr[i][j] != 'inf':
                arr[j][i] = arr[i][j]
    print(floyd(arr,n,3,3))
    
#output
(9, 
[[0.0, 5, 3, 6, 7, 9], 
 [5, 0.0, 2, 5, 6, 8], 
 [3, 2, 0.0, 3, 4, 6], 
 [6, 5, 3, 0.0, 2, 3], 
 [7, 6, 4, 2, 0.0, 5], 
 [9, 8, 6, 3, 5, 0.0]])

```

# 四. A*算法(A star 算法)

https://www.cnblogs.com/zhoug2020/p/3468167.html

http://www.cppblog.com/mythit/archive/2009/04/19/80492.aspx


```python
# -*- coding: utf-8 -*-
import math

# 地图
tm = ['##########',
      '#........#',
      '#S...#...#',
      '#....#...#',
      '#..###....',
      '####...E..',
      '..........']

# 因为python里string不能直接改变某一元素，所以用test_map来存储搜索时的地图
test_map = []


#########################################################
class Node_Elem:
    """
    开放列表和关闭列表的元素类型，parent用来在成功的时候回溯路径
    """

    def __init__(self, parent, x, y, dist):
        self.parent = parent
        self.x = x
        self.y = y
        self.dist = dist


class A_Star:
    """
    A星算法实现类
    """

    # 注意w,h两个参数，如果你修改了地图，需要传入一个正确值或者修改这里的默认参数
    def __init__(self, s_x, s_y, e_x, e_y, w=10, h=7):
        # 开始点坐标(s_x,s_y)
        self.s_x = s_x
        self.s_y = s_y
        # 结束点坐标(e_x,e_y)
        self.e_x = e_x
        self.e_y = e_y

        self.width = w # 矩阵宽度
        self.height = h # 矩阵高度

        self.open = []
        self.close = []
        self.path = []

    # 查找路径的入口函数
    def find_path(self):
        # 构建开始节点
        p = Node_Elem(None, self.s_x, self.s_y, 0.0)
        while True:
            # 扩展F值最小的节点
            self.extend_round(p)
            # 如果开放列表为空，则不存在路径，返回
            if not self.open:
                return
            # 获取F值最小的节点
            idx, p = self.get_best()
            # 找到路径，生成路径，返回
            if self.is_target(p):
                self.make_path(p)
                return
            # 把此节点压入关闭列表，并从开放列表里删除
            self.close.append(p)
            del self.open[idx]

    def make_path(self, p):
        # 从结束点回溯到开始点，开始点的parent == None
        while p:
            self.path.append((p.x, p.y))
            p = p.parent

    def is_target(self, i):
        return i.x == self.e_x and i.y == self.e_y

    def get_best(self):
        best = None
        bv = 1000000  # 如果你修改的地图很大，可能需要修改这个值
        bi = -1
        for idx, i in enumerate(self.open):
            value = self.get_dist(i)  # 获取F值
            if value < bv:  # 比以前的更好，即F值更小
                best = i
                bv = value
                bi = idx
        return bi, best

    def get_dist(self, i):
        # F = G + H
        # G 为已经走过的路径长度， H为估计还要走多远
        # 这个公式就是A*算法的精华了。
        return i.dist + math.sqrt(
            (self.e_x - i.x) ** 2
            + (self.e_y - i.y) ** 2) * 1.2

    def extend_round(self, p):
        # 可以从8个方向走
        # xs = (-1, 0, 1, -1, 1, -1, 0, 1)
        # ys = (-1, -1, -1, 0, 0, 1, 1, 1)
        # 只能走上下左右四个方向
        xs = (0, -1, 1, 0)
        ys = (-1, 0, 0, 1)
        for x, y in zip(xs, ys):
            new_x, new_y = x + p.x, y + p.y
            # 无效或者不可行走区域，则勿略
            if not self.is_valid_coord(new_x, new_y):
                continue
            # 构造新的节点
            node = Node_Elem(p, new_x, new_y, p.dist + self.get_cost(
                p.x, p.y, new_x, new_y))
            # 新节点在关闭列表，则忽略
            if self.node_in_close(node):
                continue
            i = self.node_in_open(node)
            if i != -1:
                # 新节点在开放列表
                if self.open[i].dist > node.dist:
                    # 现在的路径到比以前到这个节点的路径更好~
                    # 则使用现在的路径
                    self.open[i].parent = p
                    self.open[i].dist = node.dist
                continue
            self.open.append(node)

    def get_cost(self, x1, y1, x2, y2):
        """
        上下左右直走，代价为1.0，斜走，代价为1.4
        """
        if x1 == x2 or y1 == y2:
            return 1.0
        return 1.4

    def node_in_close(self, node):
        for i in self.close:
            if node.x == i.x and node.y == i.y:
                return True
        return False

    def node_in_open(self, node):
        for i, n in enumerate(self.open):
            if node.x == n.x and node.y == n.y:
                return i
        return -1

    def is_valid_coord(self, x, y):
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        return test_map[y][x] != '#'

    def get_searched(self):
        l = []
        for i in self.open:
            l.append((i.x, i.y))
        for i in self.close:
            l.append((i.x, i.y))
        return l


#########################################################
def print_test_map():
    """
    打印搜索后的地图   
    """
    for line in test_map:
        print(''.join(line))


def get_start_XY():
    return get_symbol_XY('S')


def get_end_XY():
    return get_symbol_XY('E')


def get_symbol_XY(s):
    for y, line in enumerate(test_map):
        try:
            x = line.index(s)
        except:
            continue
        else:
            break
    return x, y


#########################################################
def mark_path(l):
    mark_symbol(l, '*')


def mark_searched(l):
    mark_symbol(l, ' ')


def mark_symbol(l, s):
    for x, y in l:
        test_map[y][x] = s


def mark_start_end(s_x, s_y, e_x, e_y):
    test_map[s_y][s_x] = 'S'
    test_map[e_y][e_x] = 'E'


def tm_to_test_map():
    for line in tm:
        test_map.append(list(line))


def find_path():
    s_x, s_y = get_start_XY()
    e_x, e_y = get_end_XY()
    a_star = A_Star(s_x, s_y, e_x, e_y)
    a_star.find_path()
    searched = a_star.get_searched()
    path = a_star.path
    # 标记已搜索区域
    mark_searched(searched)
    # 标记路径
    mark_path(path)
    print("path length is %d" % (len(path)))
    print("searched squares count is %d" % (len(searched)))
    # 标记开始、结束点
    mark_start_end(s_x, s_y, e_x, e_y)


if __name__ == "__main__":
    # 把字符串转成列表
    tm_to_test_map()
    find_path()
    print_test_map()


#output:
path length is 12
searched squares count is 26
##########
#   *** .#
#S***#* .#
#    #* .#
#  ###** .
####.. E..
..........

```
