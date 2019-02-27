# BFS-DFS 广度和深度优先搜索

## BFS 宽度优先搜索算法（又称广度优先搜索）
* [示例1. 赛码网小赛旅游](#示例1-赛码网小赛旅游)

* [示例2. 走迷宫](#示例2-走迷宫)

* [示例3. hero--拯救公主](#示例3-hero--拯救公主)



&emsp;&emsp;宽度优先搜索算法（又称广度优先搜索）是最简便的图的搜索算法之一，这一算法也是很多重要的图的算法的原型。Dijkstra单源最短路径算法和Prim最小生成树算法都采用了和宽度优先搜索类似的思想。其别名又叫BFS，属于一种盲目搜寻法，目的是系统地展开并检查图中的所有节点，以找寻结果。换句话说，它并不考虑结果的可能位置，彻底地搜索整张图，直到找到结果为止。

&emsp;&emsp;广度优先搜索是一种分层的查找过程，每向前走一步可能访问一批顶点，不像深度优先搜索那样有回退的情况，因此它不是一个递归的算法，为了实现逐层的访问，算法必须借助一个先进先出的辅助**队列**并且以非递归的形式来实现。

**算法的基本思路：**

&emsp;&emsp;我们采用示例图来说明这个过程，在搜索的过程中，初始所有节点是白色（代表了所有点都还没开始搜索），把起点V0标志成灰色（表示即将辐射V0），下一步搜索的时候，我们把所有的灰色节点访问一次，然后将其变成黑色（表示已经被辐射过了），进而再将他们所能到达的节点标志成灰色（因为那些节点是下一步搜索的目标点了），当访问到V1节点的时候，它的下一个节点应该是V0和V4，但是V0已经在前面被染成黑色了，所以不会将它染灰色。这样持续下去，直到目标节点V6被染灰色，说明了下一步就到终点了，没必要再搜索（染色）其他节点了，此时可以结束搜索了，整个搜索就结束了。然后根据搜索过程，反过来把最短路径找出来，图中把最终路径上的节点标志成绿色。  

<p ><img alt="" src="http://my.csdn.net/uploads/201204/30/1335725797_1963.png">初始全部都是白色（未访问</span></p>   
<p ><img alt="" src="http://my.csdn.net/uploads/201204/30/1335725807_5317.png">即将搜索起点</p>    
<p ><img alt="" src="http://my.csdn.net/uploads/201204/30/1335725819_1561.png">已搜索V0，即将搜索V1、V2、V3</p>   
<p ><img alt="" src="http://my.csdn.net/uploads/201204/30/1335725831_7574.png">……终点V6被染灰色，终止</span></p>   
<p><img alt="" src="http://my.csdn.net/uploads/201204/30/1335725843_7283.png">找到最短路径</p>    

**广度优先搜索流程图**

<img alt="" src="http://my.csdn.net/uploads/201204/30/1335725885_9403.png">


**1. 无向图的广度优先搜索**

<img src="https://github.com/wangkuiwu/datastructs_and_algorithm/blob/master/pictures/graph/iterator/05.jpg?raw=true" alt="">  

```
第1步：访问A。 
第2步：依次访问C,D,F。 
    在访问了A之后，接下来访问A的邻接点。前面已经说过，在本文实现中，顶点ABCDEFG按照顺序存储的，C在"D和F"的前面，因此，先访问C。再访问完C之后，再依次访问D,F。 
第3步：依次访问B,G。 
    在第2步访问完C,D,F之后，再依次访问它们的邻接点。首先访问C的邻接点B，再访问F的邻接点G。 
第4步：访问E。 
    在第3步访问完B,G之后，再依次访问它们的邻接点。只有G有邻接点E，因此访问G的邻接点E。

因此访问顺序是：A -> C -> D -> F -> B -> G -> E
```

**2. 有向图的广度优先搜索**

<img src="https://github.com/wangkuiwu/datastructs_and_algorithm/blob/master/pictures/graph/iterator/06.jpg?raw=true" alt="">

```
第1步：访问A。 
第2步：访问B。 
第3步：依次访问C,E,F。 
    在访问了B之后，接下来访问B的出边的另一个顶点，即C,E,F。前面已经说过，在本文实现中，顶点ABCDEFG按照顺序存储的，因此会先访问C，再依次访问E,F。 
第4步：依次访问D,G。 
    在访问完C,E,F之后，再依次访问它们的出边的另一个顶点。还是按照C,E,F的顺序访问，C的已经全部访问过了，那么就只剩下E,F；先访问E的邻接点D，再访问F的邻接点G。

因此访问顺序是：A -> B -> C -> E -> F -> D -> G
```

-----------------------------

### 示例1. [赛码网：小赛旅游](http://exercise.acmcoder.com/online/online_judge_ques?ques_id=2267&konwledgeId=139)

**题目描述**

小赛很想到外面的世界看看，于是收拾行装准备旅行。背了一个大竹筐，竹筐里装满了路上吃的，这些吃的够它走N公里。为了规划路线，它查看了地图，沿途中有若干个村庄，在这些村庄它都可以补充食物。但每次补充食物都需要花费时间，在它竹筐的食物足够可以走到下一个村庄的时候它就不用补充，这样背起来不累而且不花费时间。地图上可以看到村庄之间的距离，现在它要规划一下它的路线，确定在哪些村庄补充食物可以使沿途补充食物的次数最少。你能帮帮小赛吗？  
输入描述：   
```  
第一行有两个数字，第一个数字为竹筐装满可以走的公里数，即N值；第二个数字为起点到终点之间的村庄个数。  
第二行为起点和村庄、村庄之间、村庄和终点之间的距离。且范围不能超过一个int型表达的范围。    
 示例：
 7 4    
 5  6  3  2  2  
````
输出描述：   
```
程序输出为至少需要补充食物的次数。   
示例：
2
```

```python
'''
判断每段距离与装行李的重量N的大小，当dis[i]<N时，走不完该段路程；
当N-dis[i] >= dis[i+1]即食物完全满足两段路的需求，
 将N-dis[i]重新赋给N继续走下一段路；
否则就没走一段路到达村庄后补给食物即装满N。
'''
num = list(map(int, raw_input().split()))
dis = list(map(int, raw_input().split()))
N = num[0]
m = num[1]
count = 0
for i in range(m):
    if dis[i] > num[0]:
        break
    elif N - dis[i] >= dis[i+1]:
        N = N - dis[i]
    else:
        N = num[0]
        count += 1
print  count
```


### 示例2. 走迷宫

https://github.com/ShaoQiBNU/mazes_BFS

https://github.com/BrickXu/subway   

**问题描述**

输入一组10 x 10的数据，由#和.组成的迷宫，其中#代表墙，.代表通路，入口在第一行第二列，出口在最后一行第九列，从任意一个.都能一步走到上下左右四个方向的.，请求出从入口到出口最短需要几步？   
输入示例：
```
#.########                                   #.########                     
#........#                                   #........#                      
#........#                                   ########.#
#........#                                   #........#
#........#                                   #.########
#........#                                   #........#
#........#                                   ########.#
#........#                                   #........#
#........#                                   #.######.#                                   
########.#                                   ########.#
结果为：16                                    结果为： 30

```

```python
# 因为题意是使用最少的步数走出迷宫，所要可以使用广度优先遍历的方式，每处理完一层说明走了一步，最先到达出口使用的步数最少。

import  numpy as np
def bfs(N,maps,start,end):
    """
    1:已经访问；0: 每访问
    :param N: 矩阵大小
    :param maps: 矩阵
    :param start: 开始点
    :param end: 结束点
    :return: 步数
    """
    # 上下左右四个方向的增量
    dx = [1,-1,0,0]
    dy = [0,0,1,-1]

    # 用于存放节点
    nodes = []
    # 开始的节点(x坐标，y坐标，步数)
    nodes.append((0,1,0))

    # 节点访问列表—记录节点是否被访问
    visitNodes = np.array([[0] * N] * N)
    visitNodes[0][1] = 1

    # bfs过程
    while len(nodes):
        # 上下左右四个方向遍历
        for i in range(4):
            # 从节点列表输出一个节点
            node = nodes[0]
            # 上下左右四个方向遍历
            x = node[0] + dx[i]
            y = node[1] + dy[i]
            # 步数
            step = node[2]
            # 判断是否到达终点
            if x ==9 and y == 8:
                return step+1
            # 判断节点是否符合条件
            if x>=1 and x<=9 and y>=1 and y<=9 and visitNodes[x][y] == 0 and maps[x][y] == 1:
                # 将节点压入节点列表nodes，说明进入下一层，step+1
                nodes.append((x,y,step+1))
                # 访问过该节点
                visitNodes[x][y] = 1
        # 从节点列表移除上一层的节点
        del nodes[0]
     # 没有路径无法走出时，返回0
    return  0



if __name__ == '__main__':
    maps1 = np.array([[0, 2, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
                       , [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
                       , [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
                       , [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
                       , [0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 3, 0]])
    maps2 = np.array([[0, 2, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
                       , [0, 1, 1, 1, 1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
                       , [0, 1, 1, 1, 1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
                       , [0, 1, 1, 1, 1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
                       , [0, 1, 1, 1, 1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 3, 0]])

    res = bfs(10,maps2,2,3)
    print(res)

```

### 示例3. [hero | 拯救公主](https://www.nowcoder.com/practice/661b4d5797f04b13af291befe051d5e9?tpId=3&&tqId=10875&rp=2&ru=/activity/oj&qru=/ta/hackathon/question-ranking)   

**题目描述**  
500年前，nowcoder是我国最卓越的剑客。他英俊潇洒，而且机智过人^_^。 突然有一天，nowcoder 心爱的公主被魔王困在了一个巨大的迷宫中。nowcoder 听说这个消息已经是两天以后了，他知道公主在迷宫中还能坚持T天，他急忙赶到迷宫，开始到处寻找公主的下落。 时间一点一点的过去，nowcoder 还是无法找到公主。最后当他找到公主的时候，美丽的公主已经死了。从此nowcoder 郁郁寡欢，茶饭不思，一年后追随公主而去了。T_T 500年后的今天，nowcoder 托梦给你，希望你帮他判断一下当年他是否有机会在给定的时间内找到公主。 他会为你提供迷宫的地图以及所剩的时间T。请你判断他是否能救出心爱的公主。    
输入描述：   
```
每组测试数据以三个整数N,M,T(00)开头，分别代表迷宫的长和高，以及公主能坚持的天数。
紧接着有M行，N列字符，由"."，"*"，"P"，"S"组成。其中
"." 代表能够行走的空地。
"*" 代表墙壁，redraiment不能从此通过。
"P" 是公主所在的位置。
"S" 是redraiment的起始位置。
每个时间段里redraiment只能选择“上、下、左、右”任意一方向走一步。
输入以0 0 0结束
示例：
4 4 10
....
....
....
S**P
0 0 0
```
输出描述：   
```
如果能在规定时间内救出公主输出“YES”，否则输出“NO”。
示例：
YES
```

```python
def bfs(maps, n, m, t):
    start = ()
    end = ()
    for i in range(0, m):
        for j in range(0, n):
            if maps[i][j] == 'S':
                start = (i, j)
            if maps[i][j] == 'P':
                end = (i, j)
    if len(start) == 0 or len(end)==0:
        return 'NO'

    dx = [1, -1, 0, 0]
    dy = [0, 0, 1, -1]
    nodes_cur = []
    nodes_cur.append(start)
    nodes_next = []
    node_visit = [[0 for _ in range(n)] for _ in range(m) ]
    node_visit[start[0]][start[1]] = 1
    while len(nodes_cur) != 0:
        for i in range(0, 4):
            node = nodes_cur[0]
            x = node[0] + dx[i]
            y = node[1] + dy[i]
            if x == end[0] and y == end[1] :
                return 'YES'
            if x >= 0 and x < m and y >= 0 and y < n and node_visit[x][y] == 0 and maps[x][y] == '.':
                nodes_next.append((x, y))
                node_visit[x][y] = 1
        del (nodes_cur[0])
        if len(nodes_cur) == 0:
            t = t - 1
            if t < 0:
                return 'NO'
            else:
                nodes_cur = nodes_next.copy()
                nodes_next = []
    return 'NO'
if __name__ == '__main__':
    maps = []
    s=input()
    if s == '0 0 0':
        print('NO')
    else:
        n,m,t = map(int,s.split())
        while 1:
            s = input()
            if s == '0 0 0':
                break
            else:
                maps.append(list(s))
        res = bfs(maps, n, m, t)
        print (res )

```


##  深度优先搜索 

&emsp;&emsp;简要来说dfs是对每一个可能的分支路径深入到不能再深入为止，而且每个节点只能访问一次。深度优先搜索的缺点也出来了：**难以寻找最优解**，仅仅只能寻找有解。其优点就是**内存消耗小**。

算法思想：

&emsp;&emsp;假设初始状态是图中所有顶点均未被访问，则从某个顶点v出发，首先访问该顶点，然后依次从它的各个未被访问的邻接点出发深度优先搜索遍历图，直至图中所有和v有路径相通的顶点都被访问到。 若此时尚有其他顶点未被访问到，则另选一个未被访问的顶点作起始点，重复上述过程，直至图中所有顶点都被访问到为止。   
显然，深度优先搜索是一个递归的过程。

**1. 无向图的深度优先搜索**

<img src="https://github.com/wangkuiwu/datastructs_and_algorithm/blob/master/pictures/graph/iterator/02.jpg?raw=true" alt="">  

对上面的图进行深度优先遍历，从顶点A开始。  

```
第1步：访问A。    
第2步：访问(A的邻接点)C。     
  在第1步访问A之后，接下来应该访问的是A的邻接点，即"C,D,F"中的一个。但在本文的实现中，顶点ABCDEFG是按照顺序存储，C在"D和F"的前面，因此，先访问C。   
第3步：访问(C的邻接点)B。    
  在第2步访问C之后，接下来应该访问C的邻接点，即"B和D"中一个(A已经被访问过，就不算在内)。而由于B在D之前，先访问B。    
第4步：访问(C的邻接点)D。    
  在第3步访问了C的邻接点B之后，B没有未被访问的邻接点；因此，返回到访问C的另一个邻接点D。    
第5步：访问(A的邻接点)F。    
  前面已经访问了A，并且访问完了"A的邻接点B的所有邻接点(包括递归的邻接点在内)"；因此，此时返回到访问A的另一个邻接点F。    
第6步：访问(F的邻接点)G。   
第7步：访问(G的邻接点)E。

因此访问顺序是：A -> C -> B -> D -> F -> G -> E
```

**2. 有向图的深度优先搜索**

<img src="https://github.com/wangkuiwu/datastructs_and_algorithm/blob/master/pictures/graph/iterator/04.jpg?raw=true" alt="">  
对上面的图进行深度优先遍历，从顶点A开始。   

```
第1步：访问A。 
第2步：访问B。 
    在访问了A之后，接下来应该访问的是A的出边的另一个顶点，即顶点B。 
第3步：访问C。 
    在访问了B之后，接下来应该访问的是B的出边的另一个顶点，即顶点C,E,F。在本文实现的图中，顶点ABCDEFG按照顺序存储，因此先访问C。 
第4步：访问E。 
    接下来访问C的出边的另一个顶点，即顶点E。 
第5步：访问D。 
    接下来访问E的出边的另一个顶点，即顶点B,D。顶点B已经被访问过，因此访问顶点D。 
第6步：访问F。 
    接下应该回溯"访问A的出边的另一个顶点F"。 
第7步：访问G。

因此访问顺序是：A -> B -> C -> E -> D -> F -> G
```

### 示例1. 城堡问题

**问题描述：**

```
     1   2   3   4   5   6   7  
   #############################
 1 #   |   #   |   #   |   |   #
   #####---#####---#---#####---#
 2 #   #   |   #   #   #   #   #
   #---#####---#####---#####---#
 3 #   |   |   #   #   #   #   #
   #---#########---#####---#---#
 4 #   #   |   |   |   |   #   #
   #############################
           (图 1)

   #  = Wall   
   |  = No wall
   -  = No wall

图1是一个城堡的地形图。请你编写一个程序，计算城堡一共有多少房间，最大的房间有多大。城堡被分割成mn(m≤50，n≤50)个方块，每个方块可以有0~4面墙。 
Input程序从标准输入设备读入数据。第一行是两个整数，分别是南北向、东西向的方块数。在接下来的输入行里，每个方块用一个数字(0≤p≤50)描述。用一个数字表示方块周围的墙，1表示西墙，2表示北墙，4表示东墙，8表示南墙。每个方块用代表其周围墙的数字之和表示。城堡的内墙被计算两次，方块(1,1)的南墙同时也是方块(2,1)的北墙。输入的数据保证城堡至少有两个房间。Output城堡的房间数、城堡中最大房间所包括的方块数。结果显示在标准输出设备上。 
Sample Input:
4  7 
11 6 11 6 3 10 6 
7 9 6 13 5 15 5 
1 10 12 7 13 7 5 
13 11 10 8 10 12 13 
Sample Output
5
9
```

<img src="//img-blog.csdn.net/20180316152743872?watermark/2/text/Ly9ibG9nLmNzZG4ubmV0L0xaSF8xMjM0NQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" alt="">   
<img src="//img-blog.csdn.net/20180316152812703?watermark/2/text/Ly9ibG9nLmNzZG4ubmV0L0xaSF8xMjM0NQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" alt="">  

<img src="//img-blog.csdn.net/20180316152908233?watermark/2/text/Ly9ibG9nLmNzZG4ubmV0L0xaSF8xMjM0NQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" alt="">  

```python
rows, cols = map(int, input().split())
rooms = []
for i in range(rows):
    rooms.append(list(map(int, input().split())))

# 同一房间有相同的color值
color = [[0] * cols for _ in range(rows)]
roomNum = 0 # 房间数量
maxRoomArea = 0 # 房间的方块数

def DFS(i,j):
    global roomNum
    global roomArea
    if color[i][j]!=0:
        return
    roomArea += 1
    color[i][j] = roomNum
    # 向西走
    if rooms[i][j] & 1 == 0:
        DFS(i, j - 1)
    # 向北走
    if rooms[i][j] & 2 == 0:
        DFS(i - 1, j)
    # 向东走
    if rooms[i][j] & 4 == 0:
        DFS(i, j + 1)
    # 向南走
    if rooms[i][j] & 8 == 0:
        DFS(i + 1, j)

for i in range(rows):
    for j in range(cols):
        if color[i][j] == 0:
            roomNum += 1
            roomArea = 0
            DFS(i,j)
            maxRoomArea = max(roomArea,maxRoomArea)
print('房间数量:',roomNum)
print('最大房间的方块数：',maxRoomArea)
print(color)

#output
房间数量: 5
最大房间的方块数： 9
[[1, 1, 2, 2, 3, 3, 3], [1, 1, 1, 2, 3, 4, 3], [1, 1, 1, 5, 3, 5, 3], [1, 5, 5, 5, 5, 5, 3]]
```

