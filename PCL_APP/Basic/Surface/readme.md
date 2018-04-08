# 点云曲面
## 滑动最小二乘 表面平滑
    重建算法
    在测量较小的数据时会产生一些误差，这些误差所造成的不规则数据如果直接拿来曲面重建的话，
    会使得重建的曲面不光滑或者有漏洞，可以采用对数据重采样来解决这样问题，
    通过对周围的数据点进行高阶多项式插值来重建表面缺少的部分，
    1）用最小二乘法对点云进行平滑处理
    Moving Least Squares (MLS) surface reconstruction method 
    滑动最小二乘 表面平滑　重建算法

    虽说此类放在了Surface下面，但是通过反复的研究与使用，
    我发现此类并不能输出拟合后的表面，不能生成Mesh或者Triangulations，
    只是将点云进行了MLS的映射，使得输出的点云更加平滑。

    因此，在我看来此类应该放在Filter下。
    通过多次的实验与数据的处理，
    我发现此类主要适用于点云的光顺处理，
    当然输入的点云最好是滤过离群点之后的点集，
    否则将会牺牲表面拟合精度的代价来获得输出点云。

    Pcl::MovingLeastSquares<PointInT, PointOutT> mls
    其中PointInT决定了输入点云类型，
    PointOutT为点云输出类型（
    当法线估计标志位设置为true时，输出向量必须加上normals，这一点我们将会在成员函数里介绍）。
[滑动最小二乘 表面平滑](resampling.cpp)

## 把点云投影到平面上，在平面模型上提取凸（凹）多边形 最大轮廓

    对点云　直通滤波
    使用采样一致性分割算法　提取平面模型，
    再通过该估计的平面模型系数从滤波后的点云，投影一组点集　到　投影平面上(投影滤波算法)，
    最后为投影后的平面点云　计算其对应的　二维凸多边形（凸包　包围盒）
[凸（凹）多边形 ](convex_hull2d.cpp)

## 无序点云的快速三角化  
    得到点云文件的mesh网格文件ply 文件
    使用贪婪投影三角化算法对有向点云进行三角化，
    具体方法是：
    （1）先将有向点云投影到某一局部二维坐标平面内
    （2）在坐标平面内进行平面内的三角化
    （3）根据平面内三位点的拓扑连接关系获得一个三角网格曲面模型.
    贪婪投影三角化算法原理：
    是处理一系列可以使网格“生长扩大”的点（边缘点）
    延伸这些点直到所有符合几何正确性和拓扑正确性的点都被连上，
    该算法可以用来处理来自一个或者多个扫描仪扫描到得到并且有多
    个连接处的散乱点云但是算法也是有很大的局限性，
    它更适用于采样点云来自表面连续光滑的曲面且点云的密度变化比较均匀的情况.
[无序点云的快速三角化](greedy_projection.cpp)    
    
## 无序点云的B曲线拟合 
    > 1.7 支持
    [参考](https://blog.csdn.net/shenziheng1/article/details/54411098)

    step 1
    主成分分析法PCA初始化B样条曲面
    step 2
    拟合B样条曲面
    step 3
    循环初始化B样条曲线
    step 4
    拟合B样条曲线
    step 5
    三角化B样条曲面

    Monte Carlo particle filter (MCPF)蒙特卡洛粒子滤波 

    NURBS fitting stuff (curve and surfaces) using 
    PDM (point-distance-minimization　点距离最小化), 
    TDM (tangent-distance-minimization　切线距离最小化) and 
    SDM (squared-distance-minimization　平方距离最小化. 
[无序点云的B曲线拟合](bspline_fitting.cpp) 


