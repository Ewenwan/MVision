# 双目相机 算法
[ ADCensus, SGBM, BM算法参考](https://github.com/DLuensch/StereoVision-ADCensus)
[ELAS论文解析](https://www.cnblogs.com/sinbad360/p/6883623.html)
[ELAS代码](https://github.com/Ewenwan/ELAS)
[棋盘格](https://github.com/DLuensch/StereoVision-ADCensus/tree/master/Documents/chessboards)

[双目视差图 精细化调整算法](https://github.com/Ewenwan/FDR)

[DynamicStereo 算法较多 参考](https://github.com/Ewenwan/DynamicStereo)


[双目算法、光流算法、分层深度图估计算法 ](https://github.com/Ewenwan/ParallelFusion)

## 双目相机 矫正 
    * 用法 ./Stereo_Calibr -w=6 -h=8  -s=24.3 stereo_calib.xml   
      我的  ./Stereo_Calibr -w=8 -h=10 -s=200 stereo_calib.xml
    * ./stereo_calib -w=9 -h=6 stereo_calib.xml 标准图像
    *  实际的正方格尺寸在程序中指定 const float squareSize = 2.43f;    
       2.43cm  mm为单位的话为 24.3  0.1mm为精度的话为   243 注意 标定结果单位(纲量)和此一致
[原理分析](https://www.cnblogs.com/polly333/p/5013505.html )
[原理分析2](http://blog.csdn.net/zc850463390zc/article/details/48975263)
 
### 径向畸变矫正 光学透镜特效  凸起                k1 k2 k3 三个参数确定

     Xp=Xd(1 + k1*r^2 + k2*r^4 + k3*r^6)
     Yp=Yd(1 + k1*r^2 + k2*r^4 + k3*r^6)

### 切向畸变矫正 装配误差                         p1  p2  两个参数确定

     Xp= Xd + ( 2*p1*y  + p2*(r^2 + 2*x^2) )
     Yp= Yd + ( p1 * (r^2 + 2*y^2) + 2*p2*x )
     r^2 = x^2+y^2
    
### 投影公式 世界坐标点 到 相机像素坐标系下
                                                    | Xw|
    *  | u|     |fx  0   ux 0|     |    R   T  |    | Yw|
    *  | v| =   |0   fy  uy 0|  *  |           | *  | Zw| = M*W
    *  | 1|     |0   0  1   0|     |   0 0  0 1|    | 1 |

[相机标定和三维重建](http://wiki.opencv.org.cn/index.php/Cv)

    *   像素坐标齐次表示(3*1)  =  内参数矩阵 齐次表示(3*4)  ×  外参数矩阵齐次表示(4*4) ×  物体世界坐标 齐次表示(4*1)
    *   内参数齐次 × 外参数齐次 整合 得到 投影矩阵  3*4    左右两个相机 投影矩阵 P1 = K1*T1   P2 = k2*T2
    *   世界坐标　W 　---->　左相机投影矩阵 P1 ------> 左相机像素点　(u1,v1,1)
    *               ----> 右相机投影矩阵 P２ ------> 右相机像素点　(u2,v2,1)

### Q为 视差转深度矩阵 disparity-to-depth mapping matrix 
    * Z = f*B/d       =   f    /(d/B)
    * X = Z*(x-c_x)/f = (x-c_x)/(d/B)
    * X = Z*(y-c_y)/f = (y-y_x)/(d/B)

[ Q为 视差转深度矩阵 参考](http://blog.csdn.net/angle_cal/article/details/50800775)

    *    Q= | 1   0    0         -c_x     |    Q03
    *       | 0   1    0         -c_y     |    Q13
    *       | 0   0    0          f       |    Q23
    *       | 0   0   -1/B   (c_x-c_x')/B |    c_x和c_x'　为左右相机　平面坐标中心的差值（内参数）
    *                   Q32        Q33

    *  以左相机光心为世界坐标系原点   左手坐标系Z  垂直向后指向 相机平面  
    *           |x|      | x-c_x           |     |X|
    *           |y|      | y-c_y           |     |Y|
    *   Q    *  |d| =    |   f             |  =  |Z|======>  Z'  =   Z/W =     f/((-d+c_x-c_x')/B)
    *           |1|      |(-d+c_x-c_x')/B  |     |W|         X'  =   X/W = ( x-c_x)/((-d+c_x-c_x')/B)
                                                             Y'  =   Y/W = ( y-c_y)/((-d+c_x-c_x')/B)
                                                            与实际值相差一个负号
 ## 双目 视差与深度的关系                                                         
    Z = f * T / D   
    f 焦距 量纲为像素点  
    T 左右相机基线长度 
    量纲和标定时 所给标定板尺寸 相同 
    D视差 量纲也为 像素点 分子分母约去，
    Z的量纲同 T
 
 ## 像素单位与 实际物理尺寸的关系
    * CCD的尺寸是8mm X 6mm，帧画面的分辨率设置为640X480，那么毫米与像素点之间的转换关系就是80pixel/mm
    * CCD传感器每个像素点的物理大小为dx*dy，相应地，就有 dx=dy=1/80
    * 假设像素点的大小为k x l，其中 fx = f / k， fy = f / (l * sinA)， 
      A一般假设为 90°，是指摄像头坐标系的偏斜度（就是镜头坐标和CCD是否垂直）。
    * 摄像头矩阵（内参）的目的是把图像的点从图像坐标转换成实际物理的三维坐标。因此其中的fx,y, cx, cy 都是使用类似上面的纲量。
    * 同样，Q 中的变量 f，cx, cy 也应该是一样的。

        参考代码　
        https://github.com/yuhuazou/StereoVision/blob/master/StereoVision/StereoMatch.cpp
        https://blog.csdn.net/hujingshuang/article/details/47759579

        http://blog.sina.com.cn/s/blog_c3db2f830101fp2l.html

## 视差快匹配代价函数：

     【1】对应像素差的绝对值和（SAD, Sum of Absolute Differences） 

     【2】对应像素差的平方和（SSD, Sum of Squared Differences）

     【3】图像的相关性（NCC, Normalized Cross Correlation） 归一化积相关算法

     【4】ADCensus 代价计算  = AD＋Census

         1. AD即 Absolute Difference 三通道颜色差值绝对值之和求均值
         2. Census Feature特征原理很简单
                  在指定窗口内比较周围亮度值与中心点的大小，匹配距离用汉明距表示。
                  Census保留周边像素空间信息，对光照变化有一定鲁棒性。
         3. 信息结合
                  cost = r(Cad , lamd1) + r(Cces, lamd2)
                  r(C , lamd) = 1 - exp(- c/ lamd)
                 Cross-based 代价聚合:
                  自适应窗口代价聚合，
          在设定的最大窗口范围内搜索，
               满足下面三个约束条件确定每个像素的十字坐标，完成自适应窗口的构建。
              Scanline 代价聚合优化
### ADCensus计算流程
      1．ADcensus代价初始化　+　
      2. 自适应窗口代价聚合　+　
      3. 扫描线全局优化　+ 
      4. 代价转视差, 外点(遮挡点+不稳定)检测  + 
      5. 视差传播－迭代区域投票法 使用临近好的点为外点赋值　   + 
      6. 视差传播-16方向极线插值（对于区域内点数量少的　外点　再优化） +
      7. candy边缘矫正 + 
      8. 亚像素求精,中值滤波 

### ELAS 算法分析  Efficient Large-Scale Stereo Matching 
      该算法主要用于在双目立体视觉中进行快速高清晰度图像匹配。
#### 算法基本思想为：
      1. 通过计算一些支持点组成稀疏视差图，
      
      2. 对这些支持点在图像坐标空间进行三角剖分，构建视差的先验值
      
      3. 由于支持点可被精确匹配，避免了使用其余点进行匹配造成的匹配模糊。
      
      4. 进而可以通过有效利用视差搜索空间，重建精确的稠密视差图，而不必进行全局优化。
      
#### 算法分为以下几个部分：
      
##### 1.匹配支持点
      首先确定支持点匹配的特征描述算子，
      文中采用简单的9X9尺寸的sobel滤波 
      并连结周围像素窗口的sobel值组成特征。

      特征算子维度为1+11+5=17，作者有提到使用更复杂的surf特征对提高匹配的精度并无益处，反而使得速度更慢。

      匹配方法为L1向量距离，并进行从左到右及从右到左两次匹配。

      为防止多个匹配点歧义，剔除最大匹配点与次匹配点匹配得分比超过一定阀值的点。

      另外则是增加图像角点作为支持点，角点视差取其最近邻点的值。

##### 2.立体匹配生成模型
      这里所谓的生成模型，
      1. 简单来讲就是基于上面确定的支持点集，也可以扩展一些角点；
      2. 再对这些支持点集进行三角剖分，形成多个三角形区域。
      3. 在每个三角形内基于三个已知顶点的精确视差值 进行 MAP最大后验估计 插值 该三角区域内的其他点视差。
      
      最大似然估计（Maximum likelihood estimation, 简称MLE）和
      最大后验概率估计（Maximum a posteriori estimation, 简称MAP）是很常用的两种参数估计方法,都是统计学问题。
      
###### 使用支持点 视差 周围构建一个高斯分布的 先验值      
      
      
###### 概率（probabilty）和统计（statistics）看似两个相近的概念，其实研究的问题刚好相反。
###### A. 概率研究的问题是，已知一个模型和参数，怎么去预测这个模型产生的结果的特性（例如均值，方差，协方差等等）。
[参考](https://blog.csdn.net/u011508640/article/details/72815981)

      举个例子，
      1. 我想研究怎么养猪（模型是猪），
      2. 我选好了想养的品种、喂养方式、猪棚的设计等等（选择参数），
      3. 我想知道我养出来的猪大概能有多肥，肉质怎么样（预测结果）。
      
###### B. 统计研究的问题则相反。统计是，有一堆数据，要利用这堆数据去预测模型和参数。
      仍以猪为例。
      1. 现在我买到了一堆肉，
      2. 通过观察和判断，我确定这是猪肉
      （这就确定了模型。在实际研究中，也是通过观察数据推测模型 是 高斯分布? 指数分布? 拉普拉斯分布? ），
      3. 然后，可以进一步研究，判定这猪的品种、这是圈养猪还是跑山猪还是网易猪，等等（推测模型参数）。
      
###### C. 一句话总结：
      概率是已知模型和参数，推数据。
      统计是已知数据，推模型和参数。
      
###### 贝叶斯公式到底在说什么
      1. 条件概率： 
                  P(A/B) = P(AB)/P(B)  B事件发生的条件下， A事件发生的概率

      2. 乘法公式： 
                  可以由条件概率得到 P(AB) = P(A/B) * P(B) = P(AB)/P(A) * P(A) = P(B/A) * P(A)
                  AB事件都发生的概率 = B发生下A也发生 = A发生下B也发生
      3. 全概率公式： 
                  S为全集，B1,...Bn为S集合的一个全划分
                  P(A) = P(AS)=P(A(B1+B2+...+BN)) = P(AB1)+...+P(ABn)
                       = P(A/B1)P(B1) + ... + P(A/Bn)P(Bn)
                       = sum(P(A/Bi) * P(Bi))
      4.贝叶斯公式：
                 P(Bi/A) = P(ABi)/P(A)                          条件概率 + 
                         = P(A/Bi)P(Bi)/P(A)                    乘法公式 +
                         = P(A/Bi)P(Bi)/(sum(P(A/Bi) * P(Bi)))  全概率公式

      意思就是说 有多种情况可以导致 观测事件A发生，需要求当观测到 事件A发生了，由事件B1引发的概率（P(B1/A)）
      但是直接求不容易求解，可以转化成 在原因事件Bi发生下，结果事件A发生的概率来联合求解。

      例如，观测到 汽车报警响了A，可能由 行人误碰 引起B1，也可能真是由于小偷偷车引起B2

      一般情况下：
      行人误碰 B1 事件发生的概率为 0.7
      小偷偷车 B2 事件发生的概率为 0.3

      而 行人误碰事件B1发生下 汽车发出声响的概率为 0.6  不发出声响的概率为 0.4
      而 小偷偷车事件B2发生下 汽车发出声响的概率为 0.4  不发出声响的概率为 0.6

      当 观测到 汽车报警响了 事件A发生，求是由于小偷偷车引起的概率

      P(B2/A) = P(AB2)/P(A) 
              = P(A/B2) * P(B2)/( P(A/B1)* P(B1) + P(A/B2)* P(B2) )
              = 0.4 * 0.3 / (0.7 * 0.6 + 0.3 * 0.4 )
              = 0.2222222
###### 似然函数（likelihood function）  似然（likelihood）这个词其实和概率（probability）是差不多的意思

      1. P(X | O) 表示一个概率， X是一个数据， O是模型参数；
      2. 模型参数O已知， 数据X是变量，已知模型参数，求不同数据样本点出现的概率，是概率函数，是用来确定发生概率最大的数据；
      3. 数据X已知， 模型参数O是变量，已知数据X出现，求这个数据符合不同模型参数的概率，是似然函数，是用来估计模型参数的。

######  最大似然估计（Maximum likelihood estimation, 简称MLE）
      已知数据，模型参数为变量，求使得函数值最大的，模型参数。
      例如：
      1. 有一枚硬币，我们想确定这枚硬币（模型对象）是否是均匀的（抛出来正反面概率一致，模型参数）
      2. 我们需要获取数据，于是我们那这枚硬币抛了10次，得到数据X0是：反正正正正反正正正反。
      3. 假设出现正面的概率为O，是模型参数，且反面概率为 1-O （假设为二项分布，不会出现站立的情况）
      4. 那么出现这个数据的概率输是 f(x0,O) = (1-O)OOOO(1-O)OOO(1-O) = O^7 * (1-O)^3
      5. 该似然函数 f(x0,O)，是一个关于O的函数f(O)，最大似然的意思就是要最大化这个函数
      6. 可以画出f(O)的图象，可以得到 在 O = 0.7的时候，似然函数取得最大值   开口向下
      
      结论：
      做10次实验，有7次正面向上，最大使然认为正面向上的概率是 0.7. 有点不太合理...? 贝叶斯学派就提出了异议
###### 最大后验概率估计（Maximum a posteriori estimation, 简称MAP）
      考虑先验概率分布。 为此，引入了最大后验概率估计。

      1. 最大似然估计     是求参数O，  使得 似然函数概率 P(X | O)最大
      2. 最大后验概率估计  是求参数O，  使得 函数概率 P(X | O) * P(O)最大 也就是 P(O|X0) 后验概率最大
      思想：
         求得的O，不单单让似然函数最大，O自己出现的概率也要比较大才可以；
         有点像正则化里的 加参数惩罚项的思想（让在参数较小的情况下，参数组成的损失函数也比较小）
         不过正则化里是利用  加法 (因为是最小化，两者都比较小，其和才比较小)
         而 最大后验概率估计 MAP 是利用 乘法(两个都比较大，乘积才比较大)

      其实有点 与门的思想， 要使得 两个量运算之后在两者都较小的情况下取得最小值，使用加法，
          两个都较小相加之后还是比较小，如果一个大一个小，相乘之后，积还是比较小，所以要使得三者都比较小，使用 加法 运算
      要使得三者都比较大，使用乘法运算，
          两个都比较大，相乘之后还是比较大，一个较大一个较小，相乘之后就比较小；而一个较大一个较小，相加之后比较大。


      MAP其实是在 最大化 P(O|X0) = P(X0|O)*P(O)/P(X0) 
      因为 X0 是确定的 在总实验次数下 出现X0数据的概率是可以有实验数据确定性得到的，所以就不考虑分母了

      P(O|X0) 就是数据X0已经出现了，要就O去什么值使得 P(O|X0) 后验概率 最大


###### 模型参数 的先验知识
      由先验知识（常识）知道 模型参数O 取的 0.5 的概率最大，所以我们可以假设，P(O)符合均值为0.5 方差为 0.1的高斯分布
      也就是 在 0.5 出概率最大，两边逐渐减小
      
      P(O) =  1/(sqrt(2*pi)*方差) * exp(-(O - 均值)^2/(2*方差^2))
           =  1/(sqrt(2*pi)*0.1) * exp(-(O - 0.5)^2/(2*0.1^2))
           
       P(X0 | O) * P(O) 的图像 开口向下 峰值横坐标 由0.7 向 0.5 偏移
       得到在 O = 0.558的情况下 后验概率  P(O|X0) 最大
###### 总结
       1. 最大后验概率估计MAP 比最大似然估计MLE 对了一个作为乘法因子的 先验概率 P(O)
       2. 或者也可以认为，MLE 相对于 MAP 是把 先验概率 P(O) = 1，即认为 模型参数O 是均匀分布
           

##### 3.视差估计
      视差估计依赖最大后验估计（MAP）来计算其余观察点的视差值。
      作为条件来判断是否对周围一定范围内像素进行能量值最小化计算，最后选取能量最小时的 d值作为该点视差值。
##### 4.提纯
      后面主要是对E(d)进行条件约束，如 d-u < 3 等约束处理。
##### 





## 现今Stereo matching算法大致可以分为三个部分： pre-process 、stereo matching 、post-process。
## 1图像增强　2匹配　 3视差优化
        
    pre-process即为USM图像增强，直方图归一化或直方图规定化。

    post-process即为常规的disparity refinement，一般stereo matching算法出来的结果不会太好，
    可能很烂，但经过refinement后会得到平滑的结果。

    种方法就是以左目图像的源匹配点为中心，

### 视差获取

    对于区域算法来说，在完成匹配代价的叠加以后，视差的获取就很容易了，
    只需在一定范围内选取叠加匹配代价最优的点
    （SAD和SSD取最小值，NCC取最大值）作为对应匹配点，
    如胜者为王算法WTA（Winner-take-all）。
    而全局算法则直接对原始匹配代价进行处理，一般会先给出一个能量评价函数，
    然后通过不同的优化算法来求得能量的最小值，同时每个点的视差值也就计算出来了。
    大多数立体匹配算法计算出来的视差都是一些离散的特定整数值，可满足一般应用的精度要求。
    但在一些精度要求比较高的场合，如精确的三维重构中，就需要在初始视差获取后采用一些措施对视差进行细化，
    如匹配代价的曲线拟合、图像滤波、图像分割等。

###   立体匹配约束
      1）极线约束
      2）唯一性约束
      3）视差连续性约束
      4）顺序一致性约束
      5）相似性约束

### 相似性判断标准
    1）像素点灰度差的平方和，即 SSD
    2）像素点灰度差的绝对值和，即 SAD
    3）归一化交叉相关，简称 NCC
    4） 零均值交叉相关，即 ZNCC
    5）Moravec 非归一化交叉相关，即 MNCC
    6）Kolmogrov-Smrnov 距离，即 KSD
    7）Jeffrey 散度
    8）Rank 变换（是以窗口内灰度值小于中心像素灰度值的像素个数来代替中心像素的灰度值）
    9）Census 变换（是根据窗口内中心像素灰度与其余像素灰度值的大小关系得一串位码，位码长度等于窗口内像素个数减一）


## SAD方法就是以左目图像的源匹配点为中心，定义一个窗口D，其大小为（2m+1） (2n+1)，
        统计其窗口的灰度值的和，然后在右目图像中逐步计算其左右窗口的灰度和的差值，
        最后搜索到的差值最小的区域的中心像素即为匹配点。
        基本流程：
          1.构造一个小窗口，类似与卷积核。
          2.用窗口覆盖左边的图像，选择出窗口覆盖区域内的所有像素点。
          3.同样用窗口覆盖右边的图像并选择出覆盖区域的像素点。
          4.左边覆盖区域减去右边覆盖区域，并求出所有像素点差的绝对值的和。
          5.移动右边图像的窗口，重复3，4的动作。（这里有个搜索范围，超过这个范围跳出）
          6.找到这个范围内SAD值最小的窗口，即找到了左边图像的最佳匹配的像素块。

         由 以上三种算法可知，SAD算法最简单，因此当模板大小确定后，SAD算法的速度最快。NCC算法与SAD算法相比要复杂得多。
        ------------------------------------
        SAD（Sum of Absolute Difference）=SAE（Sum of Absolute Error)即绝对误差和
        SSD（Sum of Squared Difference）=SSE（Sum of Squared Error)即差值的平方和
        SATD（Sum of Absolute Transformed Difference）即hadamard变换后再绝对值求和
        MAD（Mean Absolute Difference）=MAE（Mean Absolute Error)即平均绝对差值
        MSD（Mean Squared Difference）=MSE（Mean Squared Error）即平均平方误差


### 三角测量原理：
         现实世界物体坐标　—(外参数 变换矩阵Ｔ变换)—>  
         相机坐标系　—(同/Z)—>归一化平面坐标系——>径向和切向畸变矫正——>(内参数平移　Cx Cy 缩放焦距Fx Fy)
         ——> 图像坐标系下　像素坐标
         
         u=Fx *X/Z + Cx 　　像素列位置坐标　
         v=Fy *Y/Z + Cy 　　像素列位置坐标　

         反过来
         X=(u- Cx)*Z/Fx
         Y=(u- Cy)*Z/Fy
         
         Z轴归一化
         X=(u- Cx)*Z/Fx/depthScale
         Y=(u- Cy)*Z/Fy/depthScale
         Z=Z/depthScale

        外参数　T
        世界坐标　
        pointWorld = T*[X Y Z]

##  OpenCV三种立体匹配求视差图算法总结:

### 首先我们看一下BM算法：
            Ptr<StereoBM> bm = StereoBM::create(16,9);//局部的BM;
            // bm算法
            bm->setROI1(roi1);//左右视图的有效像素区域 在有效视图之外的视差值被消零
            bm->setROI2(roi2);
            bm->setPreFilterType(CV_STEREO_BM_XSOBEL);
            bm->setPreFilterSize(9);//滤波器尺寸 [5,255]奇数
            bm->setPreFilterCap(31);//预处理滤波器的截断值 [1-31] 
            bm->setBlockSize(SADWindowSize > 0 ? SADWindowSize : 15);//sad窗口大小
            bm->setMinDisparity(0);//最小视差值，代表了匹配搜索从哪里开始
            bm->setNumDisparities(numberOfDisparities);//表示最大搜索视差数
            bm->setTextureThreshold(10);//低纹理区域的判断阈值 x方向导数绝对值之和小于阈值
            bm->setUniquenessRatio(15);//视差唯一性百分比  匹配功能函数
            bm->setSpeckleWindowSize(100);//检查视差连通域 变化度的窗口大小
            bm->setSpeckleRange(32);//视差变化阈值  当窗口内视差变化大于阈值时，该窗口内的视差清零
            bm->setDisp12MaxDiff(-1);// 1

        该方法速度最快，一副320*240的灰度图匹配时间为31ms 



### 第二种方法是SGBM方法这是OpenCV的一种新算法：
        Ptr<StereoSGBM> sgbm = StereoSGBM::create(0,16,3);//全局的SGBM;
         // sgbm算法
            sgbm->setPreFilterCap(63);//预处理滤波器的截断值 [1-63] 
            int sgbmWinSize = SADWindowSize > 0 ? SADWindowSize : 3;
            sgbm->setBlockSize(sgbmWinSize);
            int cn = img0.channels();
            sgbm->setP1(8*cn*sgbmWinSize*sgbmWinSize);// 控制视差变化平滑性的参数。P1、P2的值越大，视差越平滑。
        //P1是相邻像素点视差增/减 1 时的惩罚系数；P2是相邻像素点视差变化值大于1时的惩罚系数。P2必须大于P1
            sgbm->setP2(32*cn*sgbmWinSize*sgbmWinSize);
            sgbm->setMinDisparity(0);//最小视差值，代表了匹配搜索从哪里开始
            sgbm->setNumDisparities(numberOfDisparities);//表示最大搜索视差数
            sgbm->setUniquenessRatio(10);//表示匹配功能函数
            sgbm->setSpeckleWindowSize(100);//检查视差连通域 变化度的窗口大小
            sgbm->setSpeckleRange(32);//视差变化阈值  当窗口内视差变化大于阈值时，该窗口内的视差清零
            sgbm->setDisp12MaxDiff(-1);// 1
        //左视图差（直接计算）和右视图差（cvValidateDisparity计算得出）之间的最大允许差异
            if(alg==STEREO_HH)               sgbm->setMode(StereoSGBM::MODE_HH);
            else if(alg==STEREO_SGBM)  sgbm->setMode(StereoSGBM::MODE_SGBM);
            else if(alg==STEREO_3WAY)   sgbm->setMode(StereoSGBM::MODE_SGBM_3WAY);
        各参数设置如BM方法，速度比较快，320*240的灰度图匹配时间为78ms，

         第二部分：opencvSGBM算法的参数含义及数值选取

         一 预处理参数

         1：preFilterCap：水平sobel预处理后，映射滤波器大小。默认为15

         int ftzero =max(params.preFilterCap, 15) | 1;

         opencv测试例程test_stereomatching.cpp中取63。

         二 代价参数

         2：SADWindowSize:计算代价步骤中SAD窗口的大小。由源码得，此窗口默认大小为5。

         SADWindowSize.width= SADWindowSize.height = params.SADWindowSize > 0 ?params.SADWindowSize : 5;

         注：窗口大小应为奇数，一般应在3x3到21x21之间。

         3：minDisparity：最小视差，默认为0。此参数决定左图中的像素点在右图匹配搜索的起点。int 类型

         4：numberOfDisparities：视差搜索范围，其值必须为16的整数倍（CV_Assert( D % 16 == 0 );）。最大搜索边界= numberOfDisparities+ minDisparity。int 类型

         三 动态规划参数

         动态规划有两个参数，分别是P1、P2，它们控制视差变化平滑性的参数。P1、P2的值越大，视差越平滑。P1是相邻像素点视差增/减 1 时的惩罚系数；P2是相邻像素点视差变化值大于1时的惩罚系数。P2必须大于P1。需要指出，在动态规划时，P1和P2都是常数。

         5：opencv测试例程test_stereomatching.cpp中，P1 = 8*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;

         6：opencv测试例程test_stereomatching.cpp中，P2 = 32*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;

         四：后处理参数

         7：uniquenessRatio：唯一性检测参数。对于左图匹配像素点来说，先定义在numberOfDisparities搜索区间内的最低代价为mincost，次低代价为secdmincost。如果满足

         即说明最低代价和次第代价相差太小，也就是匹配的区分度不够，就认为当前匹配像素点是误匹配的。

         opencv测试例程test_stereomatching.cpp中，uniquenessRatio=10。int 类型

         8：disp12MaxDiff：左右一致性检测最大容许误差阈值。int 类型

         opencv测试例程test_stereomatching.cpp中，disp12MaxDiff =1。

         9：speckleWindowSize：视差连通区域像素点个数的大小。对于每一个视差点，当其连通区域的像素点个数小于speckleWindowSize时，认为该视差值无效，是噪点。

         opencv测试例程test_stereomatching.cpp中，speckleWindowSize=100。

         10：speckleRange：视差连通条件，在计算一个视差点的连通区域时，当下一个像素点视差变化绝对值大于speckleRange就认为下一个视差像素点和当前视差像素点是不连通的。

         opencv测试例程test_stereomatching.cpp中，speckleWindowSize=10。

### 第三种为GC方法：
        该方法速度超慢，但效果超好。
