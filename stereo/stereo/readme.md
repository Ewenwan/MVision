# 双目相机 算法
[ ADCensus, SGBM, BM算法参考](https://github.com/DLuensch/StereoVision-ADCensus)
[ELAS论文解析](https://www.cnblogs.com/sinbad360/p/6883623.html)
[ELAS代码](https://github.com/Ewenwan/ELAS)
[棋盘格](https://github.com/DLuensch/StereoVision-ADCensus/tree/master/Documents/chessboards)

## 双目相机 矫正 
    * 用法 ./Stereo_Calibr -w=6 -h=8  -s=24.3 stereo_calib.xml   
      我的  ./Stereo_Calibr -w=8 -h=10 -s=200 stereo_calib.xml
    * ./stereo_calib -w=9 -h=6 stereo_calib.xml 标准图像
    *  实际的正方格尺寸在程序中指定 const float squareSize = 2.43f;    
       2.43cm  mm为单位的话为 24.3  0.1mm为精度的话为   243 注意 标定结果单位(纲量)和此一致
[原理分析](https://www.cnblogs.com/polly333/p/5013505.html )
[原理分析2](http://blog.csdn.net/zc850463390zc/article/details/48975263)
 
### 径向畸变矫正 光学透镜特效  凸起                k1 k2 k3 三个参数确定

    * Xp=Xd(1 + k1*r^2 + k2*r^4 + k3*r^6)
    * Yp=Yd(1 + k1*r^2 + k2*r^4 + k3*r^6)

### 切向畸变矫正 装配误差                         p1  p2  两个参数确定

    * Xp= Xd + ( 2*p1*y  + p2*(r^2 + 2*x^2) )
    * Yp= Yd + ( p1 * (r^2 + 2*y^2) + 2*p2*x )
    * r^2 = x^2+y^2
    
### 投影公式 世界坐标点 到 相机像素坐标系下
                                                      | Xw|
    * 			| u|     |fx  0   ux 0|     |    R   T  |    | Yw|
    *    | v| =   |0   fy  uy 0|  *  |  		       | *  | Zw| = M*W
    * 			| 1|     |0   0  1   0|     |   0 0  0 1|    | 1 |

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

    *     Q= | 1   0    0         -c_x     |    Q03
    *  	     | 0   1    0         -c_y     |    Q13
    *   	    | 0   0    0          f       |    Q23
    *   	    | 0   0   -1/B   (c_x-c_x')/B |    c_x和c_x'　为左右相机　平面坐标中心的差值（内参数）
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


        SAD方法就是以左目图像的源匹配点为中心，定义一个窗口D，其大小为（2m+1） (2n+1)，
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


        三角测量原理：
         现实世界物体坐标　—(外参数 变换矩阵Ｔ变换)—>  相机坐标系　—(同/Z)—>归一化平面坐标系——>径向和切向畸变矫正——>(内参数平移　Cx Cy 缩放焦距Fx Fy)
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

        OpenCV三种立体匹配求视差图算法总结:

        首先我们看一下BM算法：
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



        第二种方法是SGBM方法这是OpenCV的一种新算法：
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

        第三种为GC方法：
        该方法速度超慢，但效果超好。
