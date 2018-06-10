# 双目slam基础 Stereo camera slam
[Stereo Vision:Algorithms and Applications 双目宝典](http://vision.deis.unibo.it/~smatt/Seminars/StereoVision.pdf)
## 0.基础知识 Basic Knowledge
### 相机内参数   Intrinsic parameters
```asm
    u         x     fx  0  cx
    v  =  K * y  =  0   fy cy   
    1         z     0   0   1

     * 相机感光元件CCD的尺寸是8mm X 6mm，
       帧画面的分辨率设置为640X480，
       那么毫米与像素点之间的转换关系就是80pixel/mm  80像素每毫米
     * CCD传感器每个像素点的物理大小为dx*dy，相应地，就有 dx=dy=1/80
     * 假设像素点的大小为k x l，其中 fx = f / k， fy = f / (l * sinA)， 
                                   fx = 80*焦距  fy = 80*焦距
                                   焦距为相机光心到感光平面中心的距离
       A一般假设为 90°，是指摄像头坐标系的偏斜度（就是镜头坐标和CCD是否垂直）。
     * 摄像头矩阵（内参）的目的是把图像的点从图像坐标转换成实际物理的三维坐标。
        因此其中的fx, fy, cx, cy 都是使用类似上面的纲量。
     * 同样，Q 中的变量 f，cx, cy 也应该是一样的。

                                                 |Xw|
     *   |u|    |fx  0   cx 0|     |  R   T |    |Yw|
     *   |v| =  |0   fy  cy 0|  *  |        | *  |Zw| = M * W
     *   |1|    |0   0   1  0|     | 0 0 0 1|    |1 |

     *   像素坐标齐次表示(3*1)  =  内参数矩阵 齐次表示 3*4  ×  外参数矩阵齐次表示 4*4 ×  物体世界坐标 齐次表示  4*1
     *   内参数齐次 × 外参数齐次 整合 得到 投影矩阵M  3*4    
     *   对于左右两个相机 投影矩阵 P1=M1   P2=M2
     *    世界坐标　W 　---->　左相机投影矩阵 P1 ------> 左相机像素点　(u1,v1,1)
     *                 ----> 右相机投影矩阵 P２ ------> 右相机像素点　(u2,v2,1)

     以下更加三角测量可以得到：
     *   Q为 视差转深度矩阵 disparity-to-depth mapping matrix 

     * Z = f*B/d        =   f    /(d/B)    B为相机基线长度  d为两匹配像素的视差
     * X = Z*(x-cx)/fx = (x-c_x)/(d/B)     相似三角形
     * Y = Z*(y-cy)/fy = (y-c_x)/(d/B) 

     X        x
     Y  = Q * y
     Z        d
     1        1

     * Q= | 1   0    0         -c_x     |    Q03
     *    | 0   1    0         -c_y     |    Q13
     *    | 0   0    0          f       |    Q23
     *    | 0   0   -1/B   (c_x-c_x')/B |  
     *              Q32        Q33     

     c_x和c_x'　为左右相机　平面坐标中心的差值（内参数）

     *  以左相机光心为世界坐标系原点   左手坐标系Z  垂直向后指向 相机平面  
     *        |x|   | x-cx         |     |X'|
     *        |y|   | y-cy         |     |Y'|
     *   Q  * |d| = | f            |  =  |Z'| ====>归一化==>  Z = Z/W =  -f*B/(d-c_x+c_x')
     *        |1|   |(-d+cx-cx')/B |     |W |
     
     * Z = f * B/ D    f 焦距 量纲为像素点 B为双目基线距离   
     * 左右相机基线长度 量纲和标定时 所给标定板尺寸 相同 
     * D视差 量纲也为 像素点 分子分母约去，Z的量纲同 B
    Q 示例：
    [ 1., 0., 0., -3.3265590286254883e+02, 
      0., 1., 0., -2.3086411857604980e+02, 
      0., 0., 0., 3.9018919929094244e+02, 
      0., 0., 6.1428092115522364e-04, 0. ]
```
### 相机畸变参数  Extrinsic parameters

     r^2 = x^2+y^2
     * 径向畸变矫正 光学透镜特效  凸起                k1 k2 k3 三个参数确定
     * Xp=Xd(1 + k1*r^2 + k2*r^4 + k3*r^6)
     * Yp=Yd(1 + k1*r^2 + k2*r^4 + k3*r^6)
     
     * 切向畸变矫正 装配误差                         p1  p2  两个参数确定
     * Xp=Xd + ( 2 * p1 * y   +  p2 * (r^2 +2 * x^2) )
     * Yp=Yd + ( p1 * (r^2 + 2 * y^2) + 2 * p2 * x )
## 1. 双目相机校正Stereo Rectification 
[opencv双目校准 程序参考](https://github.com/Ewenwan/MVision/blob/master/stereo/stereo/stereo_calib.cpp)
```asm
    相对位置矩阵：
        R t 为 左相机到右相机 的 旋转与平移矩阵  
            R维度：3*3     
            t维度：3*1  
            t中第一个tx为 双目基线长度 
        
    摆正矩阵:   
        立体校正的时候需要两幅图像共面并且 行对准 以使得立体匹配更加的可靠 
        使得两幅图像共面的方法就是把两个摄像头的图像投影到一个 公共成像面上，
        这样每幅图像从本图像平面投影到 公共图像平面都需要一个旋转矩阵R 
        stereoRectify 这个函数计算的就是从图像平面投影都公共成像平面的旋转矩阵Rl,Rr。 
        Rl,Rr即为左右相机平面行对准的校正旋转矩阵。 
        左相机经过Rl旋转，右相机经过Rr旋转之后，两幅图像就已经共面并且行对准了。 
    投影矩阵：
        其中Pl,Pr为两个相机的投影矩阵，
        其作用是将3D点的坐标转换到图像的2D点的坐标:
        
            P*[X Y Z 1]' =[x y w]  
    重投影矩阵(视差转深度矩阵)：
     Q矩阵为重投影矩阵(视差转深度矩阵)，
     即矩阵Q可以把2维平面(图像平面)上的点投影到3维空间的点:
     
        Q*[x y d 1] = [X Y Z W]。
        
    其中d为左右两幅图像的视差. 
```
![](https://github.com/Ewenwan/MVision/blob/master/vSLAM/img/1_Stereo_standard_form.PNG)


**校准之后的效果**

![](https://github.com/Ewenwan/MVision/blob/master/vSLAM/img/stereo1.PNG)
## 2. 特征提取 Feature Extraction 

## 3. 双目特征匹配 Stereo Feature Matching

## 4. 三角测量得到深度 Triangulation 

## 5. 相邻帧特征匹配 Temporal Feature Matching 

## 6. 姿态恢复/跟踪/随机采样序列 Incremental Pose Recovery/RANSAC 


