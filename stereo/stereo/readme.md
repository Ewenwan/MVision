# 双目相机
[棋盘格](https://github.com/DLuensch/StereoVision-ADCensus/tree/master/Documents/chessboards)
 * 双目相机 矫正 
 * 用法 ./Stereo_Calibr -w=6 -h=8  -s=24.3 stereo_calib.xml   我的  ./Stereo_Calibr -w=8 -h=10 -s=200 stereo_calib.xml
 * ./stereo_calib -w=9 -h=6 stereo_calib.xml 标准图像
 *  实际的正方格尺寸在程序中指定 const float squareSize = 2.43f;    2.43cm  mm为单位的话为 24.3  0.1mm为精度的话为   243 注意 标定结果单位(纲量)和此一致
 * https://www.cnblogs.com/polly333/p/5013505.html 原理分析
 * http://blog.csdn.net/zc850463390zc/article/details/48975263
 * 
# * 径向畸变矫正 光学透镜特效  凸起                k1 k2 k3 三个参数确定
 * Xp=Xd(1 + k1*r^2 + k2*r^4 + k3*r^6)
 * Yp=Yd(1 + k1*r^2 + k2*r^4 + k3*r^6)
# * 切向畸变矫正 装配误差                                 p1  p2  两个参数确定
 * Xp=Xd + ( 2 * p1 * y  +p2 * (r^2 +2 * x^2) )
 * Yp=Yd + ( p1 * (r^2 + 2 * y^2) + 2 * p2 * x )
 * r^2 = x^2+y^2                  						      | Xw|
 * 			| u|     |fx  0   ux 0|     |    R   T|     | Yw|
 *      | v| =   |0   fy  uy 0|  *  |  		     | *  | Zw|=M*W
 * 			|1|      |0   0   1   0|    |    0 0  0 1|  | 1 |
 * http://wiki.opencv.org.cn/index.php/Cv相机标定和三维重建
 *   像素坐标齐次表示(3*1)  =  内参数矩阵 齐次表示 3*4  ×  外参数矩阵齐次表示 4*4 ×  物体世界坐标 齐次表示  4*1
 *   内参数齐次 × 外参数齐次 整合 得到 投影矩阵  3*4    左右两个相机 投影矩阵 P1   P2
 *  Q为 视差转深度矩阵 disparity-to-depth mapping matrix 
 * 
 *  http://blog.csdn.net/angle_cal/article/details/50800775
 * Q= | 1   0    0  	  -C_x |
 *  	  | 0   1    0  	  -C_y |
 *   	  | 0   0    0         f    |
 *   	  | 1   0    -1/T   (c_x-c_x')/T |
 *     
 *  以左相机光心为世界坐标系原点   左手坐标系Z  垂直向后指向 相机平面  
 *           |x|      | x-C_x                   |     |X|
 *           |y|      | y-C_y                   |     |Y|
 *   Q  * |d| =   |f                            |  = |Z|======>  Z  =   Z/W =    -  f*T/(d-c_x+c_x')
 *           |1|      |(-d+c_x-c_x')/T     |     |W|
 * Z = f * T / D    f 焦距 量纲为像素点  T 左右相机基线长度 量纲和标定时 所给标定板尺寸 相同  D视差 量纲也为 像素点 分子分母约去，Z的量纲同 T
 * 
 * 
 * CCD的尺寸是8mm X 6mm，帧画面的分辨率设置为640X480，那么毫米与像素点之间的转换关系就是80pixel/mm
 * CCD传感器每个像素点的物理大小为dx*dy，相应地，就有 dx=dy=1/80
 * 假设像素点的大小为k x l，其中 fx = f / k， fy = f / (l * sinA)， A一般假设为 90°，是指摄像头坐标系的偏斜度（就是镜头坐标和CCD是否垂直）。
 * 摄像头矩阵（内参）的目的是把图像的点从图像坐标转换成实际物理的三维坐标。因此其中的fx,y, cx, cy 都是使用类似上面的纲量。
 * 同样，Q 中的变量 f，cx, cy 也应该是一样的。
