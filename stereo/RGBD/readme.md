# RGBD 相机
## 图漾 FM810-HD
    是通过两个红外摄像头（获取结构光信息）,
    加一个激光发射器（发出结构光编码）进行深度信息测量，就是双目+结构光的方式，
    而市场上同类产品或者是单目结构光（一个红外摄像头），或者是纯双目（两个普通RGB摄像头）.
    
    在 Linux 环境下执行以下命令下载 SDK。
    gitclone https://github.com/percipioxyz/Camport2.git
    
    ros下的工程 https://github.com/Ewenwan/camport2
    
    1.  根据 Linux 系统的权限管理机制,默认情况下需要 root 权限才能操作深度摄像头设备,
        非 root用户可以创建 udev rule 来修改设备权限。
        /etc/udev/rules.d/88-tyusb.rules
        规则文件名的开头须为数字(0 ~ 99),数字越大,优先级越高。
    2. camport2/lib/linux 目录下相应平台的文件夹,将文件夹中的 libcamm.so 文件复制到 /usr/lib/ 目
    录下

    3. 进入 camport2 目录,执行以下命令编译实例代码。
    cd sample
    mkdir build
    cd build
    cmake ..
    make


    4. 在 camport2/sample/build/bin 目录下生成若干编译生成的可执行文件,
       以 root 权限运行如下命令,即可查看深度图。
    sudo ./SimpleView_FetchFrame
    
    
    5. 新增 RGB彩色图像采集程序,用于相机标定
       sample/Capature_rgb_Img/main.cpp
       
       修改 sample/CMakeLists.txt：
       set(ALL_SAMPLES
            Capature_rgb_Img  # 新增加
            LoopDetect
            ...
            )

# RGB相机标定结果：
[参考](https://blog.csdn.net/qingsong1001/article/details/81779236)

    %YAML:1.0
    calibration_time: "2018年09月29日 星期六 10时59分35秒"
    nframes: 40            # 进行 标定的图像数量
    image_width: 640       #  图像的像素 尺寸
    image_height: 480
    board_width: 8         # 水平方向corners数量 
    board_height: 10       # 垂直方向corners数量
    square_size: 1.        # 默认方格的大小
    aspectRatio: 1.
    flags: 2               #calibrateCamera函数进行标定时调整的方式
    camera_matrix: !!opencv-matrix  # 内参数
       rows: 3
       cols: 3
       dt: d
       data: [ 5.5991319678858906e+02, 0., 2.8763263841844133e+02, 0.,
           5.5991319678858906e+02, 2.6363575121251381e+02, 0., 0., 1. ]   # fx 0 cx 0 fy cy 0 0 1

    distortion_coefficients: !!opencv-matrix    # 畸变参数
       rows: 5
       cols: 1
       dt: d
       data: [ -4.4932428999903368e-01, 3.6014186521811137e-01,
           -8.5242709410947044e-04, 1.1130236682130820e-03,
           -2.3177353174684856e-01 ]               # k1,k2,p1,p2,k3,[k4,k5,k6,s1,s2,s3,s4]
           
## API 框架获取到的 参数
    1280*960 图像:
        内参数k: 1117.91 0      612.558 
                 0      1118.19 536.986 
                 0      0       1
        畸变参数:
             -0.40128  0.407587 0.000954767 0.000714202 0.114102 
             0.0422759 0.11784  0.370694    -0.0115273  0.00464497 -0.00642652 0.00200558
    640*480 图像:
        内参数：
                558.957 0       306.279 
                0       559.094 268.493 
                0       0       1 
        畸变参数：
                -0.40128  0.407587 0.000954767 0.000714202 0.114102 
                0.0422759 0.11784  0.370694    -0.0115273  0.00464497 -0.00642652 0.00200558
                
#  主动立体双目算法的框架

https://blog.csdn.net/u013626386/article/details/79892149

    step1. 双目设备的标定；
    step2. 双目设备的校准；
    step3. 双目立体匹配算法；
    step4. 视差数据的去噪与空洞修复
    step5. 视差数据映射到三维深度值
       如果涉及到输出显示 RGB point cloud，需要另外结合1颗RGB彩色摄像头，
       标定位置关系后可以将点云数据的RGB值一一对应上，用作3D彩色显示，
    step6. RGB与点云数据的配准与对齐。


#  视差图空洞修复算法
    // https://blog.csdn.net/u013626386/article/details/54860969
    holefilling算法流程
    Input:disp –待修复视差图Output:dstDisp -修复后视差图
    Step1.找到disp中未计算深度的空点，空点集合设为Ω；
    Step2.遍历每一空点Ω(e)，根据其邻域信息δ(e)判断其是否处于空洞中，
          如果δ(e)内包含一半以上的深度有效像素(validPixel)，则认为其为空洞点；
    Step3.使用方形滤波器对空洞点进行填补，利益滤波器与有效像素的加权值补充空洞点处深度值，得到dstDisp；

    Step4.根据设定的迭代次数(iteration)来，置disp =dstDisp，并重复上述步骤，
           直至迭代完成，输出结果修复后的dstDisp，并据此生成深度数据。

    滤波器及权重设置:
          采用类似高斯权重设置的方法设置该滤波器权重，离目标像素越远的有效像素，
          对该空洞点视差值填补的贡献越小。
    filterSize滤波器大小选择:
          滤波器目前可选取5x5, 7x7, 9x9, 11x11.

    validPixel有效像素点数选择:
        例如：使用5x5的滤波器时，需要对空点周边的24个像素值进行深度有效像素点数量的判断，
              通常认为，空洞点周边应被有效点所环绕，所以此时有效像素点数至少设置为滤波器包含像素一半以上才合理，
              可设置为validPixel =12；使用其他size滤波器时，有效像素点数设置也应大于滤波器包含像素一半。

    iteration迭代次数选择
        针对不同的滤波器大小，收敛至较好效果时的迭代次数不一样，需要根据具体场景分析设定。
```c
void holefilling(Mat _dispSrc, Mat* _dispDst)
{
  int64 t = getTickCount();
  if (CV_8UC1 != _dispSrc.type())
  {
    _dispSrc.convertTo(_dispSrc, CV_8UC1);
  }
  Mat dispBw;
  threshold(_dispSrc, dispBw, dispMin, 255, THRESH_BINARY);
  dispBw.convetTo(dispBw, CV_32F, 1.0/255);
  Mat dispValid;
  _dispSrc.convertTo(dispValid, CV_32F);
  int margin = filterSize/2;
  Mat dispFilt = _dispSrc;
  
  for (int i = margin; i < dispBw.rows; i++)
  {
    for (int j = margin; j < dispBw.cols; j++)
    {
      if (0 == dispBw.at<float>(i, j))
      {
        Mat filtMat = dispBw(Range(i - margin, i + margin + 1), Range(j - margin, j + margin + 1));
        Scalar s = sum(filtMat);
        if (s[0] > validPixel)
        {
          Mat tmpWeight;
          multiply(filtMat, domainFilter, tmpWeight);
          Scalar s1 = sum(tmpWeight);
          Mat valid = dispValid(Range(i - margin, i + margin + 1), Range(j - margin, j + margin + 1));
          Mat final;
          multiply(tmpWeight, valid, final);
          Scalar s2 = sum(final);
          dispFilt.at<unsigned char>(i, j) = (unsigned char)(s2[0] / s1[0]);
        }
      }
      else
      {
        dispFilt.at<unsigned char>(i, j) = (unsigned char)(dispValid.at<unsigned char>(i, j));
      }
    }
  }
  *dispDst = dispFilt;
  t = getTickCount() - t;
  printf("Time Elapsed t : %fms\n", t1*1000/getTickFrequency);
}

```

# 视差图去噪
```c
static int depthDenoise(Mat _dispSrc, Mat* _dispDenoise)
{
  Mat contourBw;
  threshold(_dispSrc, contourBw, dispMin, 255, THRESH_BINARY);
  vector<vector<Point>> contours;
  findContours(contourBw, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
  double minArea = 10000*scale;
  for (int i = contours.size() - 1; i >= 0; i--)
  {
    double area = countourArea(contours[i]);
    if (area < minArea)
    {
      contours.erase(contours.begin() + i);
    }
  }
  Mat contourDisp(_dispSrc.size(), CV_8UC1, Scalar(0));
  drawContours(contourDisp, contours, Scalar(1), -1);
  multiply(_dispSrc, contourDisp, *_dispDenoise);
  return 0;
}
```
