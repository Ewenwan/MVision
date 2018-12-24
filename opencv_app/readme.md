# opencv 学习
[Hands-On-Algorithms-for-Computer-Vision 代码](https://github.com/PacktPublishing/Hands-On-Algorithms-for-Computer-Vision)

[OpenCV 3 Computer Vision Application Programming Cookbook, Third Edition](https://github.com/PacktPublishing/OpenCV3-Computer-Vision-Application-Programming-Cookbook-Third-Edition)

[计算机视觉OpenCV实现 csdn专栏](https://blog.csdn.net/column/details/computer-vision.html?&page=3)

[机器视觉与计算机视觉](https://www.cnblogs.com/ironstark/category/745953.html)

[opencv代码实验](https://github.com/Ewenwan/OpenCV_Test)

[LearnOpenCV.com 项目](https://github.com/Ewenwan/learnopencv)

[OpenCV学习笔记](https://blog.csdn.net/column/details/opencv-manual.html?&page=3)

[opencv大师 项目](https://github.com/Ewenwan/code)

[OpenCV + 数字成像](http://antkillerfarm.github.io/ai/2016/07/19/opencv.html)

[OpenCV-OpenGL--Reconstuction3d 三维重建](https://github.com/Ewenwan/OpenCV-OpenGL--Reconstuction3d)

[图像处理理论（一）——直方图, 二值化, 滤波基础](http://antkillerfarm.github.io/graphics/2016/04/30/graphics.html)

[图像处理理论（二）——形态学, 边缘检测, 图像金字塔](http://antkillerfarm.github.io/graphics/2016/06/30/graphics_2.html)

[图像处理理论（三）——双边滤波, Steerable滤波, Gabor滤波](http://antkillerfarm.github.io/graphics/2016/07/16/graphics_3.html)

[图像处理理论（四）——Schmid滤波, 霍夫变换, HOG, Haar, SIFT](http://antkillerfarm.github.io/graphics/2017/08/23/graphics_4.html)

[图像处理理论（五）——SIFT 图像格式 YUV & YCbCr & RGB ISP(Image Signal Processor)，图像信号处理器](http://antkillerfarm.github.io/graphics/2017/10/17/graphics_5.html)

[图像处理理论（六）——人脸识别算法Eigenface基于PCA, LBPLocal Binary Patterns）局部二值模式, Fisherface 基于LD(ALinear Discriminant Analysis，线性判别分析）](http://antkillerfarm.github.io/graphics/2017/12/25/graphics_6.html)

[人脸识别经典算法三：Fisherface（LDA）](https://blog.csdn.net/smartempire/article/details/23377385)

[图像处理理论（七）——Viola-Jones 积分图像, 经典目标跟踪算法（camshift、meanshift、Kalman filter、particle filter、Optical flow、TLD、KCF、Struck）, 从BOW(Bag-of-words词带模型)到SPM(Spatial Pyramid Matching,空间金字塔匹配), ILSVRC 2010考古](http://antkillerfarm.github.io/graphics/2018/04/03/graphics_7.html)



## window下安装
      1、系统环境变量设置
      动态链接库配置
      计算机 -> 右键属性 ->高级系统设置 -> 高级标签  -> 最下边 环境变量

      ->在系统变量 path 中添加 路径 在原有路径后添加冒号;

      -> x64 位添加两个 路径 ...opencv\build\x86\vc11\bin;...opencv\build\x64\vc11\bin;
      (vc8 对应vc2008 vc9对应vc2009  vc10对应vc2010   vc11对应vc2012的版本  vc12 对应2013 
      vc13：vs2014 
      vc14：vs2015   vc15:vs2017 )

      -> x86  添加一个 路径 ...opencv\build\x86\vc11\bin;

      2、vs工程包含目录

      文件 -> 新建 -> 项目  -> win32控制台应用程序
      -> 进入win32应用程序设置向导
      -> 附加选项空项目  项目名
      -> 在解决方案资源管理器的源文件下
      -> 右键单击 添加 新建项  添加c++文件


      -> 打开属性管理器 （视图 属性管理器）
      -> 在属性管理器里配置一次 相当于进行了通用配置
      -> 打开属性管理器 工作区
      -> 展开Debug|Win32
      -> 对 Microsoft.Cpp.Win32.userDirectories 右键 属性配置

      -> 首先在 通用属性 
      -> VC++目录
      -> 包含目录 下添加三个路径
      ..\opencv\build\include
      ..\opencv\build\include\opencv
      ..\opencv\build\include\opencv2

      -> 首先在 通用属性 
      -> VC++目录
      -> 库目录 下添加  一个路径
      ..\opencv\build\x64\vc14\lib


      -> 首先在 通用属性 
      -> 链接器
      -> 输入
      -> 附加依赖项
      库目录 下添加  上述库目录下的  lib文件
      带d的  
       opencv_world320d.lib/opencv_world340d.lib


      也可以将这下lib 复制到 window 操作系统目录下
      C:\Windows\SysWOW64   64位
      C:\Windows\System32   32位
## window下 找不到lib 文件
	可以用notePad++等文本编辑器打开类似 *.vcxproj的工程文件，查找到类似<AdditionalDependencies>libcocos2d.lib;opengl32.lib;glew32.lib;%(AdditionalDependencies)</AdditionalDependencies>的标签，删除不想关联的lib,如glew32.lib;字符串(或者 %(AdditionalDependencies) 标签 ；来自父级关联的lib), 然后保存，重新加载项目即可

## lib文件平台框架架构 与 目标架构不匹配 问题
	打开属性管理器 （视图->属性管理器(属性窗口)->属性页->）
	分别配置 平台下 x86  x64
	的 不同库目录（区别架构）

	然后 -> 配置管理器 -> 确定使用哪一种架构编译
	活动解决方案平台 x64  / x86

## linux下安装
安装依赖：

        sudo apt-get install build-essential libgtk2.0-dev libavcodec-dev libavformat-dev libjpeg62-dev libtiff4-dev cmake libswscale-dev libjasper-dev
        
[github源码安装](https://github.com/opencv/opencv.git)

      mkdir build
      cd build
      cmake
      make -j2
      sudo make install

      安装依赖项
      sudo apt-get install build-essential libgtk2.0-dev libvtk5-dev libjpeg-dev libtiff4-dev libjasper-dev libopenexr-dev libtbb-dev 

      编译依赖 sudo apt-get install build-essential
      必须     sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
      可选     sudo apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
      
      
       安装附加模块 opencv_contrib
          git clone https://github.com/opencv/opencv_contrib.git
          和 opencv 一起安装
          $ cd <opencv 编译文件夹
          $ cmake -DOPENCV_EXTRA_MODULES_PATH=<opencv_contrib 目录>/modules <opencv_source_directory(例如上一级 ..)>
          $ make -j3
	  sudo make install
    
[3.2安装参考 好好](https://github.com/CoderEugene/opencv3.2_CMake/tree/5175fc1b0a78e79831993ed4f5021bc2b0a656db)

	LAPACKE_H_PATH-NOTFOUND/lapacke.h: 没有那个文件或目录
	安装 lapacke : sudo apt-get install liblapacke-dev checkinstall
         修改文件：
	build/opencv_lapack.h ： 
	   #include "LAPACKE_H_PATH-NOTFOUND/lapacke.h"  >>>  #include "lapacke.h"
      
## 问题1 
      安装OpenCv 3.1的过程中要下载ippicv_linux_20151201，由于网络的原因，这个文件经常会下载失败。
      下载　  ippicv_linux_20151201
      http://blog.csdn.net/huangkangying/article/details/53406370

      下载　opencv-3.1.0.zip
      https://sourceforge.net/projects/opencvlibrary/files/opencv-unix/3.1.0/opencv-3.1.0.zip/download
      或者更简单一点，在确保MD5是808b791a6eac9ed78d32a7666804320e的情况下：
      在OpenCV源代码的根目录下创建目录： 
      拷贝　ippicv_linux_20151201到　opencv-3.1.0/3rdparty/ippicv/downloads/linux-808b791a6eac9ed78d32a7666804320e
      下


      注意3.0 后一些 不免费的库被独立出来
      需要单独编译
      https://github.com/opencv/opencv_contrib

## 问题2 
	ubuntu装opencv  error: ‘NppiGraphcutState‘ has not been declared
     解决方案：
	进入opencv-3.1.0/modules/cudalegacy/src/目录，修改graphcuts.cpp文件，将：
	#include "precomp.hpp"
	#if !defined (HAVE_CUDA) || defined (CUDA_DISABLER)
	改为
	#include "precomp.hpp"
	#if !defined (HAVE_CUDA) || defined (CUDA_DISABLER) || (CUDART_VERSION >= 8000)
	然后make编译就可以了 

      
 ## 编译
     进入主目录　
      mkdir build  //建立一个build目录，把cmake的文件都放着里边
      cd build     //进入build目录
      //cmake .. 
      cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local -D OPENCV_EXTRA_MODULES_PATH=<path to opencv_contrib/modules/> ..

      OPENCV_EXTRA_MODULES_PATH 就是用来指定要编译的扩展模块，后边加上刚下载的opencv_contrib模块的路径即可。

      make -j2
      sudo make install  安装


      github下载最新
      git clone https://github.com/opencv/opencv.git


[中文教程](http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/imgproc/opening_closing_hats/opening_closing_hats.html#morphology-2)



## 使用CMake编译
[CMake教程](https://cmake.org/cmake-tutorial/)

      CMakeLists.txt

      # CMake版本限制
      cmake_minimum_required(VERSION 2.8)
      # 工程名字
      project( DisplayImage )
      # 找opencv
      find_package( OpenCV REQUIRED )
      # 包含opencv
      include_directories( ${OpenCV_INCLUDE_DIRS} )

      #编译项目文件
      add_executable( displayImage displayImage.cpp )
      target_link_libraries( displayImage ${OpenCV_LIBS} )


      biicode：一个现代的 C 依赖管理器


## 创建算法对象实例

      // 好的方式
      Ptr<SomeAlgo> algo = makePtr<SomeAlgo>(...);
      Ptr<SomeAlgo> algo = SomeAlgo::create(...);
      // 不好的方式
      Ptr<SomeAlgo> algo = new SomeAlgo(...);
      SomeAlgo * algo = new SomeAlgo(...);
      SomeAlgo algo(...);
      Ptr<SomeAlgo> algo = Algorithm::create<SomeAlgo>("name");


      算法参数设置
      // 好的方式
      double clipLimit = clahe->getClipLimit();
      clahe->setClipLimit(clipLimit);
      // 不好的方式
      double clipLimit = clahe->getDouble("clipLimit");
      clahe->set("clipLimit", clipLimit);
      clahe->setDouble("clipLimit", clipLimit);


## 机器学习模块
      2.4 	                 3.0

      状态模式
      CvStatModel 	         cv::ml::StatModel             继承自 cv::Algorithm

      已下继承自 cv::ml::StatModel
      CvNormalBayesClassifier  cv::ml::NormalBayesClassifier 贝叶斯分类器
      CvKNearest 	         cv::ml::KNearest              K近邻
      CvSVM 	                 cv::ml::SVM                   支持向量机
      CvDTree 	         cv::ml::DTrees                决策树Decision Tres   


      CvGBTrees 	         Not implemented  
      // 梯度提升树(GBT,Gradient Boosted Trees,或称为梯度提升决策树)
      CvERTrees 	         Not implemented  【Extremely randomized trees，极端随机树】
      EM 	                 cv::ml::EM
       // 期望最大化（Expectation Maximization）算法
       // http://www.cvvision.cn/tag/em%e7%ae%97%e6%b3%95/
      CvANN_MLP 	         cv::ml::ANN_MLP               多层感知机    
      // 人工神经网络(Artificial Neural Networks),最典型的多层感知器(multi-layer perceptrons, MLP)     
      Not implemented 	 cv::ml::LogisticRegression    逻辑回归
      CvMLData 	         cv::ml::TrainData             训练数据集


      已下继承自  cv::ml::DTrees
      CvBoost 	         cv::ml::Boost       提升树算法（Boost） 
      CvRTrees 	         cv::ml::RTrees      Rtrees随机森林

      基于 ARM 的 Linux 系统的 交叉编译 
      依赖：
      1 Git;
      2 CMake >=2.6 ;
      3 ARM上的交叉编译工具 
	 gnueabi:
	     sudo apt-get install gcc-arm-linux-gnueabi
	 gnueabihf:
	     sudo apt-get install gcc-arm-linux-gnueabihf
	可见这两个交叉编译器适用于armel和armhf两个不同的架构,
	 armel和armhf这两种架构在对待浮点运算采取了不同的策略
	(有fpu（Float Point Unit，浮点运算单元）的arm才能支持这两种浮点运算策略)
	soft   : 不用fpu进行浮点计算，即使有fpu浮点运算单元也不用,而是使用软件模式。
	softfp : armel架构(对应的编译器为gcc-arm-linux-gnueabi)采用的默认值，
		 用fpu计算，但是传参数用普通寄存器传，这样中断的时候，
		 只需要保存普通寄存器，中断负荷小，但是参数需要转换成浮点的再计算。

	hard   : armhf架构(对应的编译器gcc-arm-linux-gnueabihf)采用的默认值，
		 用fpu计算，传参数也用fpu中的浮点寄存器传，省去了转换, 性能最好，但是中断负荷高。

      4 pkgconfig;
      5 Python 2.6 for host system;
      6 [optional] ffmpeg or libav development packages for armeabi(hf): libavcodec-dev, libavformat-dev, libswscale-dev;
      7 [optional] GTK+2.x or higher, including headers (libgtk2.0-dev) for armeabi(hf);
      8 [optional] libdc1394 2.x;
      9 [optional] libjpeg-dev, libpng-dev, libtiff-dev, libjasper-dev for armeabi(hf).


         cd ~/opencv/platforms/linux
         mkdir -p build_hardfp
         cd build_hardfp
         cmake -DCMAKE_TOOLCHAIN_FILE=../arm-gnueabi.toolchain.cmake ../../..

         make 

###########################################
## 【】The Core Functionality (core module)
########################################
## 　==========【1】显示图片===============

      #include <opencv2/core/core.hpp>
      #include <opencv2/imgcodecs.hpp>
      #include <opencv2/highgui/highgui.hpp>
      #include <iostream>
      #include <string>
      using namespace cv;
      using namespace std;
      int main( int argc, char** argv )
      {
          string imageName("../data/HappyFish.jpg"); // 图片文件名路径（默认值）
          if( argc > 1)
          {
              imageName = argv[1];//如果传递了文件 就更新
          }
          Mat image;//图片矩阵   string转char*
          image = imread(imageName.c_str(), IMREAD_COLOR); // 按源图片颜色读取
      //  I = imread( filename, IMREAD_GRAYSCALE);//灰度格式读取
          if( image.empty() ) // 检查图片是否读取成功
          {
              cout <<  "打不开图片 image" << std::endl ;
              return -1;
          }
          namedWindow( "Display window", WINDOW_AUTOSIZE ); // 创建一个窗口来显示图片.
          imshow( "Display window", image );                // 在窗口中显示图片.
          waitKey(0); // 等待窗口中的 鼠标点击 后 程序结束
          return 0;
      }


==================================

##  转换成灰度图
	 Mat gray_image;
	 cvtColor( image, gray_image, COLOR_BGR2GRAY );
================================

## 存储图片
	 imwrite( "../data/Gray_77.jpeg", gray_image );
         imwrite(filename, img);

########################

      矩阵复制
      Mat A, C;                          // 创建头header部分 
      A = imread(argv[1], IMREAD_COLOR); // here we'll know the method used (allocate matrix)
      Mat B(A);                          // 仅仅复制了头部分
      C = A;                             // 仅仅复制了头部分
      //指向了单个相同的矩阵，头部不同
      //修改 C/B会修改A本身

      复制部分区域（感兴趣区域）
      Mat D (A, Rect(10, 10, 100, 100) ); // 使用矩形范围 rectangle
      Mat E = A(Range::all(), Range(1,3)); // 使用行列范围 boundaries
      // 包好一个指向的计数值（每复制（header）一次 计数值增加一次，header消失一次，计数减少一次）


==================

      包含头和 矩阵一同赋值 clone() / copyTo(G)
      Mat F = A.clone();
      Mat G;
      A.copyTo(G);
      //修改F/G不会修改 A

      精度转换  8UC1 >>> 32FC1:
      src.convertTo(dst, CV_32F);


====================

      定义一个矩阵 并打印
      Mat M(2,2, CV_8UC3, Scalar(0,0,255));//
      cout << "M = " << endl << " " << M << endl << endl;
      数据类型
       CV_[存储位数（精度）][Signed有符号 or Unsigned无符号][Type Prefix]C[通道数]

      int sz[3] = {2,2,2};//各个维度尺寸
      Mat L(3,sz, CV_8UC(1), Scalar::all(0));


      // 特殊 矩阵
          Mat E = Mat::eye(4, 4, CV_64F);//单位矩阵  对角矩阵
          cout << "E = " << endl << " " << E << endl << endl;
          Mat O = Mat::ones(2, 2, CV_32F);//全1矩阵
          cout << "O = " << endl << " " << O << endl << endl;
          Mat Z = Mat::zeros(3,3, CV_8UC1);//全0矩阵
          cout << "Z = " << endl << " " << Z << endl << endl;

      //小矩阵的 特殊初始化方式
          Mat C = (Mat_<double>(3,3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
          cout << "C = " << endl << " " << C << endl << endl;
      // 复制部分行
          Mat RowClone = C.row(1).clone();//赋值第二行   -1, 5, -1,
          cout << "RowClone = " << endl << " " << RowClone << endl << endl;
      //随机初始化 一个矩阵 
          Mat R = Mat(3, 2, CV_8UC3);
          randu(R, Scalar::all(0), Scalar::all(255));//最小值 0   最大值 255




========================
## 图像像素值的访问
[参考](https://blog.csdn.net/baidu_19069751/article/details/50869561)

      //单通道获取
      Scalar intensity = img.at<uchar>(y, x);//行 列
      Scalar intensity = img.at<uchar>(Point(x, y));
       0 =< intensity.val[0] <= 255. 
      // 多通道获取   8u
      Vec3b intensity = img.at<Vec3b>(y, x);
      uchar blue = intensity.val[0];
      uchar green = intensity.val[1];
      uchar red = intensity.val[2];
      // 浮点型 像素值获取  32位
      Vec3f intensity = img.at<Vec3f>(y, x);
      float blue = intensity.val[0];
      float green = intensity.val[1];
      float red = intensity.val[2];

      // 修改
      img.at<uchar>(y, x) = 128;


      // 矩阵类型的点   calib3d module, 例如 投影点 projectPoints
      vector<Point2f> points;   32位
      //... fill the array
      Mat pointsMat = Mat(points);
      使用vector<Point2f> 转化成 mat类型 只有一行
      // 获取
      Point2f point = pointsMat.at<Point2f>(i, 0);

-------------------------------------------------------------------
## ====C语言的下标[]访问   存储连续区域(所有元素排成一行（内存中）)===
      int channels = I.channels();//通道数量

      int nRows = I.rows;//行数
      int nCols = I.cols * channels;//总列数 为列数×通道数 （多通道存储方式）
      // 判断是否连续存储
      if (I.isContinuous())
      {
        nCols *= nRows;//列数扩大
        nRows = 1;//存储连续区域(所有元素排成一行（内存中）)
      // 非连续的话 行与行之间是有存储间隙的
      }

      int i,j;
      uchar* p;
      for( i = 0; i < nRows; ++i)//每一行
      {// uchar 8u 位无符号格式访问  float 32位  double 64位
        p = I.ptr<uchar>(i);//每一行的数组首地址 
        for ( j = 0; j < nCols; ++j)
        {
            p[j] = table[p[j]];//p[j]为原来像素值（0~255）对应到 查找表中 阶梯像素值  替换
        }
      }
-----------------------------------------------------------

 ## ====安全（迭代器）遍历方式=========

      const int channels = I.channels();//通道数量
      switch(channels)
      {
      case 1://1通道  灰度图
        {
            MatIterator_<uchar> it, end;//每一大列 就一个灰度分量
            for( it = I.begin<uchar>(), end = I.end<uchar>(); it != end; ++it)
                *it = table[*it];//迭代器存储的是地址 *it得到像素值
            break;
        }
      case 3://3通道彩色图
        {
            MatIterator_<Vec3b> it, end;//每一大列 具有 BRG三个通道的分量 Vec3b
      // 如果你对彩色图片使用一个简单的uchar迭代器，那么你将只能访问B通道。 只获得 一大列的第一个元素 B通道的像素值
            for( it = I.begin<Vec3b>(), end = I.end<Vec3b>(); it != end; ++it)
            {
                (*it)[0] = table[(*it)[0]];
                (*it)[1] = table[(*it)[1]];
                (*it)[2] = table[(*it)[2]];
            }
        }
      }
-------------------------------------------------------

##  ======即时地址访问 .at<>(,)　不推荐使用如下这种方法

    const int channels = I.channels();//通道数量
    switch(channels)
    {
    case 1://1通道  灰度图
        {
            for( int i = 0; i < I.rows; ++i)
                for( int j = 0; j < I.cols; ++j )
                    I.at<uchar>(i,j) = table[I.at<uchar>(i,j)];//I.at<uchar>(i,j)为原始像素值
            break;
        }
    case 3://3通道彩色图
        {
      // 需要倍乘lookup table，使用at()将会十分的繁琐，需要不断输入数据类型和关键字。opencv引入了Mat_类型来解决这个问题
         Mat_<Vec3b> _I = I;// Mat 转换成 Mat_

         for( int i = 0; i < I.rows; ++i)//每一行
            for( int j = 0; j < I.cols; ++j )//每一大列（对于彩色图来说是 3个小列  BGR）
               {
                   _I(i,j)[0] = table[_I(i,j)[0]];
                   _I(i,j)[1] = table[_I(i,j)[1]];
                   _I(i,j)[2] = table[_I(i,j)[2]];
            }
         I = _I;//Mat_ 和Mat可以方便的互相转换 
         break;
        }
    }
-------------------------------------------------
================================
## 矩阵掩模操作  mask
=========================

      利用滤波器（增强、锐化）核(掩码矩阵) 对矩阵的各个像素重新计算，根据周围像素之间的关系

      掩模矩阵控制了旧图像当前位置以及周围位置像素对新图像当前位置像素值的影响力度。
      用数学术语讲，即我们自定义一个权重表。 

      ----------------------
      // 不属于传统的增强对比度，更像是锐化  自己利用 C语言的下标[]访问 实现
      // I(i,j)=5∗I(i,j)−[I(i−1,j)+I(i+1,j)+I(i,j−1)+I(i,j+1)]
      void Sharpen(const Mat& myImage, Mat& Result)//注意形参为 引用 可以避免拷贝
      {
      // assert 确保图片格式为 8为无符号 精度，应为下面的访问格式是 8u
      // 可以换成其他格式   检查图像位深，如果条件为假则抛出异常
          CV_Assert(myImage.depth() == CV_8U);  // accept only uchar images

    const int nChannels = myImage.channels();//通道数量 确定每一行究竟有多少子列
    Result.create(myImage.size(),myImage.type());//创建一个新的矩阵

    for(int j = 1 ; j < myImage.rows-1; ++j)//遍历 除去第一行和最后一行的每一行
    {// C语言的下标[]访问
        const uchar* previous = myImage.ptr<uchar>(j - 1);//上一行
        const uchar* current  = myImage.ptr<uchar>(j    );//本行
        const uchar* next     = myImage.ptr<uchar>(j + 1);//下一行

        uchar* output = Result.ptr<uchar>(j);//对应输出矩阵的一行

        for(int i= nChannels;i < nChannels*(myImage.cols-1); ++i)//每一大列
        {
            *output++ = saturate_cast<uchar>(5*current[i]
                         -current[i-nChannels] - current[i+nChannels] - previous[i] - next[i]);//对应通道的像素操作
      // saturate_cast 加入了溢出保护  类似下面的操作 8u 上下限是0~255
      // if(data<0)  
      //         data=0;  
      // else if(data>255)  
      //     data=255;
              }
          }
      // 最下面是边界4行都设置成0；
          Result.row(0).setTo(Scalar(0));
          Result.row(Result.rows-1).setTo(Scalar(0));
          Result.col(0).setTo(Scalar(0));
          Result.col(Result.cols-1).setTo(Scalar(0));
      }
----------------------------------------------------

      =========内部函数实现（估计利用了多线程，比较快）============
          Mat kern = (Mat_<char>(3,3) <<  0, -1,  0,
                                         -1,  5, -1,
                                          0, -1,  0);
          t = (double)getTickCount();
          filter2D(I, K, I.depth(), kern );// 输入Mat I 输出Mat K 位深，mask矩阵
      // 第五个参数可以设置mask矩阵的中心位置，第六个参数决定在操作不到的地方（边界）如何进行操作。

------------------------------------------------------------

      // Sobel 横向纵向边缘检测算子
      // Gx及Gy分别代表经横向及纵向边缘检测的图像
      Mat sobelx;
      Sobel(grey, sobelx, CV_32F, 1, 0);
      // 最大最小 梯度
      double minVal, maxVal;
      minMaxLoc(sobelx, &minVal, &maxVal); //find minimum and maximum intensities

------------------------------------------

      //图像 对比度和亮度增强  g(i,j)=α⋅f(i,j)+β
      // α 对比度(提高颜色差异)  β整体增加（亮度增加）
          Mat image = imread( argv[1] );
          Mat new_image = Mat::zeros( image.size(), image.type() );

          for( int y = 0; y < image.rows; y++ ) {
              for( int x = 0; x < image.cols; x++ ) {
                  for( int c = 0; c < 3; c++ ) {
                      new_image.at<Vec3b>(y,x)[c] =
                      saturate_cast<uchar>( alpha*( image.at<Vec3b>(y,x)[c] ) + beta );
                  }
              }
          }

--------------------------------------------------------

##  图像矩阵 傅里叶变换 也逆变换
    #include "opencv2/imgproc/imgproc.hpp"  
    #include "opencv2/objdetect/objdetect.hpp"  
    #include "opencv2/highgui/highgui.hpp"  
    #include <iostream>  
   
    using namespace std;  
    using namespace cv;  
      
    int main(){  
        Mat img(3, 3, CV_64FC2);  
        int k = 0;  
        for(unsigned i = 0; i < 3; i++){  
            for(unsigned j = 0; j < 3; j++){  
                img.at<std::complex<double> >(i,j) = k++;  //赋值
            }  
        }  
        for(unsigned i = 0; i < 3; i++){  
            for(unsigned j = 0; j < 3; j++){  
                cout << img.at<std::complex<double> >(i,j) << endl; //打印变换前 
            } 
 
        dft(img, img);  //傅里叶变换
        dft(img, img, DFT_SCALE|DFT_INVERSE);//傅里叶逆变换  
        for(unsigned i = 0; i < 3; i++){  
            for(unsigned j = 0; j < 3; j++){  
                cout << img.at<std::complex<double> >(i,j) << endl;  //打印变换后
            }  
        }  
    }

-------------------------------------

##########################################
# 图像处理 Image Processing (imgproc module)
##########################################
      图像处理待添加




##########################################
## High Level GUI and Media (highgui module)
##########################################

### OpenCV的视频输入和相似度测量

      #include <iostream> // for standard I/O
      #include <string>   // for strings
      #include <iomanip>  // for controlling float print precision
      #include <sstream>  // string to number conversion

      #include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
      #include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
      #include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O

      using namespace std;
      using namespace cv;


      // 需要先定义一个 VideoCapture 类的对象来打开和读取视频流。
      const string sourceReference = argv[1],sourceCompareWith = argv[2];

      VideoCapture captRefrnc(sourceReference);
      // 或者
      VideoCapture captUndTst;
      captUndTst.open(sourceCompareWith);

      如果使用整型数当参数的话，就可以将这个对象绑定到一个摄像机，
      将系统指派的ID号当作参数传入即可。例如你可以传入0来打开第一个摄像机，
      传入1打开第二个摄像机，以此类推。如果使用字符串当参数，

      VideoCapture cap(0);

      就会打开一个由这个字符串（文件名）指定的视频文件。


=======================================

      可以用 isOpened 函数来检查视频是否成功打开与否:

      if ( !captRefrnc.isOpened())
        {
        cout  << "Could not open reference " << sourceReference << endl;
        return -1;
        }

====================================

      因为视频流是连续的，所以你需要在每次调用 read 函数后及时保存图像或者直接使用重载的>>操作符。
      Mat frameReference, frameUnderTest;
      captRefrnc >> frameReference;
      captUndTst.open(frameUnderTest);
===================

      如果视频帧无法捕获（例如当视频关闭或者完结的时候），上面的操作就会返回一个空的 Mat 对象。
============
      if( frameReference.empty()  || frameUnderTest.empty())
      {
       // 退出程序
      exit(0);
      }
==================================

      在下面的例子里我们会先获得视频的尺寸和帧数。

      Size refS = Size((int) captRefrnc.get(CV_CAP_PROP_FRAME_WIDTH),
                       (int) captRefrnc.get(CV_CAP_PROP_FRAME_HEIGHT)),

      cout << "参考帧的  宽度 =" << refS.width << "  高度 =" << refS.height << endl;

=====================

      当你需要设置这些值的时候你可以调用 set 函数。
      captRefrnc.set(CV_CAP_PROP_POS_MSEC, 1.2);  // 跳转到视频1.2秒的位置
      captRefrnc.set(CV_CAP_PROP_POS_FRAMES, 10); // 跳转到视频的第10帧
      // 然后重新调用read来得到你刚刚设置的那一帧

===============================================
## 图像比较 - PSNR 

      当我们想检查压缩视频带来的细微差异的时候，就需要构建一个能够逐帧比较差视频差异的系统。
      最常用的比较算法是PSNR( Peak signal-to-noise ratio)。
      这是个使用“局部均值误差”来判断差异的最简单的方法，
      假设有这两幅图像：I1和I2，它们的行列数分别是i，j，有c个通道。

      均值sqre误差 MSE   1/(i*j*c) * (I1 - I2)^2
      255 is max pix

      psnr = 10.0*log10((255*255)/mse);


      double getPSNR(const Mat& I1, const Mat& I2)
      {
       Mat s1;
       absdiff(I1, I2, s1);       // |I1 - I2|
       s1.convertTo(s1, CV_32F);  // 不能在8位矩阵上做平方运算
       s1 = s1.mul(s1);           // |I1 - I2|^2

       Scalar s = sum(s1);        // 叠加每个通道的元素 
       double sse = s.val[0] + s.val[1] + s.val[2]; // 叠加所有通道

       if( sse <= 1e-10) // 如果值太小就直接等于0
           return 0;
       else
       {
           double  mse = sse /(double)(I1.channels() * I1.total());
           double psnr = 10.0*log10((255*255)/mse);
           return psnr;
       }
      }

      在考察压缩后的视频时，psnr 这个值大约在30到50之间，
      数字越大则表明压缩质量越好。如果图像差异很明显，就可能会得到15甚至更低的值。
      PSNR算法简单，检查的速度也很快。但是其呈现的差异值有时候和人的主观感受不成比例。

================================

      所以有另外一种称作 结构相似性 structural similarity  的算法做出了这方面的改进。
      图像比较 - SSIM
      建议你阅读一些关于SSIM算法的文献来更好的理解算法，
      Image quality assessment: From error visibility to structural similarity
      然而及时你直接看下面的源代码，应该也能建立一个不错的映像。

      Scalar getMSSIM( const Mat& i1, const Mat& i2)
      {
       const double C1 = 6.5025, C2 = 58.5225;
       /***************************** INITS **********************************/
       int d     = CV_32F;// 4 字节 float

       Mat I1, I2;
       i1.convertTo(I1, d);           // 不能在单字节像素上进行计算，范围不够。
       i2.convertTo(I2, d);


       /***********************初步计算 ******************************/
       Mat I2_2   = I2.mul(I2);        // I2^2
       Mat I1_2   = I1.mul(I1);        // I1^2
       Mat I1_I2  = I1.mul(I2);        // I1 * I2

       Mat mu1, mu2;   //src img 高斯平滑（模糊） ksize - 核大小 Size(11, 11)  
       GaussianBlur(I1, mu1, Size(11, 11), 1.5);
       GaussianBlur(I2, mu2, Size(11, 11), 1.5);
       Mat mu1_2   =   mu1.mul(mu1);
       Mat mu2_2   =   mu2.mul(mu2);
       Mat mu1_mu2 =   mu1.mul(mu2);

      // src*src img 高斯平滑（模糊） ksize - 核大小 Size(11, 11)  
      // diff
       Mat sigma1_2, sigma2_2, sigma12;
       GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
       sigma1_2 -= mu1_2;

       GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
       sigma2_2 -= mu2_2;

       GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
       sigma12 -= mu1_mu2;

       //=======公式=======================
       Mat t1, t2, t3;

       t1 = 2 * mu1_mu2 + C1;
       t2 = 2 * sigma12 + C2;
       t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

       t1 = mu1_2 + mu2_2 + C1;
       t2 = sigma1_2 + sigma2_2 + C2;
       t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

       Mat ssim_map;
       divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;

       Scalar mssim = mean( ssim_map ); // mssim = ssim_map的平均值
       return mssim;
      }

      这个操作会针对图像的每个通道返回一个相似度，取值范围应该在0到1之间，取值为1时代表完全符合。
      然而尽管SSIM能产生更优秀的数据，但是由于高斯模糊很花时间，
      所以在一个实时系统（每秒24帧）中，人们还是更多地采用PSNR算法。

      正是这个原因，最开始的源码里，我们用PSNR算法去计算每一帧图像，
      而仅当PSNR算法计算出的结果低于输入值的时候，用SSIM算法去验证。

======================================

### 用OpenCV创建视频
         使用OpenCV中的 VideoWriter 类就可以简单的完成创建视频的工作。
          如何用OpenCV创建一个视频文件
          用OpenCV能创建什么样的视频文件
          如何释放视频文件当中的某个颜色通道


### 视频文件的结构

      首先，你需要知道一个视频文件是什么样子的。每一个视频文件本质上都是一个容器，
      文件的扩展名只是表示容器格式（例如 avi ， mov ，或者 mkv ）而不是视频和音频的压缩格式。
      容器里可能会有很多元素，例如视频流，音频流和一些字幕流等等。
      这些流的储存方式是由每一个流对应的编解码器(codec)决定的。
      通常来说，音频流很可能使用 mp3 或 aac 格式来储存。
      而视频格式就更多些，通常是 XVID ， DIVX ， H264 或 LAGS (Lagarith Lossless Codec)等等。
      具体你能够使用的编码器种类可以在操作系统的编解码器列表里找到。

      OpenCV能够处理的视频只剩下 avi 扩展名的了。
      另外一个限制就是你不能创建超过2GB的单个视频，
      还有就是每个文件里只能支持一个视频流，
      不能将音频流和字幕流等其他数据放在里面。

      找一些专门处理视频的库例如 FFMpeg 或者更多的编解码器例如 
      HuffYUV ， CorePNG 和 LCL 。
      你可以先用OpenCV创建一个原始的视频流然后通过其他编解码器转换成其他格式
      并用VirtualDub 和 AviSynth 这样的软件去创建各种格式的视频文件.

      要创建一个视频文件，你需要创建一个 VideoWriter 类的对象。
      可以通过构造函数里的参数或者在其他合适时机使用 open 函数来打开，两者的参数都是一样的：

      我们会使用输入文件名+通道名( argv[2][0])+avi来创建输出文件名。

      const string source      = argv[1];            // 原视频文件名
      string::size_type pAt = source.find_last_of('.');   // 找到扩展名的位置
      const string NAME = source.substr(0, pAt) + argv[2][0] + ".avi";   // 创建新的视频文件名

      // 编解码器
      VideoCapture inputVideo(source);                                   // 打开视频输入
      int ex = static_cast<int>(inputVideo.get(CV_CAP_PROP_FOURCC));     // 得到编码器的int表达式

      OpenCV内部使用这个int数来当作第二个参数，
      这里会使用两种方法来将这个 整型数转换为字符串：位操作符和联合体。
      前者可以用&操作符并进行移位操作，以便从int里面释放出字符：
      // 位操作符
      char EXT[] = {ex & 0XFF , (ex & 0XFF00) >> 8,(ex & 0XFF0000) >> 16,(ex & 0XFF000000) >> 24, 0};

      // 联合体 使用 联合体 来做到:
          union { int v; char c[5];} uEx ;
          uEx.v = ex;  // 通过联合体来分解字符串
          uEx.c[4]='\0';
          反过来，当你需要修改视频的格式时，你都需要修改FourCC码，
          而更改ForCC的时候都要做逆转换来指定新类型。如果你已经知道这个FourCC编码的具体字符的话，
          可以直接使用 *CV_FOURCC* 宏来构建这个int数:

      // 输出视频的帧率，也就是每秒需要绘制的图像数 inputVideo.get(CV_CAP_PROP_FPS)
      VideoWriter outputVideo;
      Size S = Size((int) inputVideo.get(CV_CAP_PROP_FRAME_WIDTH),    //获取输入尺寸
                    (int) inputVideo.get(CV_CAP_PROP_FRAME_HEIGHT));

      outputVideo.open(NAME , ex, inputVideo.get(CV_CAP_PROP_FPS),S, true);
      // 然后，最好使用 isOpened() 函数来检查是不是成功打开。

      outputVideo.write(res);  //或者
      outputVideo << res;

### 分割合并图像通道
      要“释放”出某个通道，又要保持视频为彩色，实际上也就意味着要把未选择的通道都设置为全0。
      这个操作既可以通过手工扫描整幅图像来完成，又可以通过分离通道然后再合并来做到，
      具体操作时先分离三通道图像为三个不同的单通道图像，
      然后再将选定的通道与另外两张大小和类型都相同的黑色图像合并起来。

      split(src, spl);                            // 分离三个通道
      for( int i =0; i < 3; ++i)
         if (i != channel)
            spl[i] = Mat::zeros(S, spl[0].type());//创建相同大小的黑色图像
      merge(spl, res);                            //重新合并

====================================
### 用GDAL读地理栅格文件

      GDAL(Geospatial Data Abstraction Library)是一个在X/MIT许可协议下的开源栅格空间数据转换库
      它利用抽象数据模型来表达所支持的各种文件格式。它还有一系列命令行工具来进行数据转换和处理。
      OGR是GDAL项目的一个分支，功能与GDAL类似，只不过它提供对矢量数据的支持。

[GDAL/OGR 快速入门](http://live.osgeo.org/zh/quickstart/gdal_quickstart.html)

       说下我的构思吧，opencv库里有很多关于数字图像处理的函数，
      但是它却局限于遥感图像的读取，而GDAL却对遥感影像的读取支持的很好，
      所有我想用GDAL将遥感影像读入，转成矩阵，传递到opencv中，然后使用opencv的函数来处理，
      不知道这个想法怎么样，还希望各位能指点。

    cv::Mat image = cv::imread(argv[1], cv::IMREAD_LOAD_GDAL | cv::IMREAD_COLOR );
    // load the dem model
    cv::Mat dem = cv::imread(argv[2], cv::IMREAD_LOAD_GDAL | cv::IMREAD_ANYDEPTH );
    // create our output products
    cv::Mat output_dem(   image.size(), CV_8UC3 );
    cv::Mat output_dem_flood(   image.size(), CV_8UC3 );

====================================
#### KinectRGBD Using Kinect and other OpenNI compatible depth sensors 

      VideoCapture capture(0); //  VideoCapture capture( CAP_OPENNI );

      capture.set( CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CAP_OPENNI_VGA_30HZ );
      cout << "FPS    " << capture.get( CAP_OPENNI_IMAGE_GENERATOR+CAP_PROP_FPS ) << endl;

      for(;;)
      {
          Mat depthMap;
          Mat bgrImage;
          capture.grab();
          capture.retrieve( depthMap, CAP_OPENNI_DEPTH_MAP );
          capture.retrieve( bgrImage, CAP_OPENNI_BGR_IMAGE );
          if( waitKey( 30 ) >= 0 )
              break;
      }

==========================================
#### 英特尔３Ｄ传感器　Intel
      Using Creative Senz3D and other Intel Perceptual Computing SDK compatible depth sensors 


      VideoCapture capture(CAP_INTELPERC);

      VideoCapture capture( CAP_INTELPERC );
      capture.set( CAP_INTELPERC_DEPTH_GENERATOR | CAP_PROP_INTELPERC_PROFILE_IDX, 0 );
      cout << "FPS    " << capture.get( CAP_INTELPERC_DEPTH_GENERATOR+CAP_PROP_FPS ) << endl;


      for(;;)
      {
          Mat depthMap;
          Mat image;
          Mat irImage;
          capture.grab();
          capture.retrieve( depthMap, CAP_INTELPERC_DEPTH_MAP );
          capture.retrieve(    image, CAP_INTELPERC_IMAGE );
          capture.retrieve(  irImage, CAP_INTELPERC_IR_MAP);
          if( waitKey( 30 ) >= 0 )
              break;
      }




##########################################
## 三维重建　Camera calibration and 3D reconstruction (calib3d module)
##########################################

### 【1】 ply格式数据 储存立体扫描结果的三维数值

	ply文件不支持中文格式的文件名字,所以在使用过程中避免使用中文来命名。

	https://www.cnblogs.com/liangliangdetianxia/p/4000295.html
	PLY是一种电脑档案格式，全名为多边形档案（Polygon File Format）或 
	斯坦福三角形档案（Stanford Triangle Format）。 

	该格式主要用以储存立体扫描结果的三维数值，透过多边形片面的集合描述三维物体，
	与其他格式相较之下这是较为简单的方法。它可以储存的资讯包含颜色、
	透明度、表面法向量、材质座标与资料可信度，并能对多边形的正反两面设定不同的属性。

	在档案内容的储存上PLY有两种版本，分别是纯文字（ASCII）版本与二元码（binary）版本，
	其差异在储存时是否以ASCII编码表示元素资讯。

	每个PLY档都包含档头（header），用以设定网格模型的“元素”与“属性”，
	以及在档头下方接着一连串的元素“数值资料”。
	一般而言，网格模型的“元素”
	就是顶点（vertices）、
	面（faces），另外还可能包含有
	边（edges）、
      
      (定点数　－　边数　－　面数　＝２)
	深度图样本（samples of range maps）与
	三角带（triangle strips）等元素。
	无论是纯文字与二元码的PLY档，档头资讯都是以ASCII编码编写，
	接续其后的数值资料才有编码之分。PLY档案以此行：

	ply     开头作为PLY格式的识别。接着第二行是版本资讯，目前有三种写法：

	format ascii 1.0
	format binary_little_endian 1.0
	format binary_big_endian 1.0
	       // 其中ascii, binary_little_endian, binary_big_endian是档案储存的编码方式，
	       // 而1.0是遵循的标准版本（现阶段仅有PLY 1.0版）。

	comment This is a comment!   // 使用'comment'作为一行的开头以编写注解
	comment made by anonymous
	comment this file is a cube

	element vertex 8    //  描述元素  element <element name> <number in file>   8个顶点
		            //  以下 接续的6行property描述构成vertex元素的数值字段顺序代表的意义，及其资料形态。
	property float32 x  //  描述属性  property <data_type> <property name 1>
	property float32 y  //  每个顶点使用3个 float32 类型浮点数（x，y，z）代表点的坐标
	property float32 z

	property uchar blue // 使用3个unsigned char代表顶点颜色，颜色顺序为 (B, G, R)
	property uchar green
	property uchar red

	element face 12       
	property list uint8 int32 vertex_index
	 			// 12 个面(6*2)   另一个常使用的元素是面。
				// 由于一个面是由3个以上的顶点所组成，因此使用一个“顶点列表”即可描述一个面, 
				// PLY格式使用一个特殊关键字'property list'定义之。 
	end_header              // 最后，标头必须以此行结尾：

	// 档头后接着的是元素资料（端点座标、拓朴连结等）。在ASCII格式中各个端点与面的资讯都是以独立的一行描述

	0 0 0                   // 8个顶点 索引 0~7
	0 25.8 0
	18.9 0 0
	18.9 25.8 0
	0 0 7.5
	0 25.8 7.5
	18.9 0 7.5
	18.9 25.8 7.5

	3 5 1 0            // 前面的3表示3点表示的面   有的一个面 它用其中的三个点 表示了两次  6*2=12
	3 5 4 0            // 后面是上面定点的 索引 0~7
	3 4 0 2
	3 4 6 2
	3 7 5 4
	3 7 6 4
	3 3 2 1
	3 1 2 0
	3 5 7 1
	3 7 1 3
	3 7 6 3
	3 6 3 2


### 【2】由简单的 长方体 顶点  面 描述的ply文件 和 物体的彩色图像 生产 物体的三维纹理模型文件


	【a】手动指定 图像中 物体顶点的位置（得到二维像素值位置）
		ply文件 中有物体定点的三维坐标

		由对应的 2d-3d点对关系
		u
		v  =  K × [R t] X
		1               Y
				Z
				1
		K 为图像拍摄时 相机的内参数

		世界坐标中的三维点(以文件中坐标为(0,0,0)某个定点为世界坐标系原点)
		经过 旋转矩阵R  和平移向量t 变换到相机坐标系下
		在通过相机内参数 变换到 相机的图像平面上

	【b】由 PnP 算法可解的 旋转矩阵R  和平移向量t 

	【c】把从图像中得到的纹理信息 加入到 物体的三维纹理模型中

		在图像中提取特征点 和对应的描述子
		利用 内参数K 、 旋转矩阵R  和平移向量t  反向投影到三维空间
		   标记 该反投影的3d点 是否在三维物体的 某个平面上

	【d】将 2d-3d点对 、关键点 以及 关键点描述子 存入物体的三维纹理模型中

### 【3】纹理物体实时位姿估计

	【a】 读取网格数据文件 和 三维纹理数据文件 获取3d描述数据库 
	      设置特征检测器 描述子提取器 描述子匹配器 
	【b】 场景中提取特征点获取描述子  在　模型库(3d描述子数据库 3d points +description )中匹配 得到　匹配点 
	【c】 获取场景图片中　和　模型库中匹配的2d-3d点对
	【d】 使用PnP + Ransac进行姿态估计  2d-3d点对求解 [R t]
	【e】 显示 PNP求解后　得到的内点
	【f】 使用线性卡尔曼滤波去除错误的姿态估计
	【g】 更新pnp 的　变换矩阵
	【h】显示位姿　轴 帧率 可信度 
	【i】显示调试数据 


############################################
## 2D Features framework (feature2d module) 特征检测描述模块
###########################################

### 【1】Harris角点  cornerHarris()

	算法基本思想是使用一个固定窗口在图像上进行任意方向上的滑动，
	比较滑动前与滑动后两种情况，窗口中的像素灰度变化程度，
	如果存在任意方向上的滑动，都有着较大灰度变化，
	那么我们可以认为该窗口中存在角点。

	图像特征类型:
	    边缘 （Edges   物体边缘）
	    角点 (Corners  感兴趣关键点（ interest points） 边缘交叉点 )
	    斑点(Blobs  感兴趣区域（ regions of interest ） 交叉点形成的区域 )


	为什么角点是特殊的?
	    因为角点是两个边缘的连接点(交点)它代表了两个边缘变化的方向上的点。
	    图像梯度有很高的变化。这种变化是可以用来帮助检测角点的。


	G = SUM( W(x,y) * [I(x+u, y+v) -I(x,y)]^2 )

	 [u,v]是窗口的偏移量
	 (x,y)是窗口内所对应的像素坐标位置，窗口有多大，就有多少个位置
	 w(x,y)是窗口函数，最简单情形就是窗口内的所有像素所对应的w权重系数均为1。
		           设定为以窗口中心为原点的二元正态分布

	泰勒展开（I(x+u, y+v) 相当于 导数）
	G = SUM( W(x,y) * [I(x,y) + u*Ix + v*Iy - I(x,y)]^2)
	  = SUM( W(x,y) * (u*u*Ix*Ix + v*v*Iy*Iy))
	  = SUM(W(x,y) * [u v] * [Ix^2   Ix*Iy] * [u 
		                  Ix*Iy  Iy^2]     v] )
	  = [u v]  * SUM(W(x,y) * [Ix^2   Ix*Iy] ) * [u  应为 [u v]为常数 可以拿到求和外面
		                   Ix*Iy  Iy^2]      v]    
	  = [u v] * M * [u
		         v]
	则计算 det(M)   矩阵M的行列式的值  取值为一个标量，写作det(A)或 | A |  矩阵表示的空间的单位面积/体积/..
	       trace(M) 矩阵M的迹         矩阵M的对角线元素求和，用字母T来表示这种算子，他的学名叫矩阵的迹

	M的两个特征值为 lamd1  lamd2

	det(M)    = lamd1 * lamd2
	trace(M) = lamd1 + lamd2

	R = det(M)  -  k*(trace(M))^2 
	其中k是常量，一般取值为0.04~0.06，
	R大于一个阈值的话就认为这个点是 角点

	因此可以得出下列结论：
	>特征值都比较大时，即窗口中含有角点
	>特征值一个较大，一个较小，窗口中含有边缘
	>特征值都比较小，窗口处在平坦区域

	https://blog.csdn.net/woxincd/article/details/60754658

### 【2】 Shi-Tomasi 算法 goodFeaturesToTrack()
	是Harris 算法的改进。
	Harris 算法最原始的定义是将矩阵 M 的行列式值与 M 的迹相减，
	再将差值同预先给定的阈值进行比较。

	后来Shi 和Tomasi 提出改进的方法，
	若两个特征值中较小的一个大于最小阈值，则会得到强角点。
	M 对角化>>> M的两个特征值为 lamd1  lamd2

	R = mini(lamd1,lamd2) > 阈值 认为是角点


### 【3】FAST角点检测算法  ORB特征检测中使用的就是这种角点检测算法
        FAST(src_gray, keyPoints,thresh);
	周围区域灰度值 都较大 或 较小

        若某像素与其周围邻域内足够多的像素点相差较大，则该像素可能是角点。

	该算法检测的角点定义为在像素点的周围邻域内有足够多的像素点与该点处于不同的区域。
	应用到灰度图像中，即有足够多的像素点的灰度值大于该点的灰度值或者小于该点的灰度值。

	p点附近半径为3的圆环上的16个点，
	一个思路是若其中有连续的12个点的灰度值与p点的灰度值差别超过某一阈值，
	则可以认为p点为角点。

	这一思路可以使用机器学习的方法进行加速。
	对同一类图像，例如同一场景的图像，可以在16个方向上进行训练，
	得到一棵决策树，从而在判定某一像素点是否为角点时，
	不再需要对所有方向进行检测，
	而只需要按照决策树指定的方向进行2-3次判定即可确定该点是否为角点。

	std::vector<KeyPoint> keyPoints; 
	//fast.detect(src_gray, keyPoints);  // 检测角点
	FAST(src_gray, keyPoints,thresh);

### 【4】 使用cornerEigenValsAndVecs()函数和cornerMinEigenVal()函数自定义角点检测函数。
	过自定义 R的计算方法和自适应阈值 来定制化检测角点

	计算 M矩阵
	计算判断矩阵 R

	设置自适应阈值

	阈值大小为 判断矩阵 最小值和最大值之间 百分比
	阈值为 最小值 + （最大值-最小值）× 百分比
	百分比 = myHarris_qualityLevel/max_qualityLevel

## 【5】亚像素级的角点检测
	如果对角点的精度有更高的要求，可以用cornerSubPix()函数将角点定位到子像素，
	从而取得亚像素级别的角点检测效果。

	使用cornerSubPix()函数在goodFeaturesToTrack()的角点检测基础上将角点位置精确到亚像素级别

	常见的亚像素级别精准定位方法有三类：
		1. 基于插值方法
		2. 基于几何矩寻找方法
		3. 拟合方法 - 比较常用

	拟合方法中根据使用的公式不同可以分为
		1. 高斯曲面拟合与
		2. 多项式拟合等等。

	以高斯拟合为例:

		窗口内的数据符合二维高斯分布
		Z = n / (2 * pi * 西格玛^2) * exp(-P^2/(2*西格玛^2))
		P = sqrt( (x-x0)^2 + (y-y0)^2)

		x,y   原来 整数点坐标
		x0,y0 亚像素补偿后的 坐标 需要求取

		ln(Z) = n0 + x0/(西格玛^2)*x +  y0/(西格玛^2)*y - 1/(2*西格玛^2) * x^2 - 1/(2*西格玛^2) * y^2
			n0 +            n1*x + n2*y +             n3*x^2 +              n3 * y^2
		   
		对窗口内的像素点 使用最小二乘拟合 得到上述 n0 n1 n2 n3
		  则 x0 = - n1/(2*n3)
		     y0 = - n2/(2*n3)


	// SURF放在另外一个包的xfeatures2d里边了，在github.com/Itseez/opencv_contrib 这个仓库里。
	// 按说明把这个仓库编译进3.0.0就可以用了。
	opencv2中SurfFeatureDetector、SurfDescriptorExtractor、BruteForceMatcher在opencv3中发生了改变。
	具体如何完成特征点匹配呢？示例如下：

	//寻找关键点
	int minHessian = 700;
	Ptr<SURF>detector = SURF::create(minHessian);
	detector->detect( srcImage1, keyPoint1 );
	detector->detect( srcImage2, keyPoints2 );

	//绘制特征关键点
	Mat img_keypoints_1; Mat img_keypoints_2;
	drawKeypoints( srcImage1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
	drawKeypoints( srcImage2, keypoints_2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

	//显示效果图
	imshow("特征点检测效果图1", img_keypoints_1 );
	imshow("特征点检测效果图2", img_keypoints_2 );

	//计算特征向量
	Ptr<SURF>extractor = SURF::create();
	Mat descriptors1, descriptors2;
	extractor->compute( srcImage1, keyPoint1, descriptors1 );
	extractor->compute( srcImage2, keyPoints2, descriptors2 );

	//使用BruteForce进行匹配
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
	std::vector< DMatch > matches;
	matcher->match( descriptors1, descriptors2, matches );

	//绘制从两个图像中匹配出的关键点
	Mat imgMatches;
	drawMatches( srcImage1, keyPoint1, srcImage2, keyPoints2, matches, imgMatches );//进行绘制
	//显示
	imshow("匹配图", imgMatches );


	3.x的特征检测:

	    算法：SURF,SIFT,BRIEF,FREAK 
	    类：cv::xfeatures2d::SURF

	    cv::xfeatures2d::SIFT
	    cv::xfeatures::BriefDescriptorExtractor
	    cv::xfeatures2d::FREAK
	    cv::xfeatures2d::StarDetector

	    需要进行以下几步

	    加入opencv_contrib
	    包含opencv2/xfeatures2d.hpp
	    using namepsace cv::xfeatures2d
	    使用create(),detect(),compute(),detectAndCompute()


## 【6】斑点检测原理 SIFT  SURF
 
### SIFT定位算法关键步骤的说明 
[SIFT原理](http://www.cnblogs.com/ronny/p/4028776.html)

[SIFT特征提取分析](https://blog.csdn.net/x_r_su/article/details/54848590)

[SIFT原理与源码分析](https://blog.csdn.net/xiaowei_cqu/article/details/8069548)

	该算法大概可以归纳为三步：1）高斯差分金字塔的构建；2）特征点的搜索；3）特征描述。

	    DoG尺度空间构造（Scale-space extrema detection）
	    关键点搜索与定位（Keypoint localization）
	    方向赋值（Orientation assignment）
	    关键点描述（Keypoint descriptor）
	    OpenCV实现：特征检测器FeatureDetector
	    SIFT中LoG和DoG的比较

### SURF算法与源码分析、上  加速鲁棒特征（SURF）
[SURF](www.cnblogs.com/ronny/p/4045979.html)
[SURF算法及源码分析](https://blog.csdn.net/x_r_su/article/details/54848527)

[参考](https://blog.csdn.net/abcjennifer/article/details/7639681)

	通过在不同的尺度上利用积分图像可以有效地计算出近似Harr小波值，
	简化了二阶微分模板的构建，搞高了尺度空间的特征检测的效率。
	在以关键点为中心的3×3×3像素邻域内进行非极大值抑制，
	最后通过对斑点特征进行插值运算，完成了SURF特征点的精确定位。

	而SURF特征点的描述，则也是充分利用了积分图，用两个方向上的Harr小波模板来计算梯度，
	然后用一个扇形对邻域内点的梯度方向进行统计，求得特征点的主方向。

## 【7】二进制字符串特征描述子

	注意到在两种角点检测算法里，我们并没有像SIFT或SURF那样提到特征点的描述问题。
	事实上，特征点一旦检测出来，无论是斑点还是角点描述方法都是一样的，
	可以选用你认为最有效的特征描述子。

	比较有代表性的就是 浮点型特征描述子(sift、surf 欧氏距离匹配)和
	二进制字符串特征描述子 (字符串汉明距离 匹配 )。

###  1 浮点型特征描述子(sift、surf 欧氏距离匹配):

		像SIFT与SURF算法里的，用梯度统计直方图来描述的描述子都属于浮点型特征描述子。
		但它们计算起来，算法复杂，效率较低.

		SIFT特征采用了128维的特征描述子，由于描述子用的浮点数，所以它将会占用512 bytes的空间。
		类似地，对于SURF特征，常见的是64维的描述子，它也将占用256bytes的空间。
		如果一幅图像中有1000个特征点（不要惊讶，这是很正常的事），
		那么SIFT或SURF特征描述子将占用大量的内存空间，对于那些资源紧张的应用，
		尤其是嵌入式的应用，这样的特征描述子显然是不可行的。而且，越占有越大的空间，
		意味着越长的匹配时间。

		我们可以用PCA、LDA等特征降维的方法来压缩特征描述子的维度。
		还有一些算法，例如LSH，将SIFT的特征描述子转换为一个二值的码串，
		然后这个码串用汉明距离进行特征点之间的匹配。这种方法将大大提高特征之间的匹配，
		因为汉明距离的计算可以用异或操作然后计算二进制位数来实现，在现代计算机结构中很方便。

### 2 二进制字符串特征描述子 (字符串汉明距离 匹配 ):

		如BRIEF。后来很多二进制串描述子ORB，BRISK，FREAK等都是在它上面的基础上的改进。

	【A】 BRIEF:  Binary Robust Independent Elementary Features
	  http://www.cnblogs.com/ronny/p/4081362.html

		它需要先平滑图像，然后在特征点周围选择一个Patch，在这个Patch内通过一种选定的方法来挑选出来nd个点对。
		然后对于每一个点对(p,q)，我们来比较这两个点的亮度值，
		如果I(p)>I(q)则这个点对生成了二值串中一个的值为1，
		如果I(p)<I(q)，则对应在二值串中的值为-1，否则为0。
		所有nd个点对，都进行比较之间，我们就生成了一个nd长的二进制串。

		对于nd的选择，我们可以设置为128，256或512，这三种参数在OpenCV中都有提供，
		但是OpenCV中默认的参数是256，这种情况下，非匹配点的汉明距离呈现均值为128比特征的高斯分布。
		一旦维数选定了，我们就可以用汉明距离来匹配这些描述子了。

		对于BRIEF，它仅仅是一种特征描述符，它不提供提取特征点的方法。
		所以，如果你必须使一种特征点定位的方法，如FAST、SIFT、SURF等。
		这里，我们将使用CenSurE方法来提取关键点，对BRIEF来说，CenSurE的表现比SURF特征点稍好一些。
		总体来说，BRIEF是一个效率很高的提取特征描述子的方法，
		同时，它有着很好的识别率，但当图像发生很大的平面内的旋转。


		关于点对的选择：

		设我们在特征点的邻域块大小为S×S内选择nd个点对(p,q)，Calonder的实验中测试了5种采样方法：

			1）在图像块内平均采样；
			2）p和q都符合(0,1/25 * S^2)的高斯分布；
			3）p符合(0,1/25 * S^2)的高斯分布，而q符合(0,1/100 *S^2)的高斯分布；
			4）在空间量化极坐标下的离散位置随机采样
			5）把p固定为(0,0)，q在周围平均采样

	【B】BRISK算法
	 	BRISK算法在特征点检测部分没有选用FAST特征点检测，而是选用了稳定性更强的AGAST算法。
		在特征描述子的构建中，BRISK算法通过利用简单的像素灰度值比较，
		进而得到一个级联的二进制比特串来描述每个特征点，这一点上原理与BRIEF是一致的。
		BRISK算法里采用了邻域采样模式，即以特征点为圆心，构建多个不同半径的离散化Bresenham同心圆，
		然后再每一个同心圆上获得具有相同间距的N个采样点。	

	【C】ORB算法 Oriented FAST and Rotated BRIEF
 	   http://www.cnblogs.com/ronny/p/4083537.html
		ORB算法使用FAST进行特征点检测，然后用BREIF进行特征点的特征描述，
		但是我们知道BRIEF并没有特征点方向的概念，所以ORB在BRIEF基础上引入了方向的计算方法，
		并在点对的挑选上使用贪婪搜索算法，挑出了一些区分性强的点对用来描述二进制串。

          	通过构建高斯金字塔 来实现 尺度不变性
	  	利用灰度质心法     来实现 记录方向
			灰度质心法假设角点的灰度与质心之间存在一个偏移，这个向量可以用于表示一个方向。
	【D】FREAK算法 Fast Retina KeyPoint，即快速视网膜关键点。
	 	根据视网膜原理进行点对采样，中间密集一些，离中心越远越稀疏。
		并且由粗到精构建描述子，穷举贪婪搜索找相关性小的。
		42个感受野，一千对点的组合，找前512个即可。这512个分成4组，
		前128对相关性更小，可以代表粗的信息，后面越来越精。匹配的时候可以先看前16bytes，
		即代表精信息的部分，如果距离小于某个阈值，再继续，否则就不用往下看了。

	【E】Local binary pattern (LBP) 局部二值模式 
   		LBP 的算法非常简单，简单来说，就是对图像中的某一像素点的灰度值
		与其邻域的像素点的灰度值做比较，如果邻域像素值比该点大，则赋为1，反之，则赋为0，
		这样从左上角开始，可以形成一个bit chain，
		然后将该 bit chain 转换为一个十进制的数。
		
		我们可以看到，R 表示邻域的半径，P 表示邻域像素的个数，或者bit chain 的长度，
		如果邻域的半径为1，则邻域的像素个数为8， bit chain 的长度为8，
		如果邻域半径为2，则邻域的像素个数为16，bit chain 的长度为16，
		邻域半径为3, 邻域的像素个数为24，bit chain 长度为24。
                





## 【8】KAZE非线性尺度空间 特征

	基于非线性尺度空间的KAZE特征提取方法以及它的改进AKATE
[简介](https://blog.csdn.net/chenyusiyuan/article/details/8710462)

	KAZE是日语‘风’的谐音，寓意是就像风的形成是空气在空间中非线性的流动过程一样，
	KAZE特征检测是在图像域中进行非线性扩散处理的过程。

	传统的SIFT、SURF等特征检测算法都是基于 线性的高斯金字塔 进行多尺度分解来消除噪声和提取显著特征点。
	但高斯分解是牺牲了局部精度为代价的，容易造成边界模糊和细节丢失。

	非线性的尺度分解有望解决这种问题，但传统方法基于正向欧拉法（forward Euler scheme）
	求解非线性扩散（Non-linear diffusion）方程时迭代收敛的步长太短，耗时长、计算复杂度高。

	由此，KAZE算法的作者提出采用加性算子分裂算法(Additive Operator Splitting, AOS)
	来进行非线性扩散滤波，可以采用任意步长来构造稳定的非线性尺度空间。


### 非线性扩散滤波
		Perona-Malik扩散方程:
			具体地，非线性扩散滤波方法是将图像亮度（L）在不同尺度上的变化视为某种形式的
			流动函数（flow function）的散度（divergence），可以通过非线性偏微分方程来描述：
		AOS算法:
			由于非线性偏微分方程并没有解析解，一般通过数值分析的方法进行迭代求解。
			传统上采用显式差分格式的求解方法只能采用小步长，收敛缓慢。

	KAZE特征检测与描述

	KAZE特征的检测步骤大致如下：
	1) 首先通过AOS算法和可变传导扩散（Variable  Conductance  Diffusion）（[4,5]）方法来构造非线性尺度空间。
	2) 检测感兴趣特征点，这些特征点在非线性尺度空间上经过尺度归一化后的Hessian矩阵行列式是局部极大值（3×3邻域）。
	3) 计算特征点的主方向，并且基于一阶微分图像提取具有尺度和旋转不变性的描述向量。

	特征点检测
	KAZE的特征点检测与SURF类似，是通过寻找不同尺度归一化后的Hessian局部极大值点来实现的。

#############################################################
## 视频分析模块　　Video analysis (video module)
##############################################################

### 背景减除 Background subtraction (BS)

	背景减除在很多基础应用中占据很重要的角色。
	列如顾客统计，使用一个静态的摄像头来记录进入和离开房间的人数，
	或者交通摄像头，需要提取交通工具的信息等。
	我们需要把单独的人或者交通工具从背景中提取出来。
	技术上说，我们需要从静止的背景中提取移动的前景.

	提供一个无干扰的背景图像
	实时计算当前图像和 背景图像的差异（图像做差）  阈值二值化 得到 多出来的物体(mask)   再区域分割

	BackgroundSubtractorMOG2 是以高斯混合模型为基础的背景/前景分割算法。
	它是以2004年和2006年Z.Zivkovic的两篇文章为基础。
	这个算法的一个特点是它为每个像素选择一个合适的高斯分布。
	这个方法有一个参数detectShadows，默认为True，他会检测并将影子标记出来，
	但是这样做会降低处理速度。影子会被标记成灰色。


	 背景与前景都是相对的概念，以高速公路为例：有时我们对高速公路上来来往往的汽车感兴趣，
	这时汽车是前景，而路面以及周围的环境是背景；有时我们仅仅对闯入高速公路的行人感兴趣，
	这时闯入者是前景，而包括汽车之类的其他东西又成了背景。背景剪除是使用非常广泛的摄像头视频中探测移动的物体。
	这种在不同的帧中检测移动的物体叫做背景模型，其实背景剪除也是前景检测。



	一个强劲的背景剪除算法应当能够解决光强的变化，杂波的重复性运动，和长期场景的变动。
	下面的分析中会是用函数V(x,y,t)表示视频流，t是时间，x和y代表像素点位置。
	例如，V(1,2,3)是在t=3时刻，像素点(1,2)的光强。下面介绍几种背景剪除的方法。


#### 【1】 利用帧的不同（临帧差）
		该方法假定是前景是会动的，而背景是不会动的，而两个帧的不同可以用下面的公式：
			D(t+1) = I(x,y,t+1) - I(x,y,t)
		用它来表示同个位置前后不同时刻的光强只差。
		只要把那些D是0的点取出来，就是我们的前景，同时也完成了背景剪除。
		当然，这里的可以稍作改进，不一定说背景是一定不会动的。
		可以用一个阀值来限定。看下面的公式：
		  	I(x,y,t+1) - I(x,y,t) > 阈值
	 通过Th这个阀值来进行限定，把大于Th的点给去掉，留下的就是我们想要的前景。

	    #define threshold_diff 10 临帧差阈值
	    // 可使用 矩阵相减 
	    subtract(gray1, gray2, sub);  

		for (int i = 0;i<bac.rows; i++)  
		    for (int j = 0;j<bac.cols; j++)  
			if (abs(bac.at<unsigned char>(i, j)) >= threshold_diff)
		            //这里模板参数一定要用 unsigned char 8位(灰度图)，否则就一直报错  
			    bac.at<unsigned char>(i, j) = 255;  
			else bac.at<unsigned char>(i, j) = 0;
	 

#### 【2】 均值滤波（Mean filter） 当前帧 和 过去帧均值  做差

	      M(x,y,t) = 1/N SUM(I(x,y,(1..t)))
	      I(x,y,t+1) - M(x,y,t) > 阈值        为前景

		不是通过当前帧和上一帧的差别，而是当前帧和过去一段时间的平均差别来进行比较，
		同时通过阀值Th来进行控制。当大于阀值的点去掉，留下的就是我们要的前景了。

#### 【3】 使用高斯均值  + 马尔科夫过程
	// 初始化：
	     均值 u0 = I0 均值为第一帧图像
	     方差 c0 = (默认值)
	// 迭代：
	       ut = m * It + (1-m) * ut-1       马尔科夫过程
	       d  = |It - ut|                   计算差值
	       ct = m * d^2  + (1-m) * ct-1^2   更新 方差

	判断矩阵 d/ct > 阈值    为前景



#############################################################
## 物体检测模块　Object Detection (objdetect module)
##############################################################

	级联分类器 （CascadeClassifier） 
	 AdaBoost强分类器串接
	级联分类器是将若干个分类器进行连接，从而构成一种多项式级的强分类器。
	从弱分类器到强分类器的级联（AdaBoost 集成学习  改变训练集）
	级联分类器使用前要先进行训练，怎么训练？
	用目标的特征值去训练，对于人脸来说，通常使用Haar特征进行训练。



	【1】提出积分图(Integral image)的概念。在该论文中作者使用的是Haar-like特征，
		然后使用积分图能够非常迅速的计算不同尺度上的Haar-like特征。
	【2】使用AdaBoost作为特征选择的方法选择少量的特征在使用AdaBoost构造强分类器。
	【3】以级联的方式，从简单到 复杂 逐步 串联 强分类器，形成 级联分类器。

	级联分类器。该分类器由若干个简单的AdaBoost强分类器串接得来。
	假设AdaBoost分类器要实现99%的正确率，1%的误检率需要200维特征，
	而实现具有99.9%正确率和50%的误检率的AdaBoost分类器仅需要10维特征，
	那么通过级联，假设10级级联，最终得到的正确率和误检率分别为:
	(99.9%)^10 = 99%
	(0.5)^10   = 0.1

	检测体系：是以现实中很大一副图片作为输入，然后对图片中进行多区域，多尺度的检测，
	所谓多区域，是要对图片划分多块，对每个块进行检测，由于训练的时候一般图片都是20*20左右的小图片，
	所以对于大的人脸，还需要进行多尺度的检测。多尺度检测一般有两种策略，一种是不改变搜索窗口的大小，
	而不断缩放图片，这种方法需要对每个缩放后的图片进行区域特征值的运算，效率不高，而另一种方法，
	是不断初始化搜索窗口size为训练时的图片大小，不断扩大搜索窗口进行搜索。
	在区域放大的过程中会出现同一个人脸被多次检测，这需要进行区域的合并。
	无论哪一种搜索方法，都会为输入图片输出大量的子窗口图像，
	这些子窗口图像经过筛选式级联分类器会不断地被每个节点筛选，抛弃或通过。


	级联分类器的策略是，将若干个强分类器由简单到复杂排列，
	希望经过训练使每个强分类器都有较高检测率，而误识率可以放低。

	AdaBoost训练出来的强分类器一般具有较小的误识率，但检测率并不很高，
	一般情况下，高检测率会导致高误识率，这是强分类阈值的划分导致的，
	要提高强分类器的检测率既要降低阈值，要降低强分类器的误识率就要提高阈值，
	这是个矛盾的事情。据参考论文的实验结果，
	增加分类器个数可以在提高强分类器检测率的同时降低误识率，
	所以级联分类器在训练时要考虑如下平衡，一是弱分类器的个数和计算时间的平衡，
	二是强分类器检测率和误识率之间的平衡。


	// 级联分类器 类
	CascadeClassifier face_cascade;
	// 加载级联分类器
	face_cascade.load( face_cascade_name );
	// 多尺寸检测人脸

	std::vector<Rect> faces;//检测到的人脸 矩形区域 左下点坐标 长和宽
	Mat frame_gray;
	cvtColor( frame, frame_gray, CV_BGR2GRAY );//转换成灰度图
	equalizeHist( frame_gray, frame_gray );    //直方图均衡画
	//-- 多尺寸检测人脸
	face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
	// image：当然是输入图像了，要求是8位无符号图像，即灰度图
	//objects：输出向量容器（保存检测到的物体矩阵）
	//scaleFactor：每张图像缩小的尽寸比例 1.1 即每次搜索窗口扩大10%
	//minNeighbors：每个候选矩阵应包含的像素领域
	//flags:表示此参数模型是否更新标志位；
	//minSize ：表示最小的目标检测尺寸；
	//maxSize：表示最大的目标检测尺寸；


	Haar和 LBP (Local Binary Patterns)两种特征，并易于增加其他的特征。
	与Haar特征相比，LBP特征是整数特征，因此训练和检测过程都会比Haar特征快几倍。
	LBP和Haar特征用于检测的准确率，是依赖训练过程中的训练数据的质量和训练参数。
	训练一个与基于Haar特征同样准确度的LBP的分类器是可能的。


      与其他分类器模型的训练方法类似，同样需要训练数据与测试数据；
      其中训练数据包含正样本pos与负样本neg。
      训练程序opencv_haartraining.exe与opencv_traincascade.exe
      对输入的数据格式是有要求的，所以需要相关的辅助程序：

      opencv_createsamples 用来准备训练用的正样本数据和测试数据。
      opencv_createsamples 能够生成能被opencv_haartraining 和 
      opencv_traincascade 程序支持的正样本数据。
      它的输出为以 *.vec 为扩展名的文件，该文件以二进制方式存储图像。

      所以opencv级联分类器训练与测试可分为以下四个步骤：

	【1】准备训练数据
	【2】训练级联分类器
	【3】测试分类器性能
	【4】利用训练好的分类器进行目标检测

### 1、准备训练数据

      注：以行人数据为例，介绍分类器的训练

#### 1.1准备正样本

       正样本由opencv_createsamples生成。
       正样本可以由包含待检测物体的一张图片生成，也可由一系列标记好的图像生成。
       首先将所有的正样本放在一个文件夹，如图所示1。其中，pos.dat文件为所有图像的列表文件，
       格式如图2所示：
       其中，第一列为图像名，第二列为该图像中正样本的个数，最后的为正样本在图像中的位置以及需要抠出的正样本的尺寸。
       pos.dat文件的生成方式：在dos窗口进入pos文件夹，输入dir /b > pos.dat ; 
       这样只能生成文件名列表，后面的正样本个数与位置尺寸还需手动添加。
        位置尺寸 labelImg 

      opencv_createsamples.exe程序的命令行参数：

      -info <collection_file_name> 	描述物体所在图像以及大小位置的描述文件。
      -vec <vec_file_name>		输出文件，内含用于训练的正样本。
      -img <image_file_name>		输入图像文件名（例如一个公司的标志）。
      -bg<background_file_name>	背景图像的描述文件，文件中包含一系列的图像文件名，这些图像将被随机选作物体的背景。
      -num<number_of_samples>		生成的正样本的数目。
      -bgcolor<background_color>	背景颜色（目前为灰度图）；背景颜色表示透明颜色。
                              因为图像压缩可造成颜色偏差，颜色的容差可以由-bgthresh指定。
                              所有处于bgcolor-bgthresh和bgcolor+bgthresh之间的像素都被设置为透明像素。
      -bgthresh <background_color_threshold>
      -inv				如果指定该标志，前景图像的颜色将翻转。
      -randinv			如果指定该标志，颜色将随机地翻转。
      -maxidev<max_intensity_deviation>	前景样本里像素的亮度梯度的最大值。
      -maxxangle <max_x_rotation_angle>	X轴最大旋转角度，必须以弧度为单位。
      -maxyangle <max_y_rotation_angle>	Y轴最大旋转角度，必须以弧度为单位。
      -maxzangle<max_z_rotation_angle>	Z轴最大旋转角度，必须以弧度为单位。
      -show		很有用的调试选项。如果指定该选项，每个样本都将被显示。如果按下Esc键，程序将继续创建样本但不再显示。
      -w <sample_width>	输出样本的宽度（以像素为单位）。
      -h<sample_height>	输出样本的高度（以像素为单位）。

#### 1.3准备负样本

         负样本可以是任意图像，但是这些图像中不能包含待检测的物体。
         用于抠取负样本的图像文件名被列在一个neg.dat文件中。
         生成方式与正样本相同，但仅仅包含文件名列表就可以了。
         这个文件是纯文本文件，每行是一个文件名（包括相对目录和文件名）这些图像可以是不同的尺寸，
         但是图像尺寸应该比训练窗口的尺寸大，
         因为这些图像将被用于抠取负样本，并将负样本缩小到训练窗口大小。


#### 2、训练级联分类器

       OpenCV提供了两个可以训练的级联分类器的程序：opencv_haartraining与opencv_traincascade。
       opencv_haartraining是一个将被弃用的程序；
       opencv_traincascade是一个新程序。
       opencv_traincascade程序 命令行参数如下所示：

#### 1.通用参数：
         -data <cascade_dir_name> 目录名，如不存在训练程序会创建它，用于存放训练好的分类器。
         -vec <vec_file_name>     包含正样本的vec文件名（由opencv_createsamples程序生成）。
         -bg <background_file_name>            背景描述文件，也就是包含负样本文件名的那个描述文件。
         -numPos <number_of_positive_samples>  每级分类器训练时所用的正样本数目。
         -numNeg <number_of_negative_samples>  每级分类器训练时所用的负样本数目，可以大于 -bg 指定的图片数目。
         -numStages <number_of_stages>         训练的分类器的级数。
         -precalcValBufSize<precalculated_vals_buffer_size_in_Mb>  缓存大小，用于存储预先计算的特征值(feature values)，单位为MB。
         -precalcIdxBufSize<precalculated_idxs_buffer_size_in_Mb>  缓存大小，用于存储预先计算的特征索引(feature indices)，单位为MB。
                                                     内存越大，训练时间越短。
         -baseFormatSave    这个参数仅在使用Haar特征时有效。如果指定这个参数，那么级联分类器将以老的格式存储。

#### 2.级联参数：
         -stageType <BOOST(default)>   级别（stage）参数。目前只支持将BOOST分类器作为级别的类型。
         -featureType<{HAAR(default), LBP}>  特征的类型： HAAR - 类Haar特征； LBP - 局部纹理模式特征。
         -w <sampleWidth>
         -h <sampleHeight>  训练样本的尺寸（单位为像素）。必须跟训练样本创建（使用 opencv_createsamples 程序创建）时的尺寸保持一致。

#### 3.分类器参数：        
       -bt <{DAB, RAB, LB,GAB(default)}>  Boosted分类器参数：
       DAB - Discrete AdaBoost, RAB - Real AdaBoost, LB - LogitBoost, GAB -Gentle AdaBoost。 Boosted分类器的类型：
       -minHitRate<min_hit_rate>   分类器的每一级希望得到的最小检测率。总的检测率大约为 min_hit_rate^number_of_stages。
       -maxFalseAlarmRate<max_false_alarm_rate>   分类器的每一级希望得到的最大误检率。总的误检率大约为 max_false_alarm_rate^number_of_stages.
       -weightTrimRate <weight_trim_rate>
        Specifies whether trimmingshould be used and its weight.一个还不错的数值是0.95。
       -maxDepth <max_depth_of_weak_tree>   弱分类器树最大的深度。一个还不错的数值是1，是二叉树（stumps）。
       -maxWeakCount<max_weak_tree_count>   每一级中的弱分类器的最大数目。The boostedclassifier (stage) will have so many weak trees (<=maxWeakCount), as neededto achieve the given -maxFalseAlarmRate.

#### 4.类Haar特征参数：
       -mode <BASIC (default) |CORE | ALL>  选择训练过程中使用的Haar特征的类型。 BASIC 只使用右上特征， ALL使用所有右上特征和45度旋转特征。

#### 5.LBP特征参数：

      LBP特征无参数。



### 3.测试分类器性能

        opencv_performance 可以用来评估分类器的质量，但只能评估 opencv_haartraining 
	输出的分类器。它读入一组标注好的图像，运行分类器并报告性能，如检测到物体的数目，
	漏检的数目，误检的数目，以及其他信息。同样准备测试数据集test，生成图像列表文件，
	格式与训练者正样本图像列表相同，需要标注目标文件的个数与位置。


      opencv_haartraining程序训练一个分类器模型

      opencv_haartraining 的命令行参数如下：

      －data<dir_name>    	存放训练好的分类器的路径名。
      －vec<vec_file_name> 	正样本文件名（由trainingssamples程序或者由其他的方法创建的）
      －bg<background_file_name>		背景描述文件。
      －npos<number_of_positive_samples>，
      －nneg<number_of_negative_samples>	用来训练每一个分类器阶段的正/负样本。合理的值是：nPos = 7000;nNeg= 3000
      －nstages<number_of_stages>		训练的阶段数。
      －nsplits<number_of_splits>		决定用于阶段分类器的弱分类器。如果1，则一个简单的stump classifier被使用。
                                    如果是2或者更多，则带有number_of_splits个内部节点的CART分类器被使用。
      －mem<memory_in_MB>			预先计算的以MB为单位的可用内存。内存越大则训练的速度越快。
      －sym（default）
      －nonsym指				定训练的目标对象是否垂直对称。垂直对称提高目标的训练速度。例如，正面部是垂直对称的。
      －minhitrate<min_hit_rate>		每个阶段分类器需要的最小的命中率。总的命中率为min_hit_rate的number_of_stages次方。
      －maxfalsealarm<max_false_alarm_rate>	没有阶段分类器的最大错误报警率。总的错误警告率为max_false_alarm_rate的number_of_stages次方。
      －weighttrimming<weight_trimming>	指定是否使用权修正和使用多大的权修正。一个基本的选择是0.9
      －eqw
      －mode<basic(default)|core|all>		选择用来训练的haar特征集的种类。basic仅仅使用垂直特征。all使用垂直和45度角旋转特征。
      －w<sample_width>
      －h<sample_height>			训练样本的尺寸，（以像素为单位）。必须和训练样本创建的尺寸相同


    opencv_performance测试分类器模型


    opencv_performance 的命令行参数如下所示：

    -data <classifier_directory_name>	训练好的分类器
    -info <collection_file_name>   	描述物体所在图像以及大小位置的描述文件
    -maxSizeDiff <max_size_difference =1.500000>
    -maxPosDiff <max_position_difference =0.300000>
    -sf <scale_factor = 1.200000>
    -ni 				选项抑制创建的图像文件的检测
    -nos <number_of_stages = -1>
    -rs <roc_size = 40>]
    -w <sample_width = 24>
    -h <sample_height = 24>



#############################################################
## Machine Learning (ml module)
##############################################################

### 【1】支持向量机(SVM) 
	Support Vector Machines 

	支持向量机 (SVM) 是一个类分类器，正式的定义是一个能够将不同类样本
	在样本空间分隔的超平面(n-1 demension)。
	 换句话说，给定一些标记(label)好的训练样本 (监督式学习), 
	SVM算法输出一个最优化的分隔超平面。

	假设给定一些分属于两类的2维点，这些点可以通过直线分割， 我们要找到一条最优的分割线.

	|
	|
	|              +
	|            +    +
	|               +
	|  *               +
	|                        +
	|   *   *           +
	|   *                 + 
	|  *      *             +
	|   *    *
	——————————————————————————————————>

	w * 超平面 = 0  

	w * x+ + b >= 1
	w * x- + b <= -1
	  
	w * x+0 + b =  1          y = +1
	w * x-0 + b = -1          y = -1


	(x+0 - x-0) * w/||w|| =
	 
	x+0 * w/||w||  - x-0 * w/||w|| = 

	(1-b) /||w|| - (-1-b)/||w||  = 

	2 * /||w||   maxmize
	=======>

	min 1/2 * ||w||^2

	这是一个拉格朗日优化问题

	L = 1/2 * ||w||^2 - SUM(a * yi *( w* xi +b) - 1)

	dL/dw  = ||w||  - sum(a * yi * xi)   subjext  to 0    calculate the  w
	  
	dL/db  =  SUM(a * yi)                subjext  to 0     

	====>
	||w||  = sum(a * yi * xi)
	SUM(a * yi)  = 0
	=====>

	L =  -1/2  SUM(SUM(ai*aj * yi*yj * xi*xi))


### 【2】 支持向量机对线性不可分数据的处理

	为什么需要将支持向量机优化问题扩展到线性不可分的情形？ 
	在多数计算机视觉运用中，我们需要的不仅仅是一个简单的SVM线性分类器， 
	我们需要更加强大的工具来解决 训练数据无法用一个超平面分割 的情形。

	我们以人脸识别来做一个例子，训练数据包含一组人脸图像和一组非人脸图像(除了人脸之外的任何物体)。 
	这些训练数据超级复杂，以至于为每个样本找到一个合适的表达 (特征向量) 以让它们能够线性分割是非常困难的。


	最优化问题的扩展

	还记得我们用支持向量机来找到一个最优超平面。 既然现在训练数据线性不可分，
	我们必须承认这个最优超平面会将一些样本划分到错误的类别中。 在这种情形下的优化问题，
	需要将 错分类(misclassification) 当作一个变量来考虑。
	新的模型需要包含原来线性可分情形下的最优化条件，即最大间隔(margin), 
	以及在线性不可分时分类错误最小化。


	比如，我们可以最小化一个函数，该函数定义为在原来模型的基础上再加上一个常量乘以样本被错误分类的次数:
	L = 1/2 * ||w||^2 - SUM(a * yi *( w* xi +b) - 1)  +  c * ( 样本被错误分类的次数)

	它没有考虑错分类的样本距离同类样本所属区域的大小。 因此一个更好的方法是考虑 错分类样本离同类区域的距离:

	L = 1/2 * ||w||^2 - SUM(a * yi *( w* xi +b) - 1)  +  c * ( 错分类样本离同类区域的距离)


	   C比较大时分类错误率较小，但是间隔也较小。 在这种情形下，
	错分类对模型函数产生较大的影响，既然优化的目的是为了最小化这个模型函数，
	那么错分类的情形必然会受到抑制。

	   C比较小时间隔较大，但是分类错误率也较大。 在这种情形下，
	模型函数中错分类之和这一项对优化过程的影响变小，
	优化过程将更加关注于寻找到一个能产生较大间隔的超平面。


###【3】PCA（Principal Component Analysis）主成分分析
	主要用于数据降维

	样本  协方差矩阵 的特征值和特征向量  取前四个特征值所对应的特征向量
	特征矩阵 投影矩阵


	对于一组样本的feature组成的多维向量，多维向量里的某些元素本身没有区分
	我们的目的是找那些变化大的元素，即方差大的那些维，
	而去除掉那些变化不大的维，从而使feature留下的都是最能代表此元素的“精品”，
	而且计算量也变小了。

	对于一个k维的feature来说，相当于它的每一维feature与
	其他维都是正交的（相当于在多维坐标系中，坐标轴都是垂直的），
	那么我们可以变化这些维的坐标系，从而使这个feature在某些维上方差大，
	而在某些维上方差很


	求得一个k维特征的投影矩阵，这个投影矩阵可以将feature从高维降到低维。
	投影矩阵也可以叫做变换矩阵。新的低维特征必须每个维都正交，特征向量都是正交的

	通过求样本矩阵的协方差矩阵，然后求出协方差矩阵的特征向量，这些特征向量就可以构成这个投影矩阵了。
	特征向量的选择取决于协方差矩阵的特征值的大

	对于一个训练集，100个样本，feature是10维，那么它可以建立一个100*10的矩阵，作为样本。
	求这个样本的协方差矩阵，得到一个10*10的协方差矩阵，然后求出这个协方差矩阵的特征值和特征向量，
	应该有10个特征值和特征向量，我们根据特征值的大小，取前四个特征值所对应的特征向量，
	构成一个10*4的矩阵，这个矩阵就是我们要求的特征矩阵，100*10的样本矩阵乘以这个10*4的特征矩阵，
	就得到了一个100*4的新的降维之后的样本矩阵，每个样本的维数下.

###【4】人工神经网络(ANN) 简称神经网络(NN)，

	 能模拟生物神经系统对物体所作出的交互反应，
	 是由具有适应性的简单单元(称为神经元)组成的广泛并行互连网络。
	 神经元
	 y = w  * x + b   线性变换   （旋转 平移 伸缩 升/降维）
	 z = a(y)         非线性变换 激活函数

	常用 Sigmoid 函数作激活函数

	y = sigmod(x)  = 1/(1+exp(-x))  映射到0 ～ 1之间
	 
	 OpenCV 中使用的激活函数是另一种形式，

	f(x) = b *  (1 - exp(-c*x)) / (1 + exp(-c*x))

	当 α = β = 1 时
	f(x) =(1 - exp(-x)) / (1 + exp(-x))
	该函数把可能在较大范围内变化的输入值，“挤压” 到 (-1, 1) 的输出范围内

	// 设置激活函数，目前只支持 ANN_MLP::SIGMOID_SYM
	virtual void cv::ml::ANN_MLP::setActivationFunction(int type, double param1 = 0, double param2 = 0); 

	神经网络
	2.1  感知机 (perceptron)
	  感知机由两层神经元组成，输入层接收外界输入信号，而输出层则是一个 M-P 神经元。
	  实际上，感知机可视为一个最简单的“神经网络”，用它可很容易的实现逻辑与、或、非等简单运算。

	2.2 层级结构
	  常见的神经网络，可分为三层：输入层、隐含层、输出层。
	  输入层接收外界输入，隐层和输出层负责对信号进行加工，输出层输出最终的结果。

	2.3  层数设置
	  	OpenCV 中，设置神经网络层数和神经元个数的函数为 setLayerSizes(InputArray _layer_sizes)，
		// (a) 3层，输入层神经元个数为 4，隐层的为 6，输出层的为 4
		Mat layers_size = (Mat_<int>(1,3) << 4,6,4);

		// (b) 4层，输入层神经元个数为 4，第一个隐层的为 6，第二个隐层的为 5，输出层的为 4
		Mat layers_size = (Mat_<int>(1,4) << 4,6,5,4);

	1)  创建
		static Ptr<ANN_MLP> cv::ml::ANN_MLP::create();  // 创建空模型

	2) 设置参数

	// 设置神经网络的层数和神经元数量
	virtual void cv::ml::ANN_MLP::setLayerSizes(InputArray _layer_sizes);

	// 设置激活函数，目前只支持 ANN_MLP::SIGMOID_SYM
	virtual void cv::ml::ANN_MLP::setActivationFunction(int type, double param1 = 0, double param2 = 0); 

	// 设置训练方法，默认为 ANN_MLP::RPROP，较常用的是 ANN_MLP::BACKPROP
	// 若设为 ANN_MLP::BACKPROP，则 param1 对应 setBackpropWeightScale()中的参数,
	// param2 对应 setBackpropMomentumScale() 中的参数
	virtual void cv::ml::ANN_MLP::setTrainMethod(int method, double param1 = 0, double param2 = 0);
	virtual void cv::ml::ANN_MLP::setBackpropWeightScale(double val); // 默认值为 0.1
	virtual void cv::ml::ANN_MLP::setBackpropMomentumScale(double val); // 默认值为 0.1
	 
	// 设置迭代终止准则，默认为 TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 0.01)
	virtual void cv::ml::ANN_MLP::setTermCriteria(TermCriteria val);

	3)  训练

	// samples - 训练样本; layout - 训练样本为 “行样本” ROW_SAMPLE 或 “列样本” COL_SAMPLE; response - 对应样本数据的分类结果
	virtual bool cv::ml::StatModel::train(InputArray samples,int layout,InputArray responses);  

	4)  预测

	// samples，输入的样本书数据；results，输出矩阵，默认不输出；flags，标识，默认为 0
	virtual float cv::ml::StatModel::predict(InputArray samples, OutputArray results=noArray(),int flags=0) const;　　　　　　 

	5) 保存训练好的神经网络参数
	    bool trained = ann->train(tData);  
	    if (trained) {
		  ann->save("ann_param");
	     }

	6) 载入训练好的神经网络
	      Ptr<ml::ANN_MLP> ann = ml::ANN_MLP::load("ann_param");



#############################################################
Computational photography (photo module)
##############################################################



#############################################################
GPU-Accelerated Computer Vision (cuda module)
##############################################################


#############################################################
OpenCV iOS

Run OpenCV and your vision apps on an iDevice
##############################################################




#############################################################
OpenCV Viz

These tutorials show how to use Viz module effectively.

##############################################################



#############################################################

##############################################################
