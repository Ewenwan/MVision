# 源文件

      iDT代码的依赖包括两个库：
      OpenCV: readme中推荐用2.4.2， 实际上用最新的2.4.13也没问题。
              但OpenCV3就不知道能不能用了，没有试过。
      ffmpeg: readme中推荐用0.11.1。实际上装最新的版本也没有问题
      这两个库的安装教程网上很多，就不再多做介绍了。
      而且也都是很常用的库。
      在安装完以上两个库后，就可以进行代码编译了。
      只需要在代码文件夹下make一下就好，
      编译好的可执行文件在./release/下。
      使用时输入 视频文件的路径作为参数即可
      ./release/DenseTrackStab ./test_sequences/person01_boxing_d1_uncomp.avi。
      代码结构
      iDT代码中主要包括如下几个代码文件
      DenseTrackStab.cpp: iDT算法主程序
      DenseTrackStab.h:   轨迹跟踪的一些参数，以及一些数据结构体的定义
      Descriptors.h:      特征相关的各种函数
      Initialize.h:       初始化相关的各种函数
      OpticalFlow.h:      光流相关的各种函数
      Video.cpp:          这个程序与iDT算法无关，
                          只是作者提供用来测试两个依赖库是否安装成功的。

      bound box相关内容
      bound box即提供视频帧中人体框的信息，
      在计算前后帧的投影变换矩阵时，不使用人体框中的匹配点对。
      从而排除人体运动干扰，使得对相机运动的估计更加准确。
      作者提供的文件中没有bb_file的格式，
      代码中也没有读入bb_file的接口，
      若需要用到需要在代码中添加一条读入文件语句
      （下面的代码解析中已经添加）。
      bb_file的格式如下所示
      frame_id a1 a2 a3 a4 a5 b1 b2 b3 b4 b5
      其中frame_id是帧的编号，从0开始。
      代码中还有检查步骤，保证bb_file的长度与视频的帧数相同。
      后面的数据5个一组，为人体框的参数。
      按顺序分别为：
      框左上角点的x，框左上角点的y，框右下角点的x，框右下角点的y，置信度。
      需要注意的是虽然要输入置信度，
      但实际上这个置信度在代码里也没有用上的样子，
      所以取任意值也不影响使用。
      因为一帧图像可能框出来的人有好多个，
      这种细粒度的控制比大致框出一个范围能更有效地滤去噪声.
      至于如何获得这些bound box的数据，最暴力的方法当然是手工标注，不过这样太辛苦了。
      在项目中我们采用了SSD（single shot multibox detector）/yolov3算法检测人体框的位置。算法检测人体框的位置。
      主程序代码解析
      iDT算法代码的大致思路为：
      1. 读入新的一帧
      2. 通过SURF特征和光流计算当前帧和上一帧的投影变换矩阵
      3. 使用求得的投影变换矩阵对当前帧进行warp变换，消除相机运动影响
      4. 利用warp变换后的当前帧图像和上一帧图像计算光流
      5. 在各个图像尺度上跟踪轨迹并计算特征
      6. 保存当前帧的相关信息，跳到1
      几个头文件：
      DenseTrackStab.h 定义了Track等的数据结构。最重要的track类里面可以看出：
        std::vector<Point2f> point; //轨迹点
        std::vector<Point2f> disp; //偏移点
        std::vector<float> hog; //hog特征
        std::vector<float> hof; //hof特征
        std::vector<float> mbhX; //mbhX特征
        std::vector<float> mbhY; //mbhY特征
        int index;// 序号
      基本方法就是在重采样中提取轨迹，
      在轨迹空间中再提取hog,hof,mbh特征，
      这些特征组合形成iDT特征，
      最终作为这个动作的描述。
      Initialize.h：涉及各种数据结构的初始化，usage()可以看看；
      OpticalFlow.h: 主要用了Farneback计算光流，
      博客参考：
      https://blog.csdn.net/ironyoung/article/details/60884929
      光流源码：
      https://searchcode.com/file/30099587/opencv_source/src/cv/cvoptflowgf.cpp
      把金字塔的方法也写进去了，
      金字塔方法主要是为了消除不同尺寸的影响，
      让描述子有更好的泛化能力。
      Descriptors.h：提供一些工具函数:
      计算直方图描述子
      计算梯度直方图
      计算光流直方图
      计算光流梯度直方图
      密集采样轨迹点  DenseSample
      载入 人体边框数据
      创建去除人体区域的mask掩膜
      对帧图进行单应矩阵反变换 去除相机移动的影响
      BFMatcher 计算匹配点对
      合并光流匹配点对和 surf匹配点对
      根据光流得到光流匹配点
