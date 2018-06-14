# 传统行为识别方法-DT识别算法-iDT改进
      1、密集采样:
        多尺度(8个)
        间隔采样（5）
        无纹理区域点的剔除
      2、光流法跟踪关键点:
        I(x, y, t) = I(x+dx, y+dy, t+dt) 
        泰勒展开：
        [Ix, Iy] * [u; v] = -It 
        局部区域 运动相同：
         A * U = b 
         U = (A转置*A)逆 * A转置 * b  伪逆求解
        光流中值滤波：
         Pt+1=(xt+1,yt+1)=(xt,yt)+(M∗Ut)|xt,yt
      3、特征计算：
        轨迹描述子(trajectories)：
        ΔPt=(Pt+1−Pt)=(xt+1 − xt, yt+1 − yt)
        长度为L的轨迹：
        (ΔPt,...,ΔPt+L−1)
        正则化：
        T=(ΔPt,...,ΔPt+L−1)/∑||ΔPj||
      3.1、区域选取：HOG HOF MBH
        1、选取相邻L帧(15);
        2、对每条轨迹在每个帧上取N*N 的像素区域(32*32)
        3、对上述区域划分成nc*nc个格子（2*2）
        4、在时间空间上划分成nt段(3段)
        5、这样就有 nc*nc*nt个空间划   分区域。
      3.2、特征计算：
        梯度直方图HOG:
          gm = sqrt(gx^2 + gy^2)   赋值
          gc  = atan(gy/gx)             方向
          0~360分成8个区间进行 
          幅值加权统计
          维度 2*2*3*8=96
        光流直方图HOF：
          bin数目取为8+1，前8个bin与HOG都相同。
          额外的一个用于统计光流幅度小于某个阈值的像素，
          维度 2*2*3*9=108
        光溜梯度直方图 MBHx 、 MBHy
          在两个光流图上计算梯度
          维度2*96=192
          
      4、剔除相机移动造成的光流的影响
      步骤：
      1、特征点匹配。使用SURF特征点提取匹配算法，配合人体目标检测剔除人体区域的匹配点对；
      2、利用RANSAC 随机采样序列一致性算法估计前后两帧的 单应投影变换矩阵得到相邻两帧的运动变换矩阵T=[R,t];
      3、计算相机运动引起的光流 Mt =  [u',v'] = I(t+1)' - I(t) = T *I(t)  - I(t) ;
      4、计算全局光流Ft 减去 运动光流Mt 得到去除运动噪声后的 光流 
## 单应矩阵H 恢复变换矩阵 R, t：

      p2 = H12 * p1 

      展开成矩阵形式： 
      u2   h1 h2 h3   u1 
      v2 = h4 h5 h6 * v1 
        h7 h8 h9   1 

      按矩阵乘法展开： 
      u2 = (h1*u1 + h2*v1 + h3) /( h7*u1 + h8*v1 + h9) 
      v2 = (h4*u1 + h5*v1 + h6) /( h7*u1 + h8*v1 + h9)

      将分母移到另一边，两边再做减法：
       -((h4*u1 + h5*v1 + h6) – 
         ( h7*u1*v2 + h8*v1*v2 + h9*v2))=0 

        h1*u1 + h2*v1 + h3 –
       ( h7*u1*u2 + h8*v1*u2 + h9*u2)=0 

      有两个约束，采用4对点写成矩阵形式：
      A*h = 0
      对矩阵A使用SVD奇异值分解可以得到 h

      单应矩阵恢复 旋转矩阵 R 和平移向量t ：
      p2 = H21 * p1 = H21 * KP 
      p2 = K( RP + t) = KTP = H21 * KP 
      T = K 逆 * H21*K 

## 代码
      作者给出的代码只包括了密集采样+轨迹跟踪+特征提取三个部分，后续的特征编码部分未给出
      
[(iDT(improved Dense Trajectory)源代码](http://lear.inrialpes.fr/people/wang/dense_trajectories))

[idt轨迹跟踪特征代码](http://lear.inrialpes.fr/people/wang/download/dense_trajectory_release_v1.2.tar.gz)

[fv编码代码](https://github.com/chensun11/dtfv)

[行为识别笔记：iDT算法用法与代码解析1](https://blog.csdn.net/wzmsltw/article/details/53221179)

[Improved Dense Trajectory用法及源码分析](https://blog.csdn.net/zackzhaoyang/article/details/50881114)

[iDT 算法 windows上编译](https://blog.csdn.net/u013913216/article/details/78646461)

## windows上编译细节
      论文中给出的代码是在Linux系统上编译的，
      有几个与Linux系统相关的函数，
      自己在windows vs2013上编译的时候将这些函数用其他的函数代替了。

[ windows上编译细节参考ao](https://blog.csdn.net/u013913216/article/details/78646461)

## 批量化运行
      在做自己的数据库时，不可能一个视频一个视频的去输入，
      这时可以写一个vim脚本，来批量运行，可以参考如下代码：
      后面的fv会直接提取命令行的输出来进行编码，这里如果不想保存也可以
      直接保存编码后的特征也可以。
```asm   
      #!/usr/bin/env bash

      cmd="release/DenseTrackStab"
      filePath="" #数据存放路径
      fileType=".txt"
      function readfile(){
      for file in `ls $1`
      do
        if [ -d $1"/"$file ]
        then
            readfile $1"/"$file
        else
            echo $file
            $cmd $1"/"$file
        fi
      done
      }
      readfile $filePath
```

## idt 特征提取

**输出feature的结构：

      特征是一个接一个计算的每一个都是单独一列，由下面的格式给出：

      frameNum mean_x mean_y var_x var_y length scale x_pos y_pos t_pos Trajectory HOG HOF MBHx MBHy

      前十个部分是关于轨迹的：

      frameNum:     The trajectory ends on which frame

      mean_x:       The mean value of the x coordinates of the trajectory

      mean_y:       The mean value of the y coordinates of the trajectory

      var_x:        The variance of the x coordinates of the trajectory

      var_y:        The variance of the y coordinates of the trajectory

      length:       The length of the trajectory

      scale:        The trajectory is computed on which scale

      x_pos:        The normalized x position w.r.t. the video (0~0.999), for spatio-temporal pyramid

      y_pos:        The normalized y position w.r.t. the video (0~0.999), for spatio-temporal pyramid

      t_pos:        The normalized t position w.r.t. the video (0~0.999), for spatio-temporal pyramid

      下面的5个部分是一个接一个连起来的：

      Trajectory:    2x[trajectory length] (default 30 dimension)

      HOG:           8x[spatial cells]x[spatial cells]x[temporal cells] (default 96 dimension)

      HOF:           9x[spatial cells]x[spatial cells]x[temporal cells] (default 108 dimension)

      MBHx:          8x[spatial cells]x[spatial cells]x[temporal cells] (default 96 dimension)

      MBHy:          8x[spatial cells]x[spatial cells]x[temporal cells] (default 96 dimension)



每隔设定的帧长度会从零开始再计算。


## fv特征编码
      对五种特征分别进行训练得到 pca压缩矩阵系数参数 和 GMM高斯混合模型参数
      对新提取的特征 先使用pca压缩后 在进行 gmm编码
      
## svm分类
      对fv编码后的特征进行分类
      
      

