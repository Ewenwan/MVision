# 目标跟踪
[目标跟踪算法总结](https://github.com/Ewenwan/benchmark_results)

# 应用
    Augmented Reality   增强现实
    Motion Capture      运动捕捉
    Surveillance        监控
    Sports Analysis     运动(足球、篮球...)分析
    ...
# 目录
    1. 运动估计/光流 Mption Estimation / optical Flow
    2. 单目标跟踪    Single Object Tracking
    3. 多目标跟踪    Multiple Object Trackink

## 1. 运动估计/光流
![](https://github.com/Ewenwan/MVision/blob/master/3D_Object_Detection/Object_Tracking/img/mf.PNG)

    3D运动 投影到 2D图像平面上
    相邻帧上对应像素点的移动 I(x,y,t) = I(x+dx,y+dy,t+dt) 灰度不变
    
    其计算方法可以分为三类： 
    （1）基于区域或者基于特征的匹配方法； 
    （2）基于频域的方法； 
    （3）基于梯度的方法； 
    光流基于光度不变假设，具体来说分成三个假设 
    （1）亮度恒定，前后帧观测到的对应点的灰度值一样。 
    （2）时间连续或者运动位移小。 
    （3）空间一致性：邻近点有相似运动，同一子图像的像素点具有相同的运动。 
### 传统算法求光流 klt Kanade-Lucas-Tomasi Feature Tracker
[OPENCV光流源码分析](https://blog.csdn.net/ironyoung/article/details/60884929)

    光流（optical flow）是指平面上，光照模式的变化情况。
    在计算机视觉领域，是指视频图像中各点像素随时间的运动情况。
    光流具有丰富的运动信息，因而在运动估计、自动驾驶和行为识别方面都有广泛应用。
    光流预测通常是从一对时间相关的图像对中，估计出第一张图像中各个像素点在相邻图像中的位置。
    
    首先找去特征点，之后用光流去跟踪的方法。
    
    a. 找到好的特征点，例如 Harris角点、Shi-Tomasi角点、FAST角点等
    b. 计算光流。Lucas-Kanade method, LK光流(灰度不变，相邻像素运动相似)
       灰度不变性:
        I(x,y,t) = I(x+dx,y+dy,t+dt)
                 = I(x,y,t) + Ix*dx +Iy*dy +It*dt  （泰勒公式展开，Ix,Iy表示灰度梯度，Ix表示相邻帧相同点像素变化）
        ====>  Ix*dx +Iy*dy +It*dt = 0 , 两边同除以 dt, 得到：
                 Ix*dx/dt + Iy*dy/dt + It = 0
                 =  Ix * Vx + Iy*Vy = -It,   Vx为像素水平速度， Vy为像素垂直速度
       相邻像素运动相似:
                Ix(p1) * Vx + Iy(p1)*Vy = -It(p1)
                Ix(p2) * Vx + Iy(p2)*Vy = -It(p2)
                ...
                Ix(pn) * Vx + Iy(pn)*Vy = -It(pn)
                注：p1,p2,...,pn是窗口内的像素点
                
        写成矩阵形式有：

            A = [Ix(p1) Iy(p1);
                 Ix(p2) Iy(p2)
                 ...
                 Ix(pn) Iy(pn)]

            B = [It(p1)
                 It(p2)
                 ...
                 It(pn)]

            V = [Vx
                 Vy]

            A * V  = -B
            线性方程组求解:
            V = (A转置*A)逆 * A转置*(-B) = A逆*A转置逆* A转置*(-B)=A逆*(-B) 
            (伪逆求解，A矩阵可能没有逆矩阵)
            
### 基于卷积神经网络的光流预测算法

    光流问题长久以来，主要被基于变分能量模型的优化算法和基于块匹配的启发式算法统治着。
    随着深度神经网络技术在计算机视觉领域取得的成功，科学家们开始尝试利用深度学习技术的优势去解决光流问题。

[代码 flownet/flownet2](https://github.com/Ewenwan/flownet2)

![](http://img.mp.sohu.com/upload/20170520/1bc91a54f9844b82b3a47f680a56798b_th.png)
    
    他们的两个神经网络大体的思路就是这样。
    首先他们有一个收缩部分，主要由卷积层组成，用于深度的提取两个图片的一些特征。
    但是pooling会使图片的分辨率降低，为了提供一个密集的光流预测，他们增加了一个扩大层，能智能的把光流恢复到高像素。
    
    FlowNet[1]是第一个尝试利用CNN去直接预测光流的工作，它将光流预测问题建模为一个有监督的深度学习问题。
    网络整体上为编码模块接解码模块结构，编码模块均为9层卷积加ReLU激活函数层，
    解码模块均为4层反卷积加ReLU激活函数层，在文中解码模块又被称为细化模块。
    整个网络结构类似于FCN(全卷机网络)，由卷积和反卷积层构成，没有全连接层，因此理论上对输入图像的大小没有要求。
    
    
## 2. 单目标跟踪
    单目标，单摄像头
    无模型的，只有第一帧指定的 框
    短期跟踪，不支持重新检测，丢失后，就跟踪失败
    跟踪器不使用任何未来帧。
    
    步骤：
        启动跟踪器 Setup tracker
        设置目标区域 Read initial object region and first image
        初始化跟踪器 Initialize tracker with provided region and image
        循环 loop
            读取下一张图像 Read next image
            图像为空 if image is empty then
                跳出循环 Break the tracking loop
            end if
            更新跟踪器 Update tracker with provided image
            记录目标区域 Write region to file
        结束循环 end loop
        清理跟踪器 Cleanup tracker
    
    
    
    目标视觉跟踪(visual object tracking),根据目标的跟踪方式，跟踪一般可以分为两大类：
        a. 生产(generative)模型方法 
        b. 判别(discriminative)模型方法。
    
    生成类方法 在当前帧对目标区域建模，下一帧寻找与模型最相似的区域就是预测位置，
        如卡尔曼滤波(Kalman Filter)，粒子滤波(Particle Filter)，均值漂移算法(Mean Shift)等。
        
    目前比较流行的是判别类方法(Discriminative Tracking)，也叫跟踪检测(tracking-by-detection)，
        当前帧以目标区域为正样本，背景区域为负样本用来训练分类器，
        下一帧用训练好的分类器找最优区域，经典的判别类方法有Struck和TLD等。
        
    最近几年相关滤波方法(Correlation Filter Tracking)如MOSSE, CF，KCF/DCF，CN，DSST也比较火。
        MOSSE算法开启了相关滤波器的大门，提出以滤波器求相关的形式来获取输出响应，进而获得最大响应处的位置也即我们期望跟踪的目标中心位置。 
        CF，KCF/DCF,三者都是核相关滤波方法，引入核函数使高维空间中的非线性问题变为线性问题从而加速训练和检测，
        利用循环矩阵增加训练样本，利用DFT的性质避免求逆操作提高跟踪速度。
        CSK利用图像的灰度信息，高斯滤波和1倍padding；
        KCF利用HOG特征，高斯滤波和1.5倍padding，
        DCF利用HOG特征，线性滤波和1.5倍padding。 
        CSK没有解决多尺度问题和循环矩阵边界效应只利用了图像的灰度信息；
        KCF和DCF输入可以是多通道的可以利用更多的图像信息进行跟踪。 
        
        CN跟踪算法是CSK跟踪算法的扩展，CSK没有利用图像的颜色信息，而CN则是在CSK的基础上加上了图像颜色信息，
        将图像的RGB颜色信息映射到包含(black , blue , brown , grey , green , orange , pink , purple , red , white , yellow)
        11颜色通道的CN空间，对每一个通道空间进行FFT变换，核映射，然后对频域信号求和，
        这种方法运算量大，可以使用PCA对颜色通道降维，把11维的颜色通道降为2维，提高运算速度。 
        
        DSST提出了基于3维尺度空间相关滤波器translation-scale的联合跟踪方式，
        利用两个滤波器位置滤波器(translation filter)和尺度滤波器(scale filter)依次进行目标定位和尺度评估，
        两个滤波器相互独立可以利用不同的特征种类和特征计算方式进行训练和测试，
        而利用尺度滤波器进行尺度估计的方法可以移植到其他算法中。
        fDSST对DSST进行加速，分别对位置滤波器和尺度滤波器进行PCA降维和QR分解来降低计算量提高计算速度。 
        
        SRDCF在KCF/DCF的基础上通过多尺度搜索解决了多尺度问题，并且加入惩罚项来解决循环矩阵的边界效应。
        在空间权重函数中加入惩罚权重w，超过边界的w更大作为惩罚；在检测时选择一定的候选框进行尺度匹配，找到最合适的尺度大小。 
        
    深度学习方法：
        
        
        
    通常目标跟踪主要面临的难点有：
        外观变化，光照变化，快速运动，运动模糊，背景干扰等。
    
    
