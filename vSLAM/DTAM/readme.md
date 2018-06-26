# DTAM Dense Tracking and Mapping RGB-D相机 直接法 稠密跟踪和建图
![](https://img-blog.csdn.net/20160406220311974?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

      目前流行的大多数VSLAM都是基于鲁棒的特征检测和跟踪，
      DTAM是基于单像素的方法，
      采用了在稠密地图（densemap）创建和相机位置估计都很有优势的低基线帧（lowbaseline frame）。
      像PATM一样，它分为两个部分：姿势跟踪和3d 映射。
      PTAM仅仅跟踪稀疏的3d点，
      DTAM保持了对关键帧的稠密深度映射（densedepth map）。
      
      lsd_slam也是直接法，不过是稀疏的地图，跟踪灰度梯度大的像素点(单目相机)
      
      DTAM继承了关键帧的架构，但对关键帧的处理和传统的特征点提取有很大的不同。
      相比传统方法对每一帧进行很稀疏的特征点提取，
      DTAM的direct method在默认环境亮度不变（brightness consistancy assumption）的前提下，
      对每一个像素的深度数据进行inverse depth的提取和不断优化来建立稠密地图并实现稳定的位置跟踪。

      用比较直观的数字对比来说明DTAM和PTAM的区别：
          DTAM的一个关键帧有30万个像素的深度估计，而PTAM一般的做法是最多一千个。

      DTAM的优缺点很明显（具体对比见图1）：
          准、稳，但速度是问题，每一个像素都计算，确实容易通过GPU并行计算，但功耗和产品化的难度也都随之升高。
