# DTAM Dense Tracking and Mapping 直接法 稠密跟踪和建图

      DTAM继承了关键帧的架构，但对关键帧的处理和传统的特征点提取有很大的不同。
      相比传统方法对每一帧进行很稀疏的特征点提取，
      DTAM的direct method在默认环境亮度不变（brightness consistancy assumption）的前提下，
      对每一个像素的深度数据进行inverse depth的提取和不断优化来建立稠密地图并实现稳定的位置跟踪。

      用比较直观的数字对比来说明DTAM和PTAM的区别：
          DTAM的一个关键帧有30万个像素的深度估计，而PTAM一般的做法是最多一千个。

      DTAM的优缺点很明显（具体对比见图1）：
          准、稳，但速度是问题，每一个像素都计算，确实容易通过GPU并行计算，但功耗和产品化的难度也都随之升高。
