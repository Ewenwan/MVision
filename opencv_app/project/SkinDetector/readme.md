# opencv写的一个人的皮肤检测器，里面封装了多种算法

[OpenCV探索之路（二十七）：皮肤检测技术](https://www.cnblogs.com/skyfsm/p/7868877.html)

[代码](https://github.com/AstarLight/skin-detector)

里面封装了以下6种主流的皮肤检测算法：

      RGB color space
      Ycrcb之cr分量+otsu阈值化
      YCrCb中133<=Cr<=173 77<=Cb<=127
      HSV中 7<H<20，s>48,v>50
      基于椭圆皮肤模型的皮肤检测
      opencv自带肤色检测类AdaptiveSkinDetector


