# RTAB-MAP（Real Time Appearance-Based Mapping）是RGB-D SLAM中比较经典的一个方案。

[代码](https://github.com/Ewenwan/rtabmap)

        它实现了RGB-D SLAM中所有应该有的东西：
          1、基于特征的视觉里程计、
          2、基于词袋的回环检测、
          3、后端的位姿图优化，
          4、以及点云和三角网格地图。
          因此，RTAB-MAP给出了一套完整的（但有些庞大的）RGB-D SLAM方案。
        目前我们已经可以直接从ROS中获得其二进制程序，
        此外，在Google Project Tango上也可以获取其App使用（如图6所示）


        RTAB-MAP支持一些常见的RGB-D和双目传感器，
        像Kinect、Xtion等，且提供实时的定位和建图功能。
        不过由于集成度较高，使得其他开发者在它的基础上进行二次开发变得困难，
        所以RTAB-MAP更适合作为SLAM应用而非研究使用。
