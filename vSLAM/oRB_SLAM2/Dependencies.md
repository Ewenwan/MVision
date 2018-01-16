##List of Known Dependencies
###ORB-SLAM2 version 1.0
### ORB-SLAM2 学习 记录
# 代码依赖

#####Code in **src** and **include** folders

* *ORBextractor.cc orb特征提取算法 源自opencv 库的 orb.cpp*.
This is a modified version of orb.cpp of OpenCV library. The original code is BSD licensed.

* *PnPsolver.h, PnPsolver.cc  3D - 2D 点对  射影几何  求解  变换矩阵  R 和 t*.
This is a modified version of the epnp.h and epnp.cc of Vincent Lepetit. 
This code can be found in popular BSD licensed computer vision libraries as [OpenCV](https://github.com/Itseez/opencv/blob/master/modules/calib3d/src/epnp.cpp) and [OpenGV](https://github.com/laurentkneip/opengv/blob/master/src/absolute_pose/modules/Epnp.cpp). The original code is FreeBSD.

* Function *ORBmatcher::DescriptorDistance* in *ORBmatcher.cc ORB特征匹配算法 二进制特征向量 字符串汉明距离匹配*.
The code is from: http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel.
The code is in the public domain.

#####Code in Thirdparty folder  第三方库

* All code in **DBoW2** folder.  Bag of word 词袋 算法  用于匹配 是图像检索领域最常用的方法，也是基于内容的图像检索中最基础的算法。
*  * 所有特征 组合成 一个词典   来一个新特征  用词典中的单词 线性表示 维度 为 词典单词个数  
* 计算图像之间相似度时仍然计算的是BoW向量之间的距离( 根据描述子的不同可能选择Hamming距离或者余弦距离 )
This is a modified version of [DBoW2](https://github.com/dorian3d/DBoW2) and [DLib](https://github.com/dorian3d/DLib) library. All files included are BSD licensed.

* All code in **g2o** folder.  非线性 图 优化话算法
This is a modified version of [g2o](https://github.com/RainerKuemmerle/g2o). All files included are BSD licensed.

#####Library dependencies 

* * Pangolin (visualization and user interface)  可视化 **.
[MIT license](https://en.wikipedia.org/wiki/MIT_License).

*  *OpenCV 计算机视觉库 **.
BSD license.

*  *Eigen3 C++ 矩阵运算库 **.
For versions greater than 3.1.1 is MPL2, earlier versions are LGPLv3.

*  * ROS (Optional, only if you build Examples/ROS) 机器人操作系统库 可选**.
BSD license. In the manifest.xml the only declared package dependencies are roscpp, tf, sensor_msgs, image_transport, cv_bridge, which are all BSD licensed.





