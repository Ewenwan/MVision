# 基于YOLO的3D目标检测：YOLO-6D

[论文 Real-Time Seamless Single Shot 6D Object Pose Prediction ](https://arxiv.org/pdf/1711.08848v2.pdf)

[代码](https://github.com/Ewenwan/singleshotpose/blob/master/README.md)

    2D图像的目标检测算法我们已经很熟悉了，物体在2D图像上存在一个2D的bounding box，
    我们的目标就是把它检测出来。而在3D空间中，
    物体也存在一个3D bounding box，如果将3D bounding box画在2D图像上.

    这个3D bounding box可以表示一个物体的姿态。那什么是物体的姿态？
    实际上就是物体在3D空间中的空间位置xyz，以及物体绕x轴，y轴和z轴旋转的角度。
    换言之，只要知道了物体在3D空间中的这六个自由度，就可以唯一确定物体的姿态。

    知道物体的姿态是很重要的。对于人来说，如果我们想要抓取一个物体，
    那么我们必须知道物体在3D空间中的空间位置xyz，但这个还不够，我们还要知道这个物体的旋转状态。
    知道了这些我们就可以愉快地抓取了。对于机器人而言也是一样，机械手的抓取动作也是需要物体的姿态的。
    因此研究物体的姿态有很重要的用途。

    Real-Time Seamless Single Shot 6D Object Pose Prediction 
    这篇文章提出了一种使用一张2D图片来预测物体6D姿态的方法。
    但是，并不是直接预测这个6D姿态，而是通过先预测3D bounding box在2D图像上的投影的1个中心点和8个角点，
    然后再由这9个点通过PNP算法计算得到6D姿态。我们这里不管怎么由PNP算法得到物体的6D姿态，
    而只关心怎么预测一个物体的3D bounding box在2D图像上的投影，即9个点的预测。

    
    整个网络采用的是yolo v2的框架。网络吃一张2D的图片（a），吐出一个SxSx(9x2+1+C)的3D tensor(e)。
    我们会将原始输入图片划分成SxS个cell（c），物体的中心点落在哪个cell，
    哪个cell就负责预测这个物体的9个坐标点（9x2），confidence（1）以及类别(C)，
    这个思路和yolo是一样的。
    
    模型输出的维度是13x13x(19+C)，这个19=9x2+1，表示9个点的坐标以及1个confidence值，
    另外C表示的是类别预测概率，总共C个类别。
  
    confidencel表示cell含有物体的概率以及bbox的准确度(confidence=P(object) *IOU)。
    我们知道，在yolo v2中，confidence的label实际上就是gt bbox和预测的bbox的IOU。
    但是在6D姿态估计中，如果要算IOU的话，需要在3D空间中算，这样会非常麻烦，
    因此本文提出了一种新的IOU计算方法，即定义了一个confidence函数：
    
    其中D(x)是预测的2D点坐标值与真实值之间的欧式距离，dth是提前设定的阈值，比如30pixel，
    alpha是超参，作者设置为2。从上图可以看出，当预测值与真实值越接近时候，D(x)越小，c(x)值越大，
    表示置信度越大。反之，表示置信度越小。需要注意的是，这里c(x)只是表示一个坐标点的c(x)，
    而一个物体有9个点，因此会计算出所有的c(x)然后求平均。
    
    
    坐标的意义
    上面讲到网络需要预测的9个点的坐标，包括8个角点和一个中心点。但是我们并不是直接预测坐标值，
    和yolo v2一样，我们预测的是相对于cell的偏移。不过中心点和角点还不一样，
    中心点的偏移一定会落在cell之内（因为中心点落在哪个cell哪个cell就负责预测这个物体），
    因此通过sigmoid函数将网络的输出压缩到0-1之间，但对于其他8个角点，是有可能落在cell之外的，
    
    

