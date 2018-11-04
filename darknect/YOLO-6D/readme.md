# 基于YOLO的3D目标检测：YOLO-6D
[论文 Real-Time Seamless Single Shot 6D Object Pose Prediction ](https://arxiv.org/pdf/1711.08848v2.pdf)

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

  
