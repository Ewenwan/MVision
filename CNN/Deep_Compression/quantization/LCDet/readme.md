# 嵌入式CNN检测网络--LCDet 基于yolov1
[论文 LCDet: Low-Complexity Fully-Convolutional Neural Networks for Object Detection in Embedded System](https://arxiv.org/pdf/1705.05922.pdf)

    本文针对嵌入式设备以 YOLO-v1 为基础, 将 YOLO 最后两个全连接层更换为全卷积层，
    LCDet 中除了最后一层，其它层都用 ReLU 激活响应，YOLO用的是 LeakyReLU,
    采用 8-bit 定点 TF-model 达到实时检测，精度也不错。
![](https://img-blog.csdn.net/20170601144859998?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvemhhbmdqdW5oaXQ=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
    
    
    
