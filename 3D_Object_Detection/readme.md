#  3D Object Detection 3D目标检测
[kitti 3d目标检测算法排名](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)

[frustum-pointnets 3d点云+2d检测框+pointnet3d标注框 ](https://github.com/Ewenwan/frustum-pointnets)

[KinectFusion ElasticFusion](https://github.com/Ewenwan/MVision/blob/master/3D_Object_Detection/pdf/KinectFusion%20%E5%92%8C%20ElasticFusion%20%E4%B8%89%E7%BB%B4%E9%87%8D%E5%BB%BA%E6%96%B9%E6%B3%95_6_3.pdf)

[YOLO-6D](https://github.com/Ewenwan/MVision/blob/master/darknect/YOLO-6D/readme.md)

[传统算法 3D目标识别---局部特征描述子介绍](https://blog.csdn.net/FireMicrocosm/article/details/78059151)

[object-3D目标检测算法调研（基于激光雷达、kitti数据集）](https://blog.csdn.net/sum_nap/article/details/80966979)


# MaskFusion rgbd-slam + 语义分割mask-rcnn 
[论文 MaskFusion: Real-Time Recognition, Tracking and Reconstruction of Multiple Moving Objects](https://arxiv.org/pdf/1804.09194.pdf)

[主页](http://visual.cs.ucl.ac.uk/pubs/maskfusion/index.html)

[视频](http://visual.cs.ucl.ac.uk/pubs/maskfusion/MaskFusion.webm)

[代码](https://github.com/Ewenwan/maskfusion)

    本文提出的MaskFusion算法可以解决这两个问题，首先，可以从Object-level理解环境，
    在准确分割运动目标的同时，可以识别、检测、跟踪以及重建目标。
    
    分割算法由两部分组成：
     1. 2d语义分割： Mask RCNN:提供多达80类的目标识别等
     2. 利用Depth以及Surface Normal等信息向Mask RCNN提供更精确的目标边缘分割。
     
    上述算法的结果输入到本文的Dynamic SLAM框架中。
       使用Instance-aware semantic segmentation比使用pixel-level semantic segmentation更好。
       目标Mask更精确，并且可以把不同的object instance分配到同一object category。
     
    本文的作者又提到了现在SLAM所面临的另一个大问题：Dynamic的问题。
    作者提到，本文提出的算法在两个方面具有优势：
        相比于这些算法，本文的算法可以解决Dynamic Scene的问题。
        本文提出的算法具有Object-level Semantic的能力。
        
        
    所以总的来说，作者就是与那些Semantic Mapping的方法比Dynamic Scene的处理能力，
    与那些Dynamic Scene SLAM的方法比Semantic能力，在或者就是比速度。
    确实，前面的作者都只关注Static Scene， 现在看来，
    实际的SLAM中还需要解决Dynamic Scene(Moving Objects存在)的问题。}
    
![](https://github.com/Ewenwan/texs/blob/master/PaperReader/SemanticSLAM/MaskFusion0.png)
    
    每新来一帧数据，整个算法包括以下几个流程：

    1. 跟踪 Tracking
       每一个Object的6 DoF通过最小化一个能量函数来确定，这个能量函数由两部分组成：
          a. 几何的ICP Error;
          b. Photometric cost。
       此外，作者仅对那些Non-static Model进行Track。
       最后，作者比较了两种确定Object是否运动的方法：
          a. Based on Motioin Incosistency
          b. Treating objects which are being touched by a person as dynamic
          
    2. 分割 Segmentation
       使用了Mask RCNN和一个基于Depth Discontinuities and surface normals 的分割算法。
       前者有两个缺点：物体边界不精确、运行不实时。
       后者可以弥补这两个缺点， 但可能会Oversegment objects。
       
    3. 融合 Fusion
       就是把Object的几何结构与labels结合起来。


# 工具
[Computer-vision dataset tools that I am using or working on 轨迹处理 误差分析](https://github.com/Ewenwan/dataset-tools)



# ElasticFusion 
[代码](https://github.com/Ewenwan/ElasticFusion)

# KinectFusion

