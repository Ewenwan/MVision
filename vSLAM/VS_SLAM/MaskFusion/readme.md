# . MaskFusion ElasticFusion(RGBD-SLAM)　+ 语义分割mask-rcnn
[论文 MaskFusion: Real-Time Recognition, Tracking and Reconstruction of Multiple Moving Objects ](https://arxiv.org/pdf/1804.09194.pdf)

    本文提出的MaskFusion算法可以解决这两个问题，首先，可以从Object-level理解环境，
    在准确分割运动目标的同时，可以识别、检测、跟踪以及重建目标。
    分割算法由两部分组成：
    Mask RCNN:提供多达80类的目标识别等,利用Depth以及Surface Normal等信息向Mask RCNN提供更精确的目标边缘分割。
    上述算法的结果输入到本文的Dynamic SLAM框架中。
     使用Instance-aware semantic segmentation比使用pixel-level semantic segmentation更好。
     目标Mask更精确，并且可以把不同的object instance分配到同一object category
     
    本文的作者又提到了现在SLAM所面临的另一个大问题：Dynamic的问题。
    作者提到，本文提出的算法在两个方面具有优势：
        相比于这些算法，本文的算法可以解决Dynamic Scene的问题。
        本文提出的算法具有Object-level Semantic的能力。
        
        
    所以总的来说，作者就是与那些Semantic Mapping的方法比Dynamic Scene的处理能力，
    与那些Dynamic Scene SLAM的方法比Semantic能力，在或者就是比速度。
    确实，前面的作者都只关注Static Scene， 现在看来，
    实际的SLAM中还需要解决Dynamic Scene(Moving Objects存在)的问题。}
