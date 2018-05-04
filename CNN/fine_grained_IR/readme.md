# 深度学习: 细粒度图像识别 (fine-grained image recognition)
> 细粒度图像识别 (fine-grained image recognition)，即 精细化分类，不同品种的花、鸟、狗、汽车

        细粒度图像分类(Fine-Grained Categorization),
        又被称作子类别图像分类(Sub-Category Recognition),
        是近年来计算机视觉、模式识别等领域一个
        非常热门的研究课题. 其目的是对粗粒度的大类别
        进行更加细致的子类划分, 但由于子类别间细微的
        类间差异和较大的类内差异, 较之普通的图像分类
        任务, 细粒度图像分类难度更大.
[南大 细粒度图像分类 综述](https://cs.nju.edu.cn/wujx/paper/AAS_2017.pdf)

[细粒度图像识别算法研究 百度文库参考](https://wenku.baidu.com/view/f4c00ec3a76e58fafbb00310.html)

[百度开放平台 细粒度图像分类](https://ai.baidu.com/tech/imagerecognition/fine_grained)

[识别后对植物/汽车详细介绍的接口为欧拉蜜-百科 自然语言理解 NLU 图灵的也可以](http://cn.olami.ai/open/website/home/home_show)

[细粒度分类-车辆分类](https://blog.csdn.net/u012938704/article/details/68934403)

# 精细化分类
    识别出物体的大类别（比如：计算机、手机、水杯等）较易，但如果进一步去判断更为精细化的物体分类名称，则难度极大。
    最大的挑战在于，同一大类别下 不同 子类别 间的 视觉差异 极小。因此，精细化分类 所需的图像分辨率 较高。
# 目前，精细化分类的方法主要有以下两类：
    1. 基于图像重要区域定位的方法：
       该方法集中探讨如何利用弱监督的信息自动找到图像中有判别力的区域，从而达到精细化分类的目的。
    2. 基于图像精细化特征表达的方法：
       该方法提出使用高维度的图像特征（如：bilinear vector）对图像信息进行高阶编码，以达到准确分类的目的。
# 按照其使用的监督信息的多少 分为 强监督 和 弱监督 信息的细粒度图像分类模型
[细粒度图像分析进展综述](https://blog.csdn.net/u011746554/article/details/75096674)
## 1. 基于强监督信息的细粒度图像分类模型
    是指，在模型训练时，为了获得更好的分类精度，
    除了图像的类别标签外，
    还使用了物体标注框（object bounding box）和
    部位标注点（part annotation）等额外的人工标注信息。
    算法框架有：
    1. Part-based R-CNN 
[局部驱动的区域卷积网络 Part-based R-CNN](https://arxiv.org/pdf/1407.3867.pdf)   

    2. Pose Normalized CNN

    3. Mask-CNN 
[Mask-CNN](https://arxiv.org/pdf/1605.06878.pdf)    
    
## 2. 基于弱监督信息的细粒度图像分类模型
    在模型训练时仅使用图像级别标注信息，而不再使用额外的part annotation信息时，也能取得与强监督分类模型可比的分类精度。
    仅使用图像的类别标签 + 物体标注框。
    思路同强监督分类模型类似，也需要借助全局和局部信息来做细粒度级别的分类。
    而区别在于，弱监督细粒度分类希望在不借助part annotation的情况下，也可以做到较好的局部信息的捕捉。
    算法框架有：
    1. Two Level Attention Model
[两个不同层次的特征](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Xiao_The_Application_of_2015_CVPR_paper.pdf)    

    2. Constellations 
[Constellations ](https://arxiv.org/pdf/1504.08289v3.pd)    

    3. Bilinear CNN
[Bilinear CNN](https://arxiv.org/pdf/1504.07889.pdf) 

# 算法框架 
    1. CNN  特征提取网络(科目卷积层 、 属目卷积层、种目卷积层)    提取不同层面的特征
    2. APN  注意力建议网络   得到不同的关注区域
    3. DCNN 卷积细粒度特征描述网络
    4. 全连接层之后得到粗细粒度互补的层次化特征表达，再通过 分类网络softmax 输出结果



# 注意力模型（Attention Model） 注意力机制
    被广泛使用在自然语言处理、图像识别及语音识别等各种不同类型的深度学习任务中，
    是深度学习技术中最值得关注与深入了解的核心技术之一。  
    视觉注意力机制是人类视觉所特有的大脑信号处理机制。
    人类视觉通过快速扫描全局图像，获得需要重点关注的目标区域，也就是一般所说的注意力焦点，
    而后对这一区域投入更多注意力资源，以获取更多所需要关注目标的细节信息，而抑制其他无用信息。
    这是人类利用有限的注意力资源从大量信息中快速筛选出高价值信息的手段，
    是人类在长期进化中形成的一种生存机制，人类视觉注意力机制极大地提高了视觉信息处理的效率与准确性。
    
    把Attention仍然理解为从大量信息中有选择地筛选出少量重要信息并聚焦到这些重要信息上，
    忽略大多不重要的信息，这种思路仍然成立。
    聚焦的过程体现在权重系数的计算上，权重越大越聚焦于其对应的Value值上，
    即权重代表了信息的重要性，而Value是其对应的信息。
    
# 图片描述（Image-Caption）
    是一种典型的图文结合的深度学习应用，输入一张图片，人工智能系统输出一句描述句子，语义等价地描述图片所示内容。    
    可以使用Encoder-Decoder框架来解决任务目标。
    1. 此时编码部分Encoder输入部分是一张图片，一般会用CNN来对图片进行特征抽取；
    2. 解码Decoder部分使用RNN或者LSTM和注意力机制来输出自然语言句子。

# RA-CNN
    MSRA通过观察发现，对于精细化物体分类问题，其实形态、轮廓特征显得不那么重要，而细节纹理特征则起到了主导作用。
    因此提出了 “将判别力区域的定位和精细化特征的学习联合进行优化” 的构想，从而让两者在学习的过程中相互强化，
    也由此诞生了 “Recurrent Attention Convolutional Neural Network”（RA-CNN，基于递归注意力模型的卷积神经网络）网络结构。
    RA-CNN 网络可以更精准地找到图像中有判别力的子区域，然后采用高分辨率、精细化特征描述这些区域，进而大大提高精细化物体分类的精度： 
[论文地址](http://openaccess.thecvf.com/content_cvpr_2017/papers/Fu_Look_Closer_to_CVPR_2017_paper.pdf)

## RA-CNN思想 
    1. 首先原图大尺度图像通过 CNN 卷积网络 提取特征，
       一部分进过APN（Attention Proposal Net 注意力建议网络）得到注意力中心框（感兴趣区域，例如上半身区域），
       另一部分通过全连接层再经过softmax归一化分类概率输出；
    2. 对第一步得到的注意力中心框（感兴趣区域，例如上半身区域），再进行1的步骤，
       得到更小的注意力中心框，和分类概率；
    3. 对第二步得到的注意力中心框（感兴趣区域，例如头部区域），通过卷积网络提取特征，
       通过全连接层再经过softmax归一化分类概率输出；
