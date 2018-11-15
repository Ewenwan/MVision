# 目标跟踪 目标视觉跟踪(Visual Object Tracking)
[目标跟踪 论文代码](https://github.com/foolwood/benchmark_results)

[目标跟踪 相关滤波方法 论文和代码](https://github.com/HEscop/TBCF)

[参考](https://github.com/Ewenwan/MVision/tree/master/3D_Object_Detection/Object_Tracking)

[参考2](https://www.zhihu.com/question/26493945/answer/156025576)

    目标跟踪，是通用单目标跟踪，第一帧给个矩形框，这个框在数据库里面是人工标注的，
    在实际情况下大多是检测算法的结果，然后需要跟踪算法在后续帧紧跟住这个框，
    以下是VOT(目标视觉跟踪(Visual Object Tracking))对跟踪算法的要求：
    
    无模型、短期、随意
    
    目标视觉跟踪(Visual Object Tracking)，大家比较公认分为两大类：
        生成(generative)模型方法 和 判别(discriminative)模型方法，
        目前比较流行的是判别类方法，也叫检测跟踪tracking-by-detection。
        
    1. 生成类方法，在当前帧对目标区域建模，下一帧寻找与模型最相似的区域就是预测位置，比较著名的有
       卡尔曼滤波，粒子滤波，mean-shift等。举个例子，从当前帧知道了目标区域80%是红色，20%是绿色，
       然后在下一帧，搜索算法就像无头苍蝇，到处去找最符合这个颜色比例的区域，推荐算法ASMS.
       ASMS与DAT并称“颜色双雄”(版权所有翻版必究)，都是仅颜色特征的算法而且速度很快.
       ASMS是VOT2015官方推荐的实时算法，平均帧率125FPS，在经典mean-shift框架下加入了尺度估计，经
       典颜色直方图特征，加入了两个先验(尺度不剧变+可能偏最大)作为正则项，和反向尺度一致性检查。
       
    2. 判别类方法，CV中的经典套路图像特征+机器学习， 当前帧以目标区域为正样本，背景区域为负样本，
       机器学习方法训练分类器，下一帧用训练好的分类器找最优区域.
       与生成类方法最大的区别是，分类器采用机器学习，训练中用到了背景信息，
       这样分类器就能专注区分前景和背景，所以判别类方法普遍都比生成类好。
       举个例子，在训练时告诉tracker目标80%是红色，20%是绿色，还告诉它背景中有橘红色，
       要格外注意别搞错了，这样的分类器知道更多信息，效果也相对更好。
       
    3. 相关滤波类方法correlation filter简称CF，也叫做discriminative correlation filter简称DCF.
       高速相关滤波类跟踪算法CSK, KCF/DCF, CN。
![](https://pic4.zhimg.com/80/v2-cd6759216ec7dc24a268978a7c950d23_hd.jpg)
        
        MOSSE是单通道灰度特征的相关滤波，CSK在MOSSE的基础上扩展了密集采样(加padding)和kernel-trick，
        KCF在CSK的基础上扩展了多通道梯度的HOG特征，CN在CSK的基础上扩展了多通道颜色的Color Names。
        HOG是梯度特征，而CN是颜色特征，两者可以互补，所以HOG+CN在近两年的跟踪算法中成为了hand-craft特征标配。
        最后，根据KCF/DCF的实验结果，讨论两个问题：
        
            1. 为什么只用单通道灰度特征的KCF和用了多通道HOG特征的KCF速度差异很小？
                第一，作者用了HOG的快速算法fHOG，来自Piotr's Computer Vision Matlab Toolbox，C代码而且做了SSE优化。
                     如对fHOG有疑问，请参考论文Object Detection with Discriminatively Trained Part Based Models第12页。
                第二，HOG特征常用cell size是4，这就意味着，100*100的图像，HOG特征图的维度只有25*25，而Raw pixels是灰度图归一化，
                     维度依然是100*100，我们简单算一下：27通道HOG特征的复杂度是27*625*log(625)=47180，单通道灰度特征的复杂度是
                     10000*log(10000)=40000，理论上也差不多，符合表格。看代码会发现，作者在扩展后目标区域面积较大时，
                     会先对提取到的图像块做因子2的下采样到50*50，这样复杂度就变成了2500*log(2500)=8495，下降了非常多。
                     那你可能会想，如果下采样再多一点，复杂度就更低了，但这是以牺牲跟踪精度为代价的，再举个例子，
                     如果图像块面积为200*200，先下采样到100*100，再提取HOG特征，分辨率降到了25*25，
                     这就意味着响应图的分辨率也是25*25，也就是说，响应图每位移1个像素，原始图像中跟踪框要移动8个像素，
                     这样就降低了跟踪精度。在精度要求不高时，完全可以稍微牺牲下精度提高帧率(但看起来真的不能再下采样了)。
          2. HOG特征的KCF和DCF哪个更好？大部分人都会认为KCF效果超过DCF，而且各属性的准确度都在DCF之上，
              然而，如果换个角度来看，以DCF为基准，再来看加了kernel-trick的KCF，mean precision仅提高了0.4%，
              而FPS下降了41%，这么看是不是挺惊讶的呢？除了图像块像素总数，KCF的复杂度还主要和kernel-trick相关。
              所以，下文中的CF方法如果没有kernel-trick，就简称基于DCF，
              如果加了kernel-trick，就简称基于KCF(剧透基本各占一
              
              一句话总结，别看那些五花八门的机器学习方法，那都是虚的，目标跟踪算法中特征才是最重要的.
              
              
    4. 深度学习（Deep ConvNet based）类方法.
 
    
