# 目标跟踪 目标视觉跟踪(Visual Object Tracking)
[目标跟踪 论文代码](https://github.com/foolwood/benchmark_results)

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
       
       
    4. 深度学习（Deep ConvNet based）类方法.
 
    
