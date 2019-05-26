# 视觉词袋模型分类 Bag-of-Features-Framework Bag of Features(BoF)图像分类实践

[penCV探索之路（二十八）：Bag of Features(BoF)图像分类实践](https://www.cnblogs.com/skyfsm/p/8097397.html)

[代码](https://github.com/Ewenwan/Bag-of-Features-Framework/blob/master/README.md)

在深度学习在图像识别任务上大放异彩之前，词袋模型Bag of Features一直是各类比赛的首选方法

在2012年之前，词袋模型是VOC竞赛分类算法的基本框架，几乎所有算法都是基于词袋模型的，可以这么说，词袋模型在图像分类中统治了很多年。虽然现在深度学习在图像识别任务中的效果更胜一筹，但是我们也不要忘记在10年前，Bag of Features的框架曾经也引领过一个时代。那这篇文章就是要重温BoF这个经典框架，并从实践上看看它在图像物体分类中效果到底如何。

其实Bag of Features 是Bag of Words在图像识别领域的延伸，Bag of Words最初产生于自然处理领域，通过建模文档中单词出现的频率来对文档进行描述与表达。

词包模型还有一个起源就是纹理检测（texture recognition）,有些图像是由一些重复的基础纹理元素图案所组成，所以我们也可以将这些图案做成频率直方图，形成词包模型。

词包模型于2004年首次被引入计算机视觉领域，由此开始大量集中于词包模型的研究，在各类图像识别比赛中也大放异彩，逐渐形成了由下面4部分组成的标准物体分类框架：

      底层特征提取
      特征编码
      特征汇聚
      使用SVM等分类器进行分类

