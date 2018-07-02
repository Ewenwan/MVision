# 数据
## 分类数据集
### a. minist 手写字体数据集
    10类手写字体分类数据集，图片大小32*32pixs
### b. cifar10 / cifar 100  小图片分类数据集 
    cifar10 ：  10类常见物体， 小图片分类数据集 ， 图片大小32*32*3 pixs = 3073个字节，
                6万张 = 5万张训练 + 1万张测试，分成5个训练batches，和一个测试batch，每个包含1万张照片。
                飞机，汽车，鸟，猫，鹿，狗，青蛙，马，船，卡车。
                下载地址： https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
                
    cifar 100： 100类常见物体， 小图片分类数据集 ， 图片大小32*32pixs，
                6万张 = 5万张训练 + 1万张测试，分成5个训练batches，和一个测试batch，每个包含1万张照片。
                每种类别的图片数量少了。

### c. imagenet2012数据集 100+G ImageNet图像分类大赛
    ImageNet大规模视觉识别挑战赛（ILSVRC），软件程序竞相正确 分类 、 检测物体 和 场景识别检测
    http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_test.tar
    http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar
    http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar
    http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_devkit_t12.tar
    http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_bbox_train_v2.tar
    
    ImageNet图像分类大赛
    比赛设置：
        1000类图像分类问题，训练数据集126万张图像，验证集5万张，测试集10万张（标注未公布）。
        2012，2013，2014均采用了该数据集。
        评价标准采用top-5错误率，即对一张图像预测5个类别，只要有一个和人工标注类别相同就算对，否则算错。
        
        

## 检测数据集



# 模型
![](https://img-blog.csdn.net/20170313131253663?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd3NwYmE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


    ImageNet2012比赛冠军 (AlexNet)   ~ 60954656 params (top-5错误率16.4%，使用额外数据可达到15.3%，8层神经网络）
                        AlexNet是现代深度CNN的奠基之作。
                        2012年，Hinton的学生Alex Krizhevsky提出了深度卷积神经网络模型AlexNet。
                        首次在CNN中成功应用了ReLU、Dropout和LRN等Trick。

     ImageNet2014比赛亚军 VGGnet (Go deeper and deeper) (top-5错误率7.3%，19层神经网络）
                        VGGNet是牛津大学计算机视觉组（Visual Geometry Group）和Google DeepMind公司的研究员一起研发的的深度卷积神经网络。                   
    ImageNet2014比赛冠军 GoogLeNet (Inception Modules) ~ 11176896 params (top-5错误率6.7%，22层神经网络）
                        Inception V1有22层深，比AlexNet的8层或者VGGNet的19层还要更深。
                        但其计算量只有15亿次浮点运算，同时只有500万的参数量，

                        Inception V1参数少但效果好的原因除了模型层数更深、表达能力更强外，
                        还有两点：
                        一是去除了最后的全连接层，用全局平均池化层（即将图片尺寸变为1*1）来取代它。
                            全连接层几乎占据了AlexNet或VGGNet中90%的参数量，而且会引起过拟合，
                            去除全连接层后模型训练更快并且减轻了过拟合。
                        二是Inception V1中精心设计的Inception Module提高了参数的利用效率，
                            借鉴了Network In Network的思想，
                            形象的解释就是Inception Module本身如同大网络中的一个小网络，
                            其结构可以反复堆叠在一起形成大网络。

                        其中有4个分支：
                            第一个分支对输入进行1*1的卷积，这其实也是NIN中提出的一个重要结构。
                                1*1的卷积是一个非常优秀的结构，
                                它可以跨通道组织信息，提高网络的表达能力，同时可以对输出通道升维和降维。
                            第二个分支先使用了1*1卷积，然后连接3*3卷积，相当于进行了两次特征变换。
                            第三个分支类似，先是1*1的卷积，然后连接5*5卷积。最后一个分支则是3*3最大池化后直接使用1*1卷积。

                            2016年2月， Inception V4（top-5错误率3.08%）



    ImageNet2015年的冠军（ResNet，top-5错误率3.57%，152层神经网络）
                        ResNet（Residual Neural Network）由微软研究院的Kaiming He等4名华人提出，
                        通过使用Residual Unit成功训练152层深的神经网络
                        （加入直通通路，使得梯度可以传播，网络可以更深，类似思想LSTM 中的状态直连）。

                        ResNet最初的灵感出自这个问题：在不断加神经网络的深度时，
                           会出现一个Degradation的问题，
                           即准确率会先上升然后达到饱和，再持续增加深度则会导致准确率下降。





# 实验
