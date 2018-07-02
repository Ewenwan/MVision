# 数据
## 分类数据集
### a. minist 手写字体数据集 1998年提出
    10类手写字体分类数据集，图片大小28*28*1 pixs 灰度图
    6万 训练图片 and 1万测试图片
    
### b. cifar10 / cifar 100  小图片分类数据集  2009年提出
    cifar10 ：  10类常见物体， 小图片分类数据集 ， 图片大小32*32*3 pixs = 3073个字节，
                6万张 = 5万张训练(5000/类) + 1万张测试(1000/类)，
                分成5个训练batches，和一个测试batch，每个包含1万张照片。
                飞机，汽车，鸟，猫，鹿，狗，青蛙，马，船，卡车。
                下载地址： https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
                
    cifar 100： 100类常见物体， 小图片分类数据集 ， 图片大小32*32pixs，
                6万张 = 5万张训练 + 1万张测试，分成5个训练batches，和一个测试batch，每个包含1万张照片。
                每种类别的图片数量少了。

### c. imagenet2012数据集 100+G ImageNet图像分类大赛 2010年起 2012年稳定 
    1000类， 256*256*3 彩色图；
    130万训练图片 (732 to 1300 每一类)，10万测试图片(100/类)，5万验证图片(50/类)
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
        评价标准采用top-5错误率，即对一张图像预测5个类别(最高得分)，只要有一个和人工标注类别相同就算对，否则算错。
        top-1，为最高的预测类别和标签匹配才正确，否则算错误。
        
    ImageNet数据下的detection数据集，200类，400,000张图像，350,000个bounding box。 
    由于一些目标之间有着强烈的关系而非独立存在，
    在特定场景下检测某种目标是是否有意义的，
    因此精确的位置信息比bounding box更加重要。
    
## 检测数据集
### SVHN（Street View House Number）Dateset 来源于谷歌街景门牌号码 适用于 OCR 相关方向 
    一个真实世界（谷歌街景）的街道门牌号数字识别数据集。超过600000张带有标签的数字图像。

### voc 2007检测数据集   (2005-2012)
        1.1万张图片，20类物体，2.7万个物体标记框，其中7千个框包含实例分割信息
        包含标签和 边框数据 以及语义分割数据
        
        给定自然图片， 从中识别出特定物体。 <object-class> <x> <y> <width> <height>
        待识别的物体有20类：
        囊括了车、人、猫、狗等20类常见目标。训练样本较少、场景变化多端，非常具有挑战性。
          aeroplane    飞机
          bicycle      自行车
          bird         鸟
          boat         船舶
          bottle       瓶子
          bus          公交车
          car          汽车
          cat          猫
          chair        椅子
          cow          奶牛
          diningtable  餐桌
          dog          狗
          horse        马
          motorbike    摩托车
          person       人
          pottedplant  盆栽
          sheep        羊
          sofa         沙发
          train        火车
          tvmonitor    电视机

### Caltech Pedestrian Dataset 加理工(caltech)行人检测 数据集
        35万个标记框。
        该数据库是目前规模较大的行人数据库，采用车载摄像头拍摄，
        约10个小时左右，视频的分辨率为640×480，30帧/秒。
        标注了约250,000帧（约137分钟），350000个矩形框，2300个行人，
        另外还对矩形框之间的时间对应关系及其遮挡的情况进行标注。
        数据集分为set00~set10，其中set00~set05为训练集，set06~set10为测试集（标注信息尚未公开）。

### coco      80类 检测数据集 包含标签和边框数据
    微软发布的COCO数据库，80类目标检测数据集.
    超过 20万 张图片，80种物体类别。
    所有的物体实例都用详细的分割mask进行了标注，共标注了超过 50万 个目标实体.
    
     l1 = ["人","自行车","汽车","摩托车","飞机","公交车",
          "火车","卡车","船","交通灯","消防栓",
          "停止标志","停车计时器","长凳","鸟","猫","狗",
          "马","羊","牛","大象","熊","斑马","长颈鹿",
          "背包","雨伞","手提包","领带","手提箱","飞盘","滑雪",
          "滑雪板","体育用球","风筝","棒球棒","棒球手套",
          "滑板","冲浪板","网球拍","瓶子","红酒杯","杯子",
          "叉子","小刀","勺子","碗","香蕉","苹果","三明治",
          "橘子","西兰花","胡萝卜","热狗","披萨","甜甜圈","蛋糕",
          "椅子","沙发","盆栽","床","餐桌","厕所","显示器",
          "笔记本","鼠标","遥控器","键盘","手机","微波炉","烤箱",
          "吐司机","水槽","冰箱","书","闹钟","花瓶","剪刀",
          "玩具熊","吹风机","牙刷"]   
## Google 开源的数据集
    900万张图片，6000类
    800万个视频(50万小时)， 4800类
    


# 模型
## 目标分类 
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
    
    移动端模型：
    
     mobilenet
     
     squeezeet
     
     shufflenet
     
    

## 目标检测

### SSD检测框架

### vgg16-ssd



### mobilenet-ssd



### squeezeet-ssd



### yolo 检测框架

### yolo-v1


### yolo-v2



### yolo-v3


### fastert-rcnn检测框架



# 实验
