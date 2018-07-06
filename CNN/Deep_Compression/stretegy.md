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

#### vgg16-ssd



#### mobilenet-ssd



#### squeezeet-ssd



### yolo 检测框架

#### yolo-v1


#### yolo-v2



#### yolo-v3


### fastert-rcnn 检测框架
#### vgg16-fastert-rcnn
    Faster R-CNN coco数据集上
    baseline mAP@.5  mAP@.5:.95
    VGG-16     41.5  21.5
    ResNet-101 48.4  27.2

# 实验

## 分类
### imagenet2012数据集 140G
    类别总数：1000类
    图片格式：256*256*3 彩色图；
    数据集结构： 130万训练图片 (732 to 1300 每一类)，
                10万测试图片(100/类)，
                5万验证图片(50/类)
下载地址：

    测试集 http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_test.tar    12.7G
    验证集 http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar     6.28G
    训练集 http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar   138G

    对应标签：
    http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz
    
训练时间：
    
    
数据集抽取: 
    

###  VGGnet模型
![](https://img-blog.csdn.net/20170715114221637?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxNDI4MTM5Mg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

    论文：
        https://arxiv.org/pdf/1409.1556.pdf

    权重和框架：
        vgg16：13个卷积层+3个全链接层=16
            http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel    528M
            https://github.com/Ewenwan/MVision/blob/master/CNN/SSD/VGG_ILSVRC_16_layers_deploy.prototxt

        vgg19：16个卷积层+3个全链接层=19
            http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel   548M
            https://github.com/Ewenwan/MVision/blob/master/CNN/SSD/VGG_ILSVRC_19_layers_deploy.prototxt
    分类结果(错误率)：两者多次度分类差不多
        top-1：24.8
        top-5：7.5

###  ResNet模型

    论文: 
        https://arxiv.org/pdf/1512.03385.pdf
    模型结构：最后一个 1000-d全连接层，其他均为卷积层，18/34 是两个3*3卷积，后面是 1*1 + 3*3 + 1*1
    权重文件： https://onedrive.live.com/?authkey=%21AAFW2-FVoxeVRck&id=4006CBB8476FF777%2117887&cid=4006CBB8476FF777  需要翻墙
              csdn上有下载 需要金币
              https://download.csdn.net/download/shaxinqing/10426907  ResNet-152-model.caffemodel
              https://download.csdn.net/download/zhs233/10311355      ResNet-101-model.caffemodel
              https://download.csdn.net/download/zhs233/10311350      ResNet-50-model.caffemodel
    prototxt文件：         
        https://github.com/Ewenwan/MVision/blob/master/CNN/ResNet/ResNet-50-deploy.prototxt
        https://github.com/Ewenwan/MVision/blob/master/CNN/ResNet/ResNet-101-deploy.prototxt
        https://github.com/Ewenwan/MVision/blob/master/CNN/ResNet/ResNet-152-deploy.prototxt
    分类结果(错误率)：    
        ResNet18: 
            top-1：27.88
            top-5：-
        ResNet34：
            top-1：21.53
            top-5：5.60   
        ResNet50:
            top-1：20.74
            top-5：5.25
        ResNet101:
            top-1：19.87
            top-5：4.60
        ResNet152: 
            top-1：19.38
            top-5：4.49   / 3.57

###  mobilenet模型

    MobileNets-v1:
        论文：https://arxiv.org/pdf/1704.04861.pdf
        分类准确率：
        top-1：70.81
        top-5：89.85 
        模型大小： 16.2 MB
        模型：https://github.com/shicai/MobileNet-Caffe/blob/master/mobilenet.caffemodel
        框架：https://github.com/Ewenwan/MVision/blob/master/CNN/MobileNet/mobilenet_v1_deploy.prototxt
    MobileNets-v2:
        论文：https://arxiv.org/pdf/1801.04381.pdf
        分类准确率：
            top-1：71.90%
            top-5：90.49%
        模型大小：13.5 MB
        模型：https://github.com/shicai/MobileNet-Caffe/blob/master/mobilenet_v2.caffemodel
        框架：https://github.com/Ewenwan/MVision/blob/master/CNN/MobileNet/mobilenet_v2_deploy.prototxt

### squeezeet模型
    论文：https://arxiv.org/pdf/1602.07360.pdf
    分类准确率：
        top-1：60.4%
        top-5：82.5%
    模型大小：4.8MB    
    模型：https://github.com/DeepScale/SqueezeNet/blob/master/SqueezeNet_v1.1/squeezenet_v1.1.caffemodel
    框架：https://github.com/Ewenwan/MVision/blob/master/CNN/SqueezeNet/squeezenet_v1.1_deploy.prototxt

### shufflenet模型
    论文：https://arxiv.org/pdf/1707.01083.pdf
    分类准确率：
        top-1：65.45%
        top-5：86.38%
    模型大小：7.04MB    
    模型：https://github.com/msnqqer/ShuffleNet/blob/master/shufflenet_1x_g3.caffemodel
    框架：https://github.com/Ewenwan/MVision/blob/master/CNN/ShuffleNet/shufflenet_1x_g3_deploy.prototxt



## 检测

### coco数据集 20G

    超过 20万 张图片，80种物体类别，图片大小不一，3通道彩色图像。
    所有的物体实例都用详细的分割mask进行了标注，
    共标注了超过 50万 个目标实体.
    
下载数据集   

    1. 下载 数据库API
     git clone https://github.com/pdollar/coco
     cd coco
    2. 创建 images文件夹 并下载 图像数据 解压
     在images文件夹下下载  点击链接可直接下载
     wget -c https://pjreddie.com/media/files/train2014.zip    12.5G
     wget -c https://pjreddie.com/media/files/val2014.zip      6.18G

     解压
     unzip -q train2014.zip
     unzip -q val2014.zip
    3. 下载标注文件等
      cd ..
      wget -c https://pjreddie.com/media/files/instances_train-val2014.zip
      wget -c https://pjreddie.com/media/files/coco/5k.part
      wget -c https://pjreddie.com/media/files/coco/trainvalno5k.part
      wget -c https://pjreddie.com/media/files/coco/labels.tgz
      sudo tar xzf labels.tgz                        标签
      sudo unzip -q instances_train-val2014.zip     分割  得到 annotations  实例分割

      生成训练/测试图像列表文件  darknet框架下格式
      paste <(awk "{print \"$PWD\"}" <5k.part) 5k.part | tr -d '\t' > 5k.txt   测试验证数据
      paste <(awk "{print \"$PWD\"}" <trainvalno5k.part) trainvalno5k.part | tr -d '\t' > trainvalno5k.txt  训练数据
    4.  caffe lmdb格式转换
      2014年
      train2014.zip 训练集图片
      val2014.zip   验证集图片
      instances_train-val2014.zip  总的json标签
      labels.tgz    darknet下 的 txt标签
      2017年 
        http://images.cocodataset.org/zips/train2017.zip 训练集图片
        http://images.cocodataset.org/zips/val2017.zip   验证集图片
        http://images.cocodataset.org/annotations/annotations_trainval2017.zip 总的json标签

        处理 ：
        下载 coco数据库处理脚本 
            git clone https://github.com/weiliu89/coco.git
            cd coco
            git checkout dev  # 必要 在 PythonAPI 或出现 scripts/ 文件夹 一些处理脚本
        安装：
            cd coco/PythonAPI
            python setup.py build_ext --inplace
            
        将总的json文件拆分成 各个图像的json
            python scripts/batch_split_annotation.py 
            
        获取 图片id 对应的图片尺寸大小 长宽
            python scripts/batch_get_image_size.py
            
        创建图片地址+标签地址的 列表文件    
            python data/coco/create_list.py
        生成lmdb 文件 and make soft links at examples/coco/
           ./create_data.sh
            
            
### vgg16-ssd 检测
    论文：https://arxiv.org/pdf/1512.02325.pdf
    检测准确度：
        SSD300：voc2007 map0.5: 77.2; coco上 map0.5: 43.1, map0.7: 25.8;
        SSD512: voc2007 map0.5: 79.8; coco上 map0.5: 48.5, map0.7: 30.3;
        
      YOLO-V2下的准确度
        SSD300：voc2007 map0.5: 74.3; coco上 map0.5: 41.2
        SSD500：voc2007 map0.5: 76.8; coco上 map0.5: 46.5
        
    模型：
        https://github.com/weiliu89/caffe/tree/ssd  有链接，需要翻墙，其他资源未找到。
        SSD300_VOC0712 
        SSD512_VOC0712 
        SSD300_COCO 
        SSD512_COCO 
    框架文件：
        COCO https://github.com/Ewenwan/MVision/blob/master/CNN/SSD/coco_vgg16-ssd-300-300/VGG_coco_SSD_300x300_deploy.prototxt
        VOC0712 https://github.com/Ewenwan/MVision/blob/master/CNN/SSD/SSD_300x300/ssd_33_deploy.prototxt
### yolo-v2 检测
    论文：https://arxiv.org/pdf/1612.08242.pdf
      YOLO-V2 的准确度(darknet下)
        YOLOv2：voc2007 map0.5: 76.8; 
        YOLOv2 544x544：voc2007 map0.5: 78.6; 
        YOLOv2 608x608:coco上 map0.5: 48.1
        
      caffe下 
     448*448尺寸caffeinemodel  https://pan.baidu.com/s/1c71EB-6A1xQb2ImOISZiHA password: 9u5v
      
#### 裁剪
#### 量化
