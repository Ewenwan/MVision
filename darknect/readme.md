  # yolo darknet
    =============================================
    =============================================
  ## 0.项目主页
[darknet](https://pjreddie.com/darknet/yolo/)


    =============================================
    =============================================
  ## 1.安装 编译darknet
    git clone https://github.com/pjreddie/darknet
    cd darknet
    make
    
[opencv](https://github.com/Ewenwan/MVision/blob/master/opencv_app/readme.md)安装

    安装了 opencv之后 可以打开opencv的编译选项
    还有多线程 openMP选项
    OPENCV=1
    OPENMP=1

    问题    problem:
    /usr/bin/ld: 找不到 -lippicv
    解决办法 solution:

    pkg-config加载库的路径是/usr/local/lib,我们去这这个路径下看看，
    发现没有-lippicv对应的库，别的选项都有对应的库，然后我们把-lippicv对应的库（libippicv.a）
    放到这个路径下就好啦了。

    我的liboppicv.a 在../opencv-3.1.0/3rdparty/ippicv/unpack/ippicv_lnx/lib/intel64/liboppicv.a
    这个路径下。

    你的也在你自己opencv文件夹的对应路径下。
    先cd 到上面这个路径下，然后sudo cp liboppicv.a /usr/local/lib 
    将这个库文件复制到/usr/local/lib下就好了。
    
    查看 opencv 是否安装成功
    pkg-config --modversion opencv 
    
    
### 如需GPU 注意千万不要忘了修改nvcc  实际cuda 安装路径
    nvcc=/usr/local/cuda-8.0/bin/nvcc
    
[caffe的安装](https://blog.csdn.net/yhaolpz/article/details/71375762)

### scikit-image 安装
    命令行安装 sudo apt-get install python-skimage
    
    源码安装
    git clone https://github.com/scikit-image/scikit-image.git

    安装所有必需的依赖项:
    sudo apt-get install python-matplotlib python-numpy python-pil python-scipy python-

    使用已经安装好的的编译器:
    sudo apt-get install build-essential cython


    cd scikit-image
    如果你的编译工具完全的话，直接运行:
    pip install -U -e .
    
[cython 0.25 版本](https://packages.ubuntu.com/artful/cython)
    
    安装 sudo dpkg -i cython_0.25.2-2build3_amd64.deb 
    
    更新:
    git pull  # Grab latest source
    python setup.py build_ext -i  # Compile any modified extensions

###  from google.protobuf.internal import enum_type_wrapper ImportError: No module named google.protobuf
    sudo apt-get install python-protobuf

============================================
    =============================================
  ## 2.下载训练好的权重weight文件
    wget https://pjreddie.com/media/files/yolov3.weights

    =============================================
    =============================================
  ## 3.执行检测网络  标出框已经分类 和 置信度
    ./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg
    输出信息：
    (模型结构 和 置信度 检测时间 等信息  cpu上 6-12s/张 )
    layer     filters    size              input                output
        0 conv     32  3 x 3 / 1   416 x 416 x   3   ->   416 x 416 x  32  0.299 BFLOPs
        1 conv     64  3 x 3 / 2   416 x 416 x  32   ->   208 x 208 x  64  1.595 BFLOPs
        .......
      105 conv    255  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 255  0.353 BFLOPs
      106 detection
    truth_thresh: Using default '1.000000'
    Loading weights from yolov3.weights...Done!
    data/dog.jpg: Predicted in 0.029329 seconds.
    dog: 99%
    truck: 93%
    bicycle: 99%

    ==============================================
    ==============================================
  ## 4. 其他图片
    data/eagle.jpg, data/dog.jpg, data/person.jpg, data/horses.jpg


    ==============================================
    ==============================================
  ## 5、较长的命令行
    ./darknet detector test cfg/coco.data cfg/yolov3.cfg yolov3.weights data/dog.jpg

    ==============================================
    ==============================================
  ## 6、检测多幅图像  会提示输入图像 检测完成 再次提示输入图像
    ./darknet detect cfg/yolov3.cfg yolov3.weights


    ==============================================
    ==============================================
  ## 7、改变检测阈值
    ./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg -thresh 0

    ==============================================
    ==============================================
  ## 8、网络摄像头 实时检测
    ./darknet detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights

    ==============================================
    ==============================================
  ## 9、时时检测视频
    ./darknet detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights <video file>


    ==============================================
    ==============================================
  ## 10.在 Pascal VOC 数据集上训练
    ====================================
  ### 10.1 Pascal VOC数据集介绍：
    给定自然图片， 从中识别出特定物体。
    待识别的物体有20类：
    囊括了车、人、猫、狗等20类常见目标。训练样本较少、场景变化多端，非常具有挑战性。
      aeroplane  
      bicycle
      bird
      boat
      bottle
      bus
      car
      cat
      chair
      cow
      diningtable
      dog
      horse
      motorbike
      person
      pottedplant
      sheep
      sofa
      train
      tvmonitor

    ===================================
  ### 10.2 下载数据集：
    wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
    wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
    wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
    tar xf VOCtrainval_11-May-2012.tar
    tar xf VOCtrainval_06-Nov-2007.tar
    tar xf VOCtest_06-Nov-2007.tar
    存在于 VOCdevkit/ 子目录下


    ===================================
  ### 10.3创建标记文件 .txt ：
    每个框 类别 一行  x, y, width, and height  与图像长和宽相关
    <object-class> <x> <y> <width> <height>

    运行标记文件 脚本
    run scripts/voc_label.py
    python voc_label.py

    会在  VOCdevkit/VOC2007/labels/ and VOCdevkit/VOC2012/labels/
    下生成一些列文件
    ls
    2007_test.txt   VOCdevkit
    2007_train.txt  voc_label.py
    2007_val.txt    VOCtest_06-Nov-2007.tar
    2012_train.txt  VOCtrainval_06-Nov-2007.tar
    2012_val.txt    VOCtrainval_11-May-2012.tar

    除去2007_test.txt 生成一个文件
    cat 2007_train.txt 2007_val.txt 2012_*.txt > train.txt

    ==================================
  ### 10.4 修改 数据配置文件 cfg/voc.data
      classes= 20
      train  = <path-to-voc>/train.txt
      valid  = <path-to-voc>2007_test.txt
      names = data/voc.names
      backup = backup
## 我的
    classes= 20
    train  = /home/sujun/ewenwan/software/darknet/data/voc/my_train_data.txt
    valid  = /home/sujun/ewenwan/software/darknet/data/voc/2007_test.txt
    names = data/voc.names
    backup = backup
    
### 以及 网络配置文件 
    [net]
    # Testing
    # batch=1 # bigger gpu memory cost higher 
    #  subdivisions=1

    # Training   训练
    batch=64          # 一次训练使用多张图片
    subdivisions=16   # 分成16次载入gpu内存 也就是一次载入 4张图片
    width=416
    height=416
    channels=3
    momentum=0.9      # 动量 
    decay=0.0005      # 衰减
    angle=0# 图片旋转
    saturation = 1.5# 图像预处理
    exposure = 1.5
    hue=.1

    learning_rate=0.0001#  bigger easy spread学习率
    burn_in=1000
    max_batches = 50200
    policy=steps
    steps=40000,45000#逐步降低 学习率 牛顿下山法
    scales=.1,.1
    ...
    ...
    
    =========================================================
 ### 10.5 下载预训练分类网络参数 imagenet数据集的 分类网络参数
    from  darknet53 

    wget https://pjreddie.com/media/files/darknet53.conv.74



    =========================================================
  ### 10.6 . 在  VOC 训练
    从零开始
    ./darknet detector train cfg/voc.data cfg/yolov3-voc.cfg darknet53.conv.74
    断点继续训练
    ./darknet detector train cfg/voc_my_cfg.data cfg/yolov3-voc.cfg backup/yolov3-voc.backup
    
  ### 10.7 使用训练结果 测试、
    ./darknet detect cfg/yolov3-voc.cfg backup/yolov3-voc_20000.weights data/dog.jpg
    ==========================================================
    =========================================================
    
  ### 剩下的就是等待了。
    需要注意的是，如果学习率设置的比较大，训练结果很容易发散，训练过程输出的log会有nan字样，需要减小学习率后再进行训练。
    
  ### 先 单GPU 训练
    ./darknet detector train cfg/voc.data cfg/yolov3-voc.cfg darknet53.conv.74 2>1 | tee paul_train_log.txt
  ### 多GPU训练技巧
    darknet支持多GPU，使用多GPU训练可以极大加速训练速度。
  ### 单GPU与多GPU的切换技巧
    在darknet上使用多GPU训练需要一定技巧，盲目使用多GPU训练会悲剧的发现损失一直在下降、
    recall在上升，然而Obj几乎为零,最终得到的权重文件无法预测出bounding box。
    
    使用多GPU训练前需要先用单GPU训练至Obj有稳定上升的趋势后（我一般在obj大于0.1后切换）
    再使用backup中备份的weights通过多GPU继续训练。
    一般情况下使用单GPU训练1000个迭代即可切换到多GPU。
    
    ./darknet detector train cfg/voc_my_cfg.data cfg/yolov3-voc.cfg backup/yolov3-voc_1000.weights -gpus 0,1,2,3 2>1 | tee paul_train_log.txt
    
    nvidia-smi 差看GPU使用情况
    
  ### 使用多GPU时的学习率
    使用多GPU训练时，学习率是使用单GPU训练的n倍，n是使用GPU的个数
    
  ### 可视化训练过程的中间参数
    训练log中各参数的意义
    Region Avg IOU：平均的IOU，代表预测的bounding box和ground truth的交集与并集之比，期望该值趋近于1。
    Class:是标注物体的概率，期望该值趋近于1.
    Obj：期望该值趋近于1.
    No Obj:期望该值越来越小但不为零.
    Avg Recall：期望该值趋近1
    avg：平均损失，期望该值趋近于0
    
    使用train_loss_visualization.py脚本可以绘制loss变化曲线。
    
    保存log时会生成两个文件，文件1里保存的是网络加载信息和checkout点保存信息，paul_train_log.txt中保存的是训练信息。

    1、删除log开头的三行：
    0,1,2,3,4,5,6,7
    yolo-paul
    Learning Rate: 1e-05, Momentum: 0.9, Decay: 0.0005
    
    2、删除log的结尾几行，使最后一行为batch的输出，如：
    497001: 0.863348, 0.863348 avg, 0.001200 rate, 5.422251 seconds, 107352216 images

    3、执行extract_log.py脚本，格式化log。
    
    最终log格式：
    Loaded: 5.588888 seconds
    Region Avg IOU: 0.649881, Class: 0.854394, Obj: 0.476559, No Obj: 0.007302, Avg Recall: 0.737705,  count: 61
    Region Avg IOU: 0.671544, Class: 0.959081, Obj: 0.523326, No Obj: 0.006902, Avg Recall: 0.780000,  count: 50
    Region Avg IOU: 0.525841, Class: 0.815314, Obj: 0.449031, No Obj: 0.006602, Avg Recall: 0.484375,  count: 64
    Region Avg IOU: 0.583596, Class: 0.830763, Obj: 0.377681, No Obj: 0.007916, Avg Recall: 0.629214,  count: 89
    Region Avg IOU: 0.651377, Class: 0.908635, Obj: 0.460094, No Obj: 0.008060, Avg Recall: 0.753425,  count: 73
    Region Avg IOU: 0.571363, Class: 0.880554, Obj: 0.341659, No Obj: 0.007820, Avg Recall: 0.633663,  count: 101
    Region Avg IOU: 0.585424, Class: 0.935552, Obj: 0.358635, No Obj: 0.008192, Avg Recall: 0.644860,  count: 107
    Region Avg IOU: 0.599972, Class: 0.832793, Obj: 0.382910, No Obj: 0.009005, Avg Recall: 0.650602,  count: 83
    497001: 0.863348, 0.863348 avg, 0.000012 rate, 5.422251 seconds, 107352216 images


    4、修改train_loss_visualization.py中lines为log行数，并根据需要修改要跳过的行数。
    skiprows=[x for x in range(lines) if ((x%10!=9) |(x<1000))]
    
    运行train_loss_visualization.py会在脚本所在路径生成avg_loss.png
    
    从损失变化曲线可以看出，模型在100000万次迭代后损失下降速度非常慢，几乎没有下降。
    结合log和cfg文件发现，我自定义的学习率变化策略在十万次迭代时会减小十倍，
    十万次迭代后学习率下降到非常小的程度，导致损失下降速度降低。
    修改cfg中的学习率变化策略，10万次迭代时不改变学习率，30万次时再降低。
    
    我使用迭代97000次时的备份的checkout点来继续训练。
    ./darknet detector train cfg/voc_my_cfg.data cfg/yolov3-voc.cfg backup/yolov3-voc_97000.weights -gpus 0,1,2,3 2>1 | tee paul_train_log.txt
    
    除了可视化loss，还可以可视化Avg IOU，Avg Recall等参数。
    可视化’Region Avg IOU’, ‘Class’, ‘Obj’, ‘No Obj’, ‘Avg Recall’,’count’
    这些参数可以使用脚本train_iou_visualization.py，使用方式和train_loss_visualization.py相同。
    

### 使用验证集评估模型
    评估模型可以使用命令valid（只有预测结果，没有评价预测是否正确）或recall，这两个命令都无法满足我的需求，我实现了category命令做性能评估。
    我使用迭代97000次时的备份的checkout点来继续训练。
    
    在voc_my_cfg.data 末尾添加
    eval = imagenet #有voc、coco、imagenet三种模式
    修改Detector.c文件validate_detector函数，修改阈值（默认.005）
    float thresh = .1;
    重新编译然后执行命令
    ./darknet detector valid cfg/voc_my_cfg.data cfg/yolov3-voc.cfg backup/yolov3-voc_final.weights


### 想要查看recall可以使用recall命令。

    修改 Detector.c文件的validate_detector_recall函数：

    1、修改阈值：
    float thresh = .25;

    2、修改验证集路径：
    list *plist = get_paths("/mnt/large4t/pengchong_data/Data/Paul/filelist/val.txt");

    3、增加Precision
    //fprintf(stderr, "%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, correct, total, (float)proposals/(i+1), avg_iou*100/total, 100.*correct/total);
    fprintf(stderr, "ID:%5d Correct:%5d Total:%5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\t", i, correct, total, (float)proposals/(i+1), avg_iou*100/total, 100.*correct/total);
    fprintf(stderr, "proposals:%5d\tPrecision:%.2f%%\n",proposals,100.*correct/(float)proposals
    4、执行命令
    ./darknet detector recall cfg/voc_my_cfg.data cfg/yolov3-voc.cfg backup/yolov3-voc_final.weights
[训练参考](https://blog.csdn.net/hrsstudy/article/details/65644517?utm_source=itdadao&utm_medium=referral)
## 11 在coco数据集上训练
[数据集主页](http://cocodataset.org/)

    微软发布的COCO数据库, 除了图片以外还提供物体检测, 分割(segmentation)和对图像的语义文本描述信息.
[COCO数据库的网址是: MS COCO API - ](http://mscoco.org/) 
[Github网址 -  ](https://github.com/pdollar/coco)
[关于API更多的细节在网站: ](http://mscoco.org/dataset/#download) 

    数据库提供Matlab, Python和Lua的API接口. 其中matlab和python的API接口可以提供完整的图像标签数据的加载, 
    parsing和可视化.此外,网站还提供了数据相关的文章, 教程等. 在使用COCO数据库提供的API和demo时, 需要首先下载COCO的图像和标签数据.

### 11.1 COCO的数据标注信息包括: 
      - 类别标志 
      - 类别数量区分 
      - 像素级的分割 
      COCO数据集有超过 200,000 张图片，80种物体类别. 所有的物体实例都用详细的分割mask进行了标注，共标注了超过 500,000 个物体实体.     
      {    
      person  # 1    
      vehicle 交通工具 #8        
      { bicycle         自行车
      car             小汽车       
      motorcycle      摩托车
      airplane        飞机       
      bus             公交车
      train           火车       
      truck           卡车
      boat}           船    
      outdoor  室外#5        
      { traffic light   交通灯     
      fire hydrant    消防栓     
      stop sign       
      parking meter      
      bench}    
      animal  动物 #10        
      { bird       
      cat      
      dog      
      horse       
      sheep      
      cow       
      elephant      
      bear       
      zebra      
      giraffe}   
      accessory 饰品 #5        
      { backpack 背包       
      umbrella 雨伞       
      handbag 手提包       
      tie 领带       
      suitcase 手提箱 }   
      sports  运动 #10        
      { frisbee      
      skis      
      snowboard       
      sports ball       
      kite        
      baseball bat       
      baseball glove       
      skateboard        
      surfboard       
      tennis racket        } 

      kitchen  厨房 #7       
      { bottle        
      wine glass       
      cup       
      fork        
      knife       
      spoon        
      bowl        }  
      food  食物#10        
      { banana        
      apple       
      sandwich        
      orange       
      broccoli       
      carrot        
      hot dog        
      pizza       
      donut       
      cake        }    
      furniture 家具 #6        
      { chair       
      couch       
      potted plant       
      bed        
      dining table       
      toilet        }    
      electronic 电子产品 #6        
      { tv        
      laptop       
      mouse        
      remote        
      keyboard        
      cell phone        }   
      appliance 家用电器 #5        
      { microwave       
      oven        
      toaster       
      sink        
      refrigerator        }    
      indoor  室内物品#7        
      { book        
      clock       
      vase     
      scissors        
      teddy bear        
      hair drier       
      toothbrush        }}
      
## 11.2下载数据集   
    cp scripts/get_coco_dataset.sh data
    cd data
    bash get_coco_dataset.sh
    
## 11.3修改 coco数据集的配置文件
    vim cfg/coco.data
    classes= 80
    train  = <path-to-coco>/trainvalno5k.txt
    valid  = <path-to-coco>/5k.txt
    names = data/coco.names
    backup = backup
    
    
    
## 修改模型配置文件 
    cp cfg/yolo.cfg my_yolov3.cfg
    vim my_yolov3.cfg
## 训练
    ./darknet detector train cfg/coco.data cfg/my_yolov3.cfg darknet53.conv.74
### 多gpu训练
    ./darknet detector train cfg/coco.data cfg/yolov3.cfg darknet53.conv.74 -gpus 0,1,2,3
### 中断后 断点接着 训练
    ./darknet detector train cfg/coco.data cfg/yolov3.cfg backup/yolov3.backup -gpus 0,1,2,3


