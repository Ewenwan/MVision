  # yolo darknet
    =============================================
    =============================================
  ## 0.项目主页
    https://pjreddie.com/darknet/yolo/


    =============================================
    =============================================
  ## 1.安装 编译darknet
    git clone https://github.com/pjreddie/darknet
    cd darknet
    make
    
    安装了 opencv之后 可以打开opencv的编译选项
    还有多线程 openMP选项
    OPENCV=1
    OPENMP=1

    problem:
    /usr/bin/ld: 找不到 -lippicv
    solution:

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
    =========================================================
 ### 10.5 下载预训练分类网络参数 imagenet数据集的 分类网络参数
    from  darknet53 

    wget https://pjreddie.com/media/files/darknet53.conv.74



    =========================================================
  ### 10.6 . 在  VOC 训练
    ./darknet detector train cfg/voc.data cfg/yolov3-voc.cfg darknet53.conv.74


    ==========================================================
    =========================================================
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


