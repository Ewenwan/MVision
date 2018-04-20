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
        person
        bird, cat, cow, dog, horse, sheep
        aeroplane, bicycle, boat, bus, car, motorbike, train
        bottle, chair, dining table, potted plant, sofa, tv/monitor


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

    =========================================================
 ### 10.5 下载预训练分类网络参数 imagenet数据集的 分类网络参数
    from  darknet53 

    wget https://pjreddie.com/media/files/darknet53.conv.74



    =========================================================
  ### 10.6 . 在  VOC 训练
    ./darknet detector train cfg/voc.data cfg/yolov3-voc.cfg darknet53.conv.74


    ==========================================================
    =========================================================
