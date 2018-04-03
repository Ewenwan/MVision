# objdetect 模块. 物体检测
## 级联分类器
    级联分类器 （CascadeClassifier） 
    
    AdaBoost强分类器串接
    
    级联分类器是将若干个分类器进行连接，从而构成一种多项式级的强分类器。
    从弱分类器到强分类器的级联（AdaBoost 集成学习  改变训练集）
    级联分类器使用前要先进行训练，怎么训练？
    用目标的特征值去训练，对于人脸来说，通常使用Haar特征进行训练。

    【1】提出积分图(Integral image)的概念。在该论文中作者使用的是Haar-like特征，
        然后使用积分图能够非常迅速的计算不同尺度上的Haar-like特征。
    【2】使用AdaBoost作为特征选择的方法选择少量的特征在使用AdaBoost构造强分类器。
    【3】以级联的方式，从简单到 复杂 逐步 串联 强分类器，形成 级联分类器。

    级联分类器。该分类器由若干个简单的AdaBoost强分类器串接得来。
    假设AdaBoost分类器要实现99%的正确率，1%的误检率需要200维特征，
    而实现具有99.9%正确率和50%的误检率的AdaBoost分类器仅需要10维特征，
    那么通过级联，假设10级级联，最终得到的正确率和误检率分别为:
    (99.9%)^10 = 99%
    (0.5)^10   = 0.1

    检测体系：是以现实中很大一副图片作为输入，然后对图片中进行多区域，多尺度的检测，
    所谓多区域，是要对图片划分多块，对每个块进行检测，由于训练的时候一般图片都是20*20左右的小图片，
    所以对于大的人脸，还需要进行多尺度的检测。多尺度检测一般有两种策略，一种是不改变搜索窗口的大小，
    而不断缩放图片，这种方法需要对每个缩放后的图片进行区域特征值的运算，效率不高，而另一种方法，
    是不断初始化搜索窗口size为训练时的图片大小，不断扩大搜索窗口进行搜索。
    在区域放大的过程中会出现同一个人脸被多次检测，这需要进行区域的合并。
    无论哪一种搜索方法，都会为输入图片输出大量的子窗口图像，
    这些子窗口图像经过筛选式级联分类器会不断地被每个节点筛选，抛弃或通过。


    级联分类器的策略是，将若干个强分类器由简单到复杂排列，
    希望经过训练使每个强分类器都有较高检测率，而误识率可以放低。

    AdaBoost训练出来的强分类器一般具有较小的误识率，但检测率并不很高，
    一般情况下，高检测率会导致高误识率，这是强分类阈值的划分导致的，
    要提高强分类器的检测率既要降低阈值，要降低强分类器的误识率就要提高阈值，
    这是个矛盾的事情。据参考论文的实验结果，
    增加分类器个数可以在提高强分类器检测率的同时降低误识率，
    所以级联分类器在训练时要考虑如下平衡，一是弱分类器的个数和计算时间的平衡，
    二是强分类器检测率和误识率之间的平衡。


    // 级联分类器 类
    CascadeClassifier face_cascade;
    // 加载级联分类器
    face_cascade.load( face_cascade_name );
    // 多尺寸检测人脸

    std::vector<Rect> faces;//检测到的人脸 矩形区域 左下点坐标 长和宽
    Mat frame_gray;
    cvtColor( frame, frame_gray, CV_BGR2GRAY );//转换成灰度图
    equalizeHist( frame_gray, frame_gray );    //直方图均衡画
    //-- 多尺寸检测人脸
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
    // image：当然是输入图像了，要求是8位无符号图像，即灰度图
    //objects：输出向量容器（保存检测到的物体矩阵）
    //scaleFactor：每张图像缩小的尽寸比例 1.1 即每次搜索窗口扩大10%
    //minNeighbors：每个候选矩阵应包含的像素领域
    //flags:表示此参数模型是否更新标志位；
    //minSize ：表示最小的目标检测尺寸；
    //maxSize：表示最大的目标检测尺寸；


    Haar和 LBP (Local Binary Patterns)两种特征，并易于增加其他的特征。
    与Haar特征相比，LBP特征是整数特征，因此训练和检测过程都会比Haar特征快几倍。
    LBP和Haar特征用于检测的准确率，是依赖训练过程中的训练数据的质量和训练参数。
    训练一个与基于Haar特征同样准确度的LBP的分类器是可能的。


## 级联分类器 的训练
    与其他分类器模型的训练方法类似，同样需要训练数据与测试数据；
    其中训练数据包含正样本pos与负样本neg。
    训练程序opencv_haartraining.exe与opencv_traincascade.exe
    对输入的数据格式是有要求的，所以需要相关的辅助程序：

    opencv_createsamples 用来准备训练用的正样本数据和测试数据。
    opencv_createsamples 能够生成能被opencv_haartraining 和 
    opencv_traincascade 程序支持的正样本数据。
    它的输出为以 *.vec 为扩展名的文件，该文件以二进制方式存储图像。

    所以opencv级联分类器训练与测试可分为以下四个步骤：

      【1】准备训练数据
      【2】训练级联分类器
      【3】测试分类器性能
      【4】利用训练好的分类器进行目标检测
      
### 1、准备训练数据

    注：以行人数据为例，介绍分类器的训练
    1.1准备正样本
     正样本由opencv_createsamples生成。
     正样本可以由包含待检测物体的一张图片生成，也可由一系列标记好的图像生成。
     首先将所有的正样本放在一个文件夹，如图所示1。其中，pos.dat文件为所有图像的列表文件，
     格式如图2所示：
     其中，第一列为图像名，第二列为该图像中正样本的个数，最后的为正样本在图像中的位置以及需要抠出的正样本的尺寸。
     pos.dat文件的生成方式：在dos窗口进入pos文件夹，输入dir /b > pos.dat ; 
     这样只能生成文件名列表，后面的正样本个数与位置尺寸还需手动添加。
      位置尺寸 labelImg 

    opencv_createsamples.exe程序的命令行参数：

    -info <collection_file_name> 	描述物体所在图像以及大小位置的描述文件。
    -vec <vec_file_name>		输出文件，内含用于训练的正样本。
    -img <image_file_name>		输入图像文件名（例如一个公司的标志）。
    -bg<background_file_name>	背景图像的描述文件，文件中包含一系列的图像文件名，这些图像将被随机选作物体的背景。
    -num<number_of_samples>		生成的正样本的数目。
    -bgcolor<background_color>	背景颜色（目前为灰度图）；背景颜色表示透明颜色。
            因为图像压缩可造成颜色偏差，颜色的容差可以由-bgthresh指定。
            所有处于bgcolor-bgthresh和bgcolor+bgthresh之间的像素都被设置为透明像素。
    -bgthresh <background_color_threshold>
    -inv				如果指定该标志，前景图像的颜色将翻转。
    -randinv			如果指定该标志，颜色将随机地翻转。
    -maxidev<max_intensity_deviation>	前景样本里像素的亮度梯度的最大值。
    -maxxangle <max_x_rotation_angle>	X轴最大旋转角度，必须以弧度为单位。
    -maxyangle <max_y_rotation_angle>	Y轴最大旋转角度，必须以弧度为单位。
    -maxzangle<max_z_rotation_angle>	Z轴最大旋转角度，必须以弧度为单位。
    -show		很有用的调试选项。如果指定该选项，每个样本都将被显示。如果按下Esc键，程序将继续创建样本但不再显示。
    -w <sample_width>	输出样本的宽度（以像素为单位）。
    -h<sample_height>	输出样本的高度（以像素为单位）。

    1.3准备负样本

       负样本可以是任意图像，但是这些图像中不能包含待检测的物体。
       用于抠取负样本的图像文件名被列在一个neg.dat文件中。
       生成方式与正样本相同，但仅仅包含文件名列表就可以了。
       这个文件是纯文本文件，每行是一个文件名（包括相对目录和文件名）这些图像可以是不同的尺寸，
       但是图像尺寸应该比训练窗口的尺寸大，
       因为这些图像将被用于抠取负样本，并将负样本缩小到训练窗口大小。


### 2、训练级联分类器

     OpenCV提供了两个可以训练的级联分类器的程序：opencv_haartraining与opencv_traincascade。
     opencv_haartraining是一个将被弃用的程序；
     opencv_traincascade是一个新程序。
     opencv_traincascade程序 命令行参数如下所示：

    1.通用参数：
       -data <cascade_dir_name> 目录名，如不存在训练程序会创建它，用于存放训练好的分类器。
       -vec <vec_file_name>     包含正样本的vec文件名（由opencv_createsamples程序生成）。
       -bg <background_file_name>            背景描述文件，也就是包含负样本文件名的那个描述文件。
       -numPos <number_of_positive_samples>  每级分类器训练时所用的正样本数目。
       -numNeg <number_of_negative_samples>  每级分类器训练时所用的负样本数目，可以大于 -bg 指定的图片数目。
       -numStages <number_of_stages>         训练的分类器的级数。
       -precalcValBufSize<precalculated_vals_buffer_size_in_Mb>  缓存大小，用于存储预先计算的特征值(feature values)，单位为MB。
       -precalcIdxBufSize<precalculated_idxs_buffer_size_in_Mb>  缓存大小，用于存储预先计算的特征索引(feature indices)，单位为MB。
                         内存越大，训练时间越短。
       -baseFormatSave    这个参数仅在使用Haar特征时有效。如果指定这个参数，那么级联分类器将以老的格式存储。

    2.级联参数：
       -stageType <BOOST(default)>   级别（stage）参数。目前只支持将BOOST分类器作为级别的类型。
       -featureType<{HAAR(default), LBP}>  特征的类型： HAAR - 类Haar特征； LBP - 局部纹理模式特征。
       -w <sampleWidth>
       -h <sampleHeight>  训练样本的尺寸（单位为像素）。必须跟训练样本创建（使用 opencv_createsamples 程序创建）时的尺寸保持一致。

    3.分类器参数：        
     -bt <{DAB, RAB, LB,GAB(default)}>  Boosted分类器参数：
     DAB - Discrete AdaBoost, RAB - Real AdaBoost, LB - LogitBoost, GAB -Gentle AdaBoost。 Boosted分类器的类型：
     -minHitRate<min_hit_rate>   分类器的每一级希望得到的最小检测率。总的检测率大约为 min_hit_rate^number_of_stages。
     -maxFalseAlarmRate<max_false_alarm_rate>   分类器的每一级希望得到的最大误检率。总的误检率大约为 max_false_alarm_rate^number_of_stages.
     -weightTrimRate <weight_trim_rate>
      Specifies whether trimmingshould be used and its weight.一个还不错的数值是0.95。
     -maxDepth <max_depth_of_weak_tree>   弱分类器树最大的深度。一个还不错的数值是1，是二叉树（stumps）。
     -maxWeakCount<max_weak_tree_count>   每一级中的弱分类器的最大数目。The boostedclassifier (stage) will have so many weak trees (<=maxWeakCount), as neededto achieve the given -maxFalseAlarmRate.

    4.类Haar特征参数：
      -mode <BASIC (default) |CORE | ALL>  选择训练过程中使用的Haar特征的类型。 BASIC 只使用右上特征， ALL使用所有右上特征和45度旋转特征。

    5.LBP特征参数：

    LBP特征无参数。



### 3.测试分类器性能
            opencv_performance 可以用来评估分类器的质量，但只能评估 opencv_haartraining 
      输出的分类器。它读入一组标注好的图像，运行分类器并报告性能，如检测到物体的数目，
      漏检的数目，误检的数目，以及其他信息。同样准备测试数据集test，生成图像列表文件，
      格式与训练者正样本图像列表相同，需要标注目标文件的个数与位置。


     opencv_haartraining程序训练一个分类器模型

    opencv_haartraining 的命令行参数如下：
    －data<dir_name>    	存放训练好的分类器的路径名。
    －vec<vec_file_name> 	正样本文件名（由trainingssamples程序或者由其他的方法创建的）
    －bg<background_file_name>		背景描述文件。
    －npos<number_of_positive_samples>，
    －nneg<number_of_negative_samples>	用来训练每一个分类器阶段的正/负样本。合理的值是：nPos = 7000;nNeg= 3000
    －nstages<number_of_stages>		训练的阶段数。
    －nsplits<number_of_splits>		决定用于阶段分类器的弱分类器。如果1，则一个简单的stump classifier被使用。
              如果是2或者更多，则带有number_of_splits个内部节点的CART分类器被使用。
    －mem<memory_in_MB>			预先计算的以MB为单位的可用内存。内存越大则训练的速度越快。
    －sym（default）
    －nonsym指				定训练的目标对象是否垂直对称。垂直对称提高目标的训练速度。例如，正面部是垂直对称的。
    －minhitrate<min_hit_rate>		每个阶段分类器需要的最小的命中率。总的命中率为min_hit_rate的number_of_stages次方。
    －maxfalsealarm<max_false_alarm_rate>	没有阶段分类器的最大错误报警率。总的错误警告率为max_false_alarm_rate的number_of_stages次方。
    －weighttrimming<weight_trimming>	指定是否使用权修正和使用多大的权修正。一个基本的选择是0.9
    －eqw
    －mode<basic(default)|core|all>		选择用来训练的haar特征集的种类。basic仅仅使用垂直特征。all使用垂直和45度角旋转特征。
    －w<sample_width>
    －h<sample_height>			训练样本的尺寸，（以像素为单位）。必须和训练样本创建的尺寸相同


    opencv_performance测试分类器模型


    opencv_performance 的命令行参数如下所示：
        -data <classifier_directory_name>	训练好的分类器
        -info <collection_file_name>   	描述物体所在图像以及大小位置的描述文件
        -maxSizeDiff <max_size_difference =1.500000>
        -maxPosDiff <max_position_difference =0.300000>
        -sf <scale_factor = 1.200000>
        -ni 				选项抑制创建的图像文件的检测
        -nos <number_of_stages = -1>
        -rs <roc_size = 40>]
        -w <sample_width = 24>
        -h <sample_height = 24>
