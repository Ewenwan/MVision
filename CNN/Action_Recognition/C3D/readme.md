# 3D卷积视频分类
[代码参考](https://github.com/Ewenwan/C3D)


###  3D卷积 C3D Network
#### 提出 C3D
[Learning spatiotemporal features with 3d convolutional networks](https://arxiv.org/pdf/1412.0767.pdf)

[C3D论文笔记](https://blog.csdn.net/wzmsltw/article/details/61192243)

[C3D_caffe 代码](https://github.com/facebook/C3D)

    C3D是facebook的一个工作，采用3D卷积和3D Pooling构建了网络。
    通过3D卷积，C3D可以直接处理视频（或者说是视频帧的volume）
    实验效果：UCF101-85.2% 可以看出其在UCF101上的效果距离two stream方法还有不小差距。
             我认为这主要是网络结构造成的，C3D中的网络结构为自己设计的简单结构，如下图所示。

    速度：
            C3D的最大优势在于其速度，在文章中其速度为314fps。而实际上这是基于两年前的显卡了。
    用Nvidia 1080显卡可以达到600fps以上。
    所以C3D的效率是要远远高于其他方法的，个人认为这使得C3D有着很好的应用前景。

#### 改进  I3D[Facebook]
    即基于inception-V1模型，将2D卷积扩展到3D卷积。
[Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://arxiv.org/pdf/1705.07750.pdf)

#### T3D 时空 3d卷积
[Temporal 3D ConvNets:New Architecture and Transfer Learning for Video Classificati](https://arxiv.org/pdf/1711.08200.pdf)

    该论文值得注意的，
        一方面是采用了3D densenet，区别于之前的inception和Resnet结构；
        另一方面，TTL层，即使用不同尺度的卷积（inception思想）来捕捉讯息。
#### P3D  [MSRA]
[Learning Spatio-Temporal Representation with Pseudo-3D Residual Networks](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/10/iccv_p3d_camera.pdf)

[博客](https://blog.csdn.net/u014380165/article/details/78986416)

    改进ResNet内部连接中的卷积形式。然后，超深网络，一般人显然只能空有想法，望而却步。


# 1. C3D特征提取

##  1.1 命令参数介绍

     官方GitHub项目上同时提供了C3D-v1.0和C3D-v1.1两个版本，以下方法适用于v1.0

     官方提供的特征提取demo路径为~/C3D-master/C3D-v1.0/examples/c3d_feature_extraction

     在这个路径下，执行c3d_sport1m_feature_extraction_video.sh
     或
     c3d_sport1m_feature_extraction_frm.sh 可以分别从 视频提取特征 和从 图片提取特征的 demo

     打开c3d_sport1m_feature_extraction_video.sh 文件，
     除去一些用来生成文件夹的指令，
     可以看到启动C3D的命令如下：
     
     GLOG_logtosterr=1 
     ../../build/tools/extract_image_features.bin
     prototxt/c3d_sport1m_feature_extractor_video.prototxt 
     conv3d_deepnetA_sport1m_iter_1900000 0 50 1 
     prototxt/output_list_video_prefix.txt fc7-1 fc6-1 prob
     
     其中
      a) ../../build/tools/extract_image_features.bin
          是提取特征的可执行文件，示例命令中使用了相对路径，如果在其他路径下调用注意进行对应的修改
      b) prototxt/c3d_sport1m_feature_extractor_video.prototxt caffe网络配置文件和数据集路径等
      c) conv3d_deepnetA_sport1m_iter_1900000 这是预训练模型文件，根据自己的需求做对应的修改
      d) 接下来的三项数字是：0 50 1，分别是gpu_id，mini_batch_size 和 number_of_mini_batches。
         gpu_id是在计算机具有多块GPU时指定使用哪一块GPU的，默认是0，如果将这一项的值置为-1则启动CPU模式。
         需要注意，如果需要调整batch size，在prototxt文档中也要进行相应的修改
      e) prototxt/output_list_video_prefix.txt 是输出前缀文件，下面会详细介绍
      f) fc7-1 fc6-1 prob 是卷积层特征输出名称 要提取哪一层的特征依序写在这里即可
      
## 1.2 prototxt文档

     prototxt/c3d_sport1m_feature_extractor_video.prototxt是这个demo所使用的prototxt文档

     第8行：
     source: "prototxt/input_list_frm.txt"
     这是记录输入文件路径的文档。在这个demo中，prototxt/input_list_frm.txt对应的是以图片作为输入时的文档，
     而prototxt/input_list_video.txt对应的是以视频作为输入时的文档。
     
     以prototxt/input_list_frm.txt为例，
     该文档格式如下：
        input/frm/v_ApplyEyeMakeup_g01_c01/ 1 0
        input/frm/v_ApplyEyeMakeup_g01_c01/ 17 0
        input/frm/v_ApplyEyeMakeup_g01_c01/ 33 0
        input/frm/v_ApplyEyeMakeup_g01_c01/ 49 0
        input/frm/v_ApplyEyeMakeup_g01_c01/ 65 0
        input/frm/v_ApplyEyeMakeup_g01_c01/ 81 0
        input/frm/v_ApplyEyeMakeup_g01_c01/ 97 0
        input/frm/v_ApplyEyeMakeup_g01_c01/ 113 0
        input/frm/v_ApplyEyeMakeup_g01_c01/ 129 0
        input/frm/v_ApplyEyeMakeup_g01_c01/ 145 0
        input/frm/v_BaseballPitch_g01_c01/ 1 0
        input/frm/v_BaseballPitch_g01_c01/ 17 0
        input/frm/v_BaseballPitch_g01_c01/ 33 0
        input/frm/v_BaseballPitch_g01_c01/ 49 0
        input/frm/v_BaseballPitch_g01_c01/ 65 0
        input/frm/v_BaseballPitch_g01_c01/ 81 0
        
     其中input/frm/v_ApplyEyeMakeup_g01_c01/是保存图片的路径，
     后面的第一个数字表示从哪一帧开始提取特征，
     最后的数字表示该行对应的类别。
     由于这是提取特征而非训练，类别填写什么都不要紧，只要有就行

     多少帧提取一次特征，是由prototxt/input_list_frm.txt中第17行new_length一项参数决定的。
     例如上面例子中的视频一共有165帧，那么最后一行对应的145帧开始提取特征，
     取16帧，使用145帧-161帧的数据。在这里如果取用的帧的编号超过总帧数165，
     则会报错，要注意这一点

     如果输入时视频，则参考prototxt/input_list_video.txt。
     需要注意的是，输入为视频时帧的序号是从0开始计算的。

     第9行：
     use_image: true
     如果输入时图片，则为true，如果输入时视频，则为false。
     
     第10行
     mean_file: "fb_train16_128_mean.binaryproto"
     这里是使用的均值文件的路径，根据所使用的模型生成或选择均值文件即可
     另外也可根据需求修改其他参数。 
     
## 1.3 输出前缀文件
     参照prototxt/output_list_video_prefix.txt生成输出前缀文件，
     可以根据需求进行自定义，只要注意该文件要和prototxt/input_list_frm.txt
     输入文件清单的行数相对应即可

## 1.4 其他注意事项

     输出的特征文件所保存的路径必须自己生成，C3D不会创建文件夹

     如果提示“out of memory” 可以尝试减小batch size

     提取的特征是二进制文件，需要进行格式转换才能正常处理

     其他的注意事项可以参考官方的用户指南

# 2. C3D训练和fine-tune

     训练和fine-tune的官方demo的路径分别是

     ~/C3D-master/C3D-v1.0/examples/c3d_train_ucf101
     ~/C3D-master/C3D-v1.0/examples/c3d_finetuning
     所使用的prototxt和inputlist等文件参照特征提取和demo修改即可.
     

     
