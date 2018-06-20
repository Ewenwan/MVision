# 基于3D卷积C3D提取特征再时序上上采样做 帧分类，然后预测存在行为的视频段并分类
[原作者代码](https://bitbucket.org/columbiadvmm/cdc/src)

[主页](http://www.ee.columbia.edu/ln/dvmm/researchProjects/cdc/cdc.html)

## CDC 用于未修剪视频中精确时间动作定位的卷积-反-卷积网络
[基于3D卷积C3D做帧分类，然后预测存在行为的视频段并分类](http://www.columbia.edu/~zs2262/files/research/cvpr17_CDC_zheng_slides.pdf)

![](https://img-blog.csdn.net/20180308102108902?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbmNsZ3NqMTAyOA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)


    CDC网络[13]是在C3D网络基础上，借鉴了FCN的思想。
    在C3D网络的后面增加了时间维度的上采样操作，做到了帧预测(frame level labeling)。
    
    1、第一次将卷积、反卷积操作应用到行为检测领域，CDC同时在空间下采样，在时间域上上采样。
    2、利用CDC网络结构可以做到端到端的学习。
    3、通过反卷积操作可以做到帧预测(Per-frame action labeling)。
![](https://img-blog.csdn.net/20180309224720118)

    CDC6 反卷积   
    3DCNN能够很好的学习时空的高级语义抽象，但是丢失了时间上的细粒度，
    众所周知的C3D架构输出视频的时序长度减小了8倍
    在像素级语义分割中，反卷积被证明是一种有效的图像和视频上采样的方法，
    用于产生与输入相同分辨率的输出。
    对于时序定位问题，输出的时序长度应该和输入视频一致，
    但是输出大小应该被减小到1x1。
![](https://img-blog.csdn.net/20180122152736012)
   
    网络步骤如下所示:
    输入的视频段是112x112xL，连续L帧112x112的图像
    经过C3D网络后，时间域上L下采样到 L/8, 空间上图像的大小由 112x112下采样到了4x4
    CDC6: 时间域上上采样到 L/4, 空间上继续下采样到 1x1
    CDC7: 时间域上上采样到 L/2
    CDC8：时间域上上采样到 L，而且全连接层用的是 4096xK+1, K是类别数
    softmax层
[ CDC 论文](http://dvmmweb.cs.columbia.edu/files/CVPR17_Zheng_CDC.pdf)

# 下载编译

[代码模型下载](https://bitbucket.org/columbiadvmm/cdc/downloads/)

    有训练好的模型参数，比较大
    
## 编译 
    cd ./CDC/
    make all -j
    make pycaffe
    
[Makefile.config 参考修改](https://github.com/Ewenwan/MVision/blob/master/darknect/caffe/caffe_src_change/Makefile.config)

[由于cudnn版本较新，编译可能出错，参考cudnn.hpp修改](https://github.com/Ewenwan/MVision/blob/master/darknect/caffe/caffe_src_change/cudnn.hpp)

# 运行demo
    1. 输入数据： demo/data/window  列表文件： demo/data/test.lst

    2. 每个数据样例都是 32帧长的视频片段. 
       这里保持和 C3D-v1.0 格式一致, 
       样例是以二进制文件形式(RGB值)，每一帧都有一个 4-th 长度的标签值;

    3.运行 demo: 
       cd demo
       ./xfeat.sh

    4. 输出特征文件夹 demo/feat

#  在 THUMOS 2014 数据集上处理

## 1. 数据预处理
    数据集地址：
        inputdir = '/DATA_ROOT/THUMOS14/test/all_frames_pervideo/'
    处理成二进制文件： 
        cd THUMOS14/predata/test 
        python gen_test_bin_and_list.py
## 2. CDC 网络预测
    cd THUMOS14/test
    需要使用 网络输出的特征
    提取特征：
    /CDC_root/model/thumos_CDC/convdeconv-TH14_iter_24390
## 3. 结构后 Post-process 
    cd THUMOS14/test/postprocess
    3步骤 matlab下
    1. run matlab step1_gen_test_metadata.m and will generate metadata.mat
      frmid: frame id in each video, starts with 1
      videoid: belongs to which video
      kept_frm_index: 
    
    2. run matlab step2_read_feat.m and will read all caffe outputs into two matlab matrixs:
    3. run matlab step3_gen_CDC_det.m to produce action segment instances prediction for temporal localization.
    
    




