# ncnn 使用 
[源码仓库](https://github.com/Tencent/ncnn)

安装编译：

    git clone https://github.com/Tencent/ncnn.git
    cd ncnn 
    gedit CMakeLists.txt
    拉到最后
        add_subdirectory(examples)  # 取消两行注释
        add_subdirectory(benchmark)
        add_subdirectory(src)
        if(NOT ANDROID AND NOT IOS)
        add_subdirectory(tools)
        endif()
    mkdir build
    cd build
    cmake ..
    make 



# 1. 图像分类网络

    mobilenet

    squeezenet

# 2. 目标检测网络

    mobilenet-ssd-voc
    mobilenetv2-ssdlite-voc
    mobilenetv2-ssdlite-coco
    squeezenet-ssd-voc
    
    yolov2


![](http://oz3lw4cji.bkt.clouddn.com/0b7c0d0fb225db788d66d3e2a0468fca.png)

# 检测效果
    1. mobilenet-ssd-voc
        ./mobilenetssd dog.jpeg 
        detection time: 320 ms
        dog = 0.99912 at 470.92 44.57 295.02 x 550.56
        
![](https://github.com/Ewenwan/MVision/blob/master/CNN/HighPerformanceComputing/example/img/mobilenet-ssd-voc-dog_det.png)
     
    2. queezenet-ssd-voc
        ./squeezenetssd dog.jpeg 
        detection time: 379 ms
        dog = 0.29287 at 501.58 50.59 257.39 x 291.96
        dog = 0.28615 at 486.13 47.62 311.03 x 542.53
![](https://github.com/Ewenwan/MVision/blob/master/CNN/HighPerformanceComputing/example/img/squeezenet-ssd-voc-dog-det.png)
    
    3. mobilenetv2ssdlite
        ./mobilenetv2ssdlite dog.jpeg 
        detection time: 187 ms
        dog = 0.97443 at 476.85 64.94 295.40 x 533.33
![](https://github.com/Ewenwan/MVision/blob/master/CNN/HighPerformanceComputing/example/img/mobilenetv2-ssdlite-voc-doe-det.png)    
        


# 一 、 ncnn使用
[ncnn_wiki 指南](https://github.com/Tencent/ncnn/wiki)

## 在Ubuntu上安装使用NCNN 


### 1. 下载编译源码
      git clone https://github.com/Tencent/ncnn.git
      下载完成后，需要对源码进行编译
      修改CMakeLists.txt 文件 打开 一些文件的编译开关
      
            ##############################################
            add_subdirectory(examples)
            # add_subdirectory(benchmark)
            add_subdirectory(src)
            if(NOT ANDROID AND NOT IOS)
            add_subdirectory(tools)
            endif()
            
      开始编译:
            cd ncnn
            mkdir build && cd build
            cmake ..
            make -j
            make install

      执行完毕后我们可以看到:
            Install the project...
            -- Install configuration: "release"
            -- Installing: /home/ruyiwei/code/ncnn/build/install/lib/libncnn.a
            -- Installing: /home/ruyiwei/code/ncnn/build/install/include/blob.h
            -- Installing: /home/ruyiwei/code/ncnn/build/install/include/cpu.h
            -- Installing: /home/ruyiwei/code/ncnn/build/install/include/layer.h
            -- Installing: /home/ruyiwei/code/ncnn/build/install/include/mat.h
            -- Installing: /home/ruyiwei/code/ncnn/build/install/include/net.h
            -- Installing: /home/ruyiwei/code/ncnn/build/install/include/opencv.h
            -- Installing: /home/ruyiwei/code/ncnn/build/install/include/platform.h
       查看生成了什么工具：
       我们进入 ncnn/build/tools 目录下，如下所示， 
       我们可以看到已经生成了 ncnn2mem可执行文件，以及 caffe/caffe2ncnn 和 mxnet/mxnet2ncnn 可执行文件
       caffe2ncnn的 作用是将caffe模型生成ncnn 模型 
                  .prototxt >>> .param  .caffemodel >>> .bin；
       mxnet2ncnn 的作用是将 mxnet模型生成ncnn 模型；
       ncnn2mem 是对ncnn模型进行加密。
            drwxrwxr-x 6 wanyouwen wanyouwen   4096  6月 21 00:13 ./
            drwxrwxr-x 6 wanyouwen wanyouwen   4096  6月 21 00:14 ../
            drwxrwxr-x 3 wanyouwen wanyouwen   4096  6月 21 00:13 caffe/
            drwxrwxr-x 3 wanyouwen wanyouwen   4096  6月 21 00:13 CMakeFiles/
            -rw-rw-r-- 1 wanyouwen wanyouwen   1606  6月 21 00:13 cmake_install.cmake
            -rw-rw-r-- 1 wanyouwen wanyouwen   7141  6月 21 00:13 Makefile
            drwxrwxr-x 3 wanyouwen wanyouwen   4096  6月 21 00:13 mxnet/
            -rwxrwxr-x 1 wanyouwen wanyouwen 477538  6月 21 00:13 ncnn2mem*
            drwxrwxr-x 3 wanyouwen wanyouwen   4096  6月 21 00:13 onnx
   
[tensorflow2ncnn](https://github.com/arlose/ncnn-mobilenet-ssd/tree/master/ncnn-mobilenet-ssd/tools/tensorflow)
   
   
     而默认会生成 一个静态库build/src/libncnn.a 和一些可执行文件：
     build/examples/squeezenet 分类模型
     build/examples/fasterrcn  检测模型
     build/examples/ssd/ssdsqueezenet 检测模型
     build/examples/ssd/ssdmobilenet  检测模型
     可以使用squeezenet进行测试，这里这是一个图像分类模型。
     
     把模型和参数复制过来：
     cp ../../examples/squeezen* .
     进行检测：
     ./squeezenet cat.jpg
     >>> 
      283 = 0.377605
      281 = 0.247314
      282 = 0.100278
     这里只是输出了类别的编码，没有输出类别的字符串
     
     需要修改examples/squeezenet.c文件
```c     
// Tencent is pleased to support the open source community by making ncnn available.
// https://opensource.org/licenses/BSD-3-Clause

#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>// putText()
#include "net.h"

#include <sys/time.h>
#include <unistd.h>

// 计时
long getTimeUsec()
{
    
    struct timeval t;
    gettimeofday(&t,0);
    return (long)((long)t.tv_sec*1000*1000 + t.tv_usec);
}

static int detect_squeezenet(const cv::Mat& bgr, std::vector<float>& cls_scores)
{
    ncnn::Net squeezenet;// 前向模型
    squeezenet.load_param("squeezenet_v1.1.param");// 模型框架
    squeezenet.load_model("squeezenet_v1.1.bin");// 权重参数
    
// ncnn 用自己的数据结构 Mat 来存放输入和输出数据 输入图像的数据要转换为 Mat，依需要减去均值和乘系数
    // 图片变形
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 227, 227);
    // 各个通道均值
    const float mean_vals[3] = {104.f, 117.f, 123.f};
    in.substract_mean_normalize(mean_vals, 0);// 图像减去均值归一化

    ncnn::Extractor ex = squeezenet.create_extractor();
    ex.set_light_mode(true);// 模型 提取器 

    ex.input("data", in);
    
// 执行前向网络，获得计算结果
    ncnn::Mat out;
    ex.extract("prob", out);//提取 prob层的输出

    cls_scores.resize(out.w);
    for (int j=0; j<out.w; j++)
    {
        cls_scores[j] = out[j];//结果的每一个值 变成 vector 
    }

    return 0;
}
// 打印结果  有修改 保存类别结果和得分
static int print_topk(const std::vector<float>& cls_scores, int topk, std::vector<int>& index_result, std::vector<float> score_result)
{
    // partial sort topk with index
    int size = cls_scores.size();// 结果维度
    std::vector< std::pair<float, int> > vec;
    vec.resize(size);
    for (unsigned int i=0; i<size; i++)
    {// 成对 值:id 这里id对于类别
        vec[i] = std::make_pair(cls_scores[i], i);
    }
    // 排序
    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater< std::pair<float, int> >());

    // print topk and score
    for (int i=0; i<topk; i++)
    {
        float score = vec[i].first;//得分值
        int index = vec[i].second;// id
        //fprintf(stderr, "%d = %f\n", index, score);
       //  添加保存结果的 vector数组
        score_result.push_back(score);
        index_result.push_back(index);
    }

    return 0;
}

// 添加一个载入 类别目录的文件 的 函数
static int load_labels(std::string path, std::vector<std::string>& labels)
{
    FILE* fp = fopen(path.c_str(), "r");//读取文件
    
    while(!feof(fp))
    {
      char str_b[1024];//先读取 1024个字符
      fgets(str_b, 1024, fp);
      std::string str_block(str_b);//转换成 string 方便操作
      
      if(str_block.length() > 0)
      {
        for (unsigned int i = 0; i <  str_block.length(); i++)
        {
           if(str_block[i] == ' ')
           {
              std:: string name = str_block.substr(i, str_block.length() - i - 1);
              labels.push_back(name);
              i = str_block.length();
           }
        }
      }
    }
   return 0 ;
}


int main(int argc, char** argv)
{
   // 命令行传入的图片文件名
    const char* imagepath = argv[1];
   // opencv 读取图像
    cv::Mat m = cv::imread(imagepath, CV_LOAD_IMAGE_COLOR);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }
     
    // 读取类别标签文件
    std::vector<std::string> labels;
    load_labels("synset_words.txt", labels);
    
    std::vector<float> cls_scores;

    long time = getTimeUsec();
    detect_squeezenet(m, cls_scores);
    time = getTimeUsec() - time;
    printf("detection time: %ld ms\n",time/1000);


    std::vector<int> index;
    std::vector<float> score;

    print_topk(cls_scores, 3, index, score);

   for(unsigned int i = 0; i < index.size(); i++)
   {
     cv::putText(m, labels[index[i]], cv::Point(50, 50+30*i), CV_FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 100, 200), 2, 8);
   }
   
   cv::imshow("result", m);
   cv::imwrite("test_result.jpg", m);
   cv::waitKey(0);

   return 0;
}
```

      把模型和参数复制过来：
      cp ../../examples/squeezen* .
      以及类别列表文件：
      cp ../../examples/synset_words.txt .
      进行检测：
      ./squeezenet cat.jpg

   
### 2. caffe网络模型转换为 ncnn模型 示例
#### caffe下Alexnet网络模型转换为NCNN模型
      我们在测试的过程中需要 .caffemodel文件(模型参数文件)  以及 deploy.prototxt文件(模型框架结构) ,
      所以我们再将caffe模型转换为NCNN模型的时候，
      同样也需要 .caffemodel以及deploy.prototxt这两个文件，为了方便，我们使用AlexNet为例讲解。
      
**a. 下载 caffe 模型和参数**

      alexnet 的 deploy.prototxt 可以在这里下载 https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet 
      alexnet 的 .caffemodel 可以在这里下载 http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel
      
**b. 转换**

      由于NCNN提供的转换工具只支持转换新版的caffe模型,
      所以我们需要利用caffe自带的工具将旧版的caffe模型转换为新版的caffe模型后,
      再将新版本的模型转换为NCNN模型.

      旧版本caffe模型->新版本caffe模型->NCNN模型。
      
**c. 旧版caffe模型转新版caffe模型**

      转换ncnn网络和模型
      caffe 自带了工具可以把老版本的 caffe 网络和模型转换为新版（ncnn的工具只认识新版

      upgrade_net_proto_text [老prototxt] [新prototxt]
      upgrade_net_proto_binary [老caffemodel] [新caffemodel]
      
      模型框架转换：
      ~/code/ncnn/build/tools$ ~/caffe/build/tools/upgrade_net_proto_text deploy.prototxt new_deplpy.prototxt
      模型权重文件转换：
      ~/code/ncnn/build/tools$ ~/caffe/build/tools/upgrade_net_proto_binary bvlc_alexnet.caffemodel new_bvlc_alexnet.caffemodel
      
            上面的命令需要根据自己的caffe位置进行修改

            执行后,就可以生成新的caffe模型.

            因为我们每次检测一张图片,所以要对新生成的deploy.prototxt进行修改:第一个 dim 设为 1 一次输入的图片数量
            layer {
                  name: "data"
                  type: "Input"
                  top: "data"
                  input_param { shape: { dim: 1 dim: 3 dim: 227 dim: 227 } }
            }
            
**d. 新版caffe模型转ncnn模型**
      
      ./caffe/caffe2ncnn new_deplpy.prototxt new_bvlc_alexnet.caffemodel alexnet.param alexnet.bin
      
       caffe2ncnn的 作用是将caffe模型生成ncnn 模型 
            .prototxt >>> .param  .caffemodel >>> .bin；
            
      执行上面命令后就可以生成NCNN模型需要的param 与bin 文件.
      
**e. 对模型参数加密**

      ./ncnn2mem alexnet.param alexnet.bin alexnet.id.h alexnet.mem.h  
      注意 alexnet.id.h alexnet.mem.h 为定义的文件名 头文件名 后面后生成
      最后 
       alexnet.bi >>>  alexnet.param.bin     小
                       alexnet.id.h          小
                       alexnet.mem.h         较大

**f. 模型载入**

      对于加密文件的读取也和原来不同,在源码中,
      
      非加密param读取方式为：
            ncnn::Net net;
            net.load_param("alexnet.param");
            net.load_model("alexnet.bin");

      加密param.bin读取方式为：
      ncnn::Net net;
      net.load_param_bin("alexnet.param.bin");
      net.load_model("alexnet.bin");
      
## mobileNET 分类网络示例
[caffe 模型参数下载](https://github.com/shicai/MobileNet-Caffe)    

## 2. 模型转换 
      先caffe old转换到新版本下
            ./../../tools/caffe_tools/upgrade_net_proto_text mobilenet_deploy.prototxt mobilenet_new.prototxt
            ./../../tools/caffe_tools/upgrade_net_proto_binary mobilenet.caffemodel mobilenet_new.caffemodel
      caffe to  ncnn
      ./../../tools/caffe/caffe2ncnn mobilenet_new.prototxt mobilenet_new.caffemodel mobilenet-ncnn.param mobilenet-ncnn.bin
      

## 4. 修改检测源文件 新建 ncnn/examples/mobilenet.cpp
```c
// Tencent is pleased to support the open source community by making ncnn available.
// https://opensource.org/licenses/BSD-3-Clause

#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>// putText()
#include "net.h"

#include <sys/time.h>
#include <unistd.h>

// 计时
long getTimeUsec()
{
    
    struct timeval t;
    gettimeofday(&t,0);
    return (long)((long)t.tv_sec*1000*1000 + t.tv_usec);
}

static int detect_mobilenet(const cv::Mat& bgr, std::vector<float>& cls_scores)
{
    ncnn::Net mobilenet;// 前向模型
    mobilenet.load_param("mobilenet-ncnn.param");// 模型框架
    mobilenet.load_model("mobilenet-ncnn.bin");// 权重参数
    // 图片变形  网络输入尺寸为 224*224 之前squeezenet为227
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 224, 224);
    // 各个通道均值
    const float mean_vals[3] = {104.f, 117.f, 123.f};
    in.substract_mean_normalize(mean_vals, 0);// 图像减去均值归一化

    ncnn::Extractor ex = mobilenet.create_extractor();
    ex.set_light_mode(true);// 模型 提取器 

    ex.input("data", in);//数据输入层

    ncnn::Mat out;
    ex.extract("prob", out);//提取 prob层的输出

    cls_scores.resize(out.w);
    for (int j=0; j<out.w; j++)
    {
        cls_scores[j] = out[j];//结果的每一个值 变成 vector 
    }

    return 0;
}
// 打印结果  有修改 保存类别结果和得分
static int print_topk(const std::vector<float>& cls_scores, int topk, std::vector<int>& index_result, std::vector<float> score_result)
{
    // partial sort topk with index
    int size = cls_scores.size();// 结果维度
    std::vector< std::pair<float, int> > vec;
    vec.resize(size);
    for (unsigned int i=0; i<size; i++)
    {// 成对 值:id 这里id对于类别
        vec[i] = std::make_pair(cls_scores[i], i);
    }
    // 排序
    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater< std::pair<float, int> >());

    // print topk and score
    for (int i=0; i<topk; i++)
    {
        float score = vec[i].first;//得分值
        int index = vec[i].second;// id
        //fprintf(stderr, "%d = %f\n", index, score);
       //  添加保存结果的 vector数组
        score_result.push_back(score);
        index_result.push_back(index);
    }

    return 0;
}

// 添加一个载入 类别目录的文件 的 函数
static int load_labels(std::string path, std::vector<std::string>& labels)
{
    FILE* fp = fopen(path.c_str(), "r");//读取文件
    
    while(!feof(fp))
    {
      char str_b[1024];//先读取 1024个字符
      fgets(str_b, 1024, fp);
      std::string str_block(str_b);//转换成 string 方便操作
      
      if(str_block.length() > 0)
      {
        for (unsigned int i = 0; i <  str_block.length(); i++)
        {
           if(str_block[i] == ' ')
           {
              std:: string name = str_block.substr(i, str_block.length() - i - 1);
              labels.push_back(name);
              i = str_block.length();
           }
        }
      }
    }
   return 0 ;
}


int main(int argc, char** argv)
{
   // 命令行传入的图片文件名
    const char* imagepath = argv[1];
   // opencv 读取图像
    cv::Mat m = cv::imread(imagepath, CV_LOAD_IMAGE_COLOR);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }
     
    // 读取类别标签文件
    std::vector<std::string> labels;
    load_labels("synset_words.txt", labels);
    
    std::vector<float> cls_scores;
    long time = getTimeUsec();
    detect_mobilenet(m, cls_scores);
    time = getTimeUsec() - time;
    printf("detection time: %ld ms\n",time/1000);

    std::vector<int> index;
    std::vector<float> score;

    print_topk(cls_scores, 3, index, score);

   for(unsigned int i = 0; i < index.size(); i++)
   {
     cv::putText(m, labels[index[i]], cv::Point(50, 50+30*i), CV_FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 100, 200), 2, 8);
   }
   
   cv::imshow("result", m);
   cv::imwrite("test_result.jpg", m);
   cv::waitKey(0);

   return 0;
}

```

**修改ncnn/examples/CMakeLists.txt 添加编译选项**

      add_executable(mobilenet mobilenet.cpp)
      target_link_libraries(mobilenet ncnn ${OpenCV_LIBS})

## 5. 编译&运行
      ./mobilenet cat.jpg

    
    
## mobileNET-SSD 检测网络示例
## 1. 训练

[使用caffe-ssd mobilenet-v1 训练网络 得到网络权重参数](https://github.com/Ewenwan/MVision/tree/master/CNN/MobileNet/MobileNet_v1_ssd_caffe)

      这里训练的效果不太好，是基于mobilenet的权重进行训练的，得到 caffe版本的模型和权重文件：
            MN_ssd_33_deploy.prototxt
            MN_ssd_33_iter_26000.caffemodel
  
 
      这里的是直接训练好的 mobilenet-ssd 检测效果还可以
      其实可以基于这个权重进行训练：
[也可以直接从这里下载](https://github.com/chuanqi305/MobileNet-SSD)         


## 2. 模型转换 
     caffe_old -> caffe_new:
           注意这里的caffe版本为 caffe-ssd的版本 否则有些层的类型不认识
[caffe-ssd](https://github.com/weiliu89/caffe/tree/ssd)     

            upgrade_net_proto_text [老prototxt] [新prototxt]
            upgrade_net_proto_binary [老caffemodel] [新caffemodel]

            模型框架转换：
            ~/code/ncnn/build/tools$ ../../tools/caffe_tools/upgrade_net_proto_text MN_ssd_33_deploy.prototxt ssdmobilenet.prototxt
            模型权重文件转换：
            ~/code/ncnn/build/tools$ ../../tools/caffe_tools/upgrade_net_proto_binary MN_ssd_33_iter_26000.caffemodel ssdmobilenet.caffemodel 

            上面 对casse-ssd/build/tools 工具做了 符号链接
                  ln -s /home/wanyouwen/ewenwan/software/caffe-ssd/build/tools/ ./caffe_tools

     caffe -> ncnn
            ./../../tools/caffe/caffe2ncnn ssdmobilenet.prototxt ssdmobilenet.caffemodel mobilenet_ssd_voc_ncnn.param mobilenet_ssd_voc_ncnn.bin 
            caffe2ncnn的 作用是将caffe模型生成ncnn 模型 
            .prototxt >>> .param  .caffemodel >>> .bin；

            执行上面命令后就可以生成NCNN模型需要的param 与bin 文件.

      
## 3. 修改检测源文件
```c
// Tencent is pleased to support the open source community by making ncnn available.
//

#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

#include "net.h"

#include <sys/time.h>
#include <unistd.h>

// 计时
long getTimeUsec()
{
    
    struct timeval t;
    gettimeofday(&t,0);
    return (long)((long)t.tv_sec*1000*1000 + t.tv_usec);
}

//定义一个结果 结构体
struct Object{
    cv::Rect rec;//边框
    int class_id;//类别id
    float prob;//概率
};
// voc
const char* class_names[] = {"background",
                            "aeroplane", "bicycle", "bird", "boat",
                            "bottle", "bus", "car", "cat", "chair",
                            "cow", "diningtable", "dog", "horse",
                            "motorbike", "person", "pottedplant",
                            "sheep", "sofa", "train", "tvmonitor"};

static int detect_mobilenet(cv::Mat& raw_img, float show_threshold)
{
    ncnn::Net ssdmobilenet;
    /* 模型下载
     * model is  converted from https://github.com/chuanqi305/MobileNet-SSD
     * and can be downloaded from https://drive.google.com/open?id=0ByaKLD9QaPtucWk0Y0dha1VVY0U
     */
 
    printf("loading net... \r\n"); 
    // 原始图片尺寸 
    int img_h = raw_img.size().height;
    int img_w = raw_img.size().width;
    ssdmobilenet.load_param("mobilenet_ssd_voc_ncnn.param");
    ssdmobilenet.load_model("mobilenet_ssd_voc_ncnn.bin");
    int input_size = 300;
    // 改变图像尺寸 到网络的输入尺寸
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(raw_img.data, ncnn::Mat::PIXEL_BGR, raw_img.cols, raw_img.rows, input_size, input_size);
    // 去均值, 再归一化
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1.0/127.5,1.0/127.5,1.0/127.5};
    
    //in.substract_mean_normalize(mean_vals, norm_vals);// 上面给的模型
    
    in.substract_mean_normalize(mean_vals, 0);// 我自己训练的 没有进行归一化

    ncnn::Extractor ex = ssdmobilenet.create_extractor();
    ex.set_light_mode(true);
    //ex.set_num_threads(4);//线程数量
    ex.input("data", in);
    
    printf("begin detecting... \r\n");
     
    long time = getTimeUsec();
     
    ncnn::Mat out;
    ex.extract("detection_out", out);//网络输出
    
    time = getTimeUsec() - time;
    printf("detection time: %ld ms\n",time/1000);

    // 打印总结果
    printf("%d %d %d\n", out.w, out.h, out.c);
    std::vector<Object> objects;
    
    // 获取结果
    for (int iw=0;iw<out.h;iw++)
    {
        Object object;
        const float *values = out.row(iw);//一行是一个结果
        object.class_id = values[0];//类别id
        object.prob = values[1];// 概率
        object.rec.x = values[2] * img_w;//边框中心点
        object.rec.y = values[3] * img_h;
        object.rec.width = values[4] * img_w - object.rec.x;//半尺寸
        object.rec.height = values[5] * img_h - object.rec.y;
        objects.push_back(object);
    }
    // 打印结果
    for(unsigned int i = 0;i<objects.size();++i)
    {
        Object object = objects.at(i);
        if(object.prob > show_threshold)//按阈值显示
        {
            cv::RNG rng(cvGetTickCount()); 
            //cv::rectangle(raw_img, object.rec, cv::Scalar(255, 0, 0));// 绿色 物体标注框
            // 随机颜色框
            cv::rectangle(raw_img, object.rec, cv::Scalar(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255)),1);
            std::ostringstream pro_str;
            pro_str<<object.prob;//概率大小字符串
            // 类别+概率字符串
            std::string label = std::string(class_names[object.class_id]) + ": " + pro_str.str();
            int baseLine = 0;
            // 获取文字大小
            cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            // 画文本矩形边框
            cv::rectangle(raw_img, cv::Rect(cv::Point(object.rec.x, object.rec.y- label_size.height),
                                  cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), CV_FILLED);//白色
            // 添加文字
            cv::putText(raw_img, label, cv::Point(object.rec.x, object.rec.y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));//黑色
        }
    }
    // 显示带结果标签的 图像
    cv::imshow("result",raw_img);
    cv::waitKey();

    return 0;
}

int main(int argc, char** argv)
{   
    if(argc != 2){
       printf("usage: ./ssdmobilenet *.jpg\r\n"); 
       return -1;
    }
    // 图像地址
    const char* imagepath = argv[1];
    // 读取图像
    cv::Mat m = cv::imread(imagepath, CV_LOAD_IMAGE_COLOR);
    
    printf("read img: %s successful\r\n", argv[1]);
 
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }
     
    printf("detecting... \r\n");
    
    // 检测结果
    detect_mobilenet(m,0.5);

    return 0;
}
```

## 4. 编译
make -j

## 5. 运行测试

./ssdmobilenet person.jpg

      这里如果没使用 新版本caffe格式可能会有错误：
            find_blob_index_by_name data_splitncnn_6 failed
            find_blob_index_by_name data_splitncnn_5 failed
            find_blob_index_by_name data_splitncnn_4 failed
            find_blob_index_by_name data_splitncnn_3 failed
            find_blob_index_by_name data_splitncnn_2 failed
            find_blob_index_by_name data_splitncnn_1 failed
            find_blob_index_by_name data_splitncnn_0 failed
            Segmentation fault

# ssdsuqeezenet 
[prototxt & caffemodel ](https://github.com/chuanqi305/SqueezeNet-SSD)


```c
// Tencent is pleased to support the open source community by making ncnn available.
//
#include <stdio.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "net.h"

////////
#include <sys/time.h>
#include <unistd.h>

// 计时
long getTimeUsec()
{
    
    struct timeval t;
    gettimeofday(&t,0);
    return (long)((long)t.tv_sec*1000*1000 + t.tv_usec);
}
///////

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static int detect_squeezenet(const cv::Mat& bgr, std::vector<Object>& objects)
{
    ncnn::Net squeezenet;

    // original pretrained model from https://github.com/chuanqi305/SqueezeNet-SSD
    // squeezenet_ssd_voc_deploy.prototxt
    // https://drive.google.com/open?id=0B3gersZ2cHIxdGpyZlZnbEQ5Snc
    printf("loading net... \r\n"); 

    squeezenet.load_param("squeezenet_ssd_voc.param");
    squeezenet.load_model("squeezenet_ssd_voc.bin");
    const int target_size = 300;//网络输入尺寸
    // 原始图片尺寸 
    int img_w = bgr.cols;
    int img_h = bgr.rows;
    // 改变图像尺寸 到网络的输入尺寸
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, target_size, target_size);
    // 去均值, 再归一化
    const float mean_vals[3] = {104.f, 117.f, 123.f};
    in.substract_mean_normalize(mean_vals, 0);//注意这里未归一化

    ncnn::Extractor ex = squeezenet.create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(4);//线程数量

    ex.input("data", in);
    
    printf("begin detecting... \r\n");
     
    long time = getTimeUsec();

    ncnn::Mat out;
    ex.extract("detection_out",out);
    
    time = getTimeUsec() - time;
    printf("detection time: %ld ms\n",time/1000);

    // 打印总结果
    // printf("%d %d %d\n", out.w, out.h, out.c);
    objects.clear();
    for (int i=0; i<out.h; i++)
    {
        const float* values = out.row(i);//一行是一个结果

        Object object;
        object.label = values[0];//类别id
        object.prob = values[1];// 概率
        object.rect.x = values[2] * img_w;//边框中心点
        object.rect.y = values[3] * img_h;
        object.rect.width = values[4] * img_w - object.rect.x;//半尺寸
        object.rect.height = values[5] * img_h - object.rect.y;

        objects.push_back(object);
    }

    return 0;
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
 // voc
    static const char* class_names[] = {"background",
        "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair",
        "cow", "diningtable", "dog", "horse",
        "motorbike", "person", "pottedplant",
        "sheep", "sofa", "train", "tvmonitor"};

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);
 
        cv::RNG rng(cvGetTickCount()); 
        //cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));// 绿色 物体标注框
        // 随机颜色框
        cv::rectangle(image, obj.rect, cv::Scalar(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255)));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;
        // 画文本矩形边框
        cv::rectangle(image, cv::Rect(cv::Point(x, y),
                                      cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), CV_FILLED);
        // 添加文字
        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    cv::imshow("image", image);
    cv::waitKey(0);
}

int main(int argc, char** argv)
{   
    if(argc != 2){
       printf("usage: ./ssdmobilenet *.jpg\r\n"); 
       return -1;
    }

    // 图像地址
    const char* imagepath = argv[1];
    // 读取图像
    cv::Mat m = cv::imread(imagepath, CV_LOAD_IMAGE_COLOR);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<Object> objects;
    
    detect_squeezenet(m, objects);

    draw_objects(m, objects);

    return 0;
}
```
