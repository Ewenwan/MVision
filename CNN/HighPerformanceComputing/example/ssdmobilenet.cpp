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
            cv::rectangle(raw_img, object.rec, cv::Scalar(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255)));
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
