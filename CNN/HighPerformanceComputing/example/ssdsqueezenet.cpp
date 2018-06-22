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
