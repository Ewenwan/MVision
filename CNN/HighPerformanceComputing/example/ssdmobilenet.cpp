// Tencent is pleased to support the open source community by making ncnn available.
//  voc 模型=================
#include <stdio.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

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


// 检测结果 目标对象结构体
struct Object
{
    cv::Rect_<float> rect;// 矩形框
    int label; // 类别标签
    float prob;// 置信度
};

static int detect_mobilenet(const cv::Mat& bgr, std::vector<Object>& objects)
{
    ncnn::Net mobilenet;

    // model is converted from https://github.com/chuanqi305/MobileNet-SSD
    // and can be downloaded from https://drive.google.com/open?id=0ByaKLD9QaPtucWk0Y0dha1VVY0U
    mobilenet.load_param("./model_ncnn/mobilenet_ssd_voc.param");
    mobilenet.load_model("./model_ncnn/mobilenet_ssd_voc.bin");

    const int target_size = 300;

    int img_w = bgr.cols;// 原图像大小
    int img_h = bgr.rows;
    // 转换成 300*300大小
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, target_size, target_size);

    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};//
    const float norm_vals[3] = {1.0/127.5,1.0/127.5,1.0/127.5};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = mobilenet.create_extractor();
//     ex.set_num_threads(4);// 线程数

    ex.input("data", in);// 输入数据

    ncnn::Mat out;
    ex.extract("detection_out",out);

//     printf("%d %d %d\n", out.w, out.h, out.c);
    objects.clear();
    for (int i=0; i<out.h; i++)// 目标数量
    {
        const float* values = out.row(i);

        Object object;
        object.label = values[0];// 类别标签
        object.prob = values[1]; // 置信度
        object.rect.x = values[2] * img_w;// 边框中心点
        object.rect.y = values[3] * img_h;
        object.rect.width = values[4] * img_w - object.rect.x;
        object.rect.height = values[5] * img_h - object.rect.y;

        objects.push_back(object);
    }

    return 0;
}

// 显示框
static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
// voc标签================
    static const char* class_names[] = {"background",
        "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair",
        "cow", "diningtable", "dog", "horse",
        "motorbike", "person", "pottedplant",
        "sheep", "sofa", "train", "tvmonitor"};

    cv::Mat image = bgr.clone();// 原图像

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%s = %.5f at %.2f %.2f %.2f x %.2f\n", class_names[obj.label], obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));// 显示矩形框

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

        cv::rectangle(image, cv::Rect(cv::Point(x, y),
                                      cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), CV_FILLED);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    cv::imshow("image", image);
    cv::waitKey(0);
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];

    cv::Mat pic = cv::imread(imagepath, CV_LOAD_IMAGE_COLOR);
    if (pic.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<Object> objects;

    long time = getTimeUsec();
    detect_mobilenet(pic, objects);
    time = getTimeUsec() - time;
    printf("detection time: %ld ms\n",time/1000);

    draw_objects(pic, objects);

    return 0;
}
