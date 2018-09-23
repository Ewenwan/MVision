// Tencent is pleased to support the open source community by making ncnn available.
// mobilenetv2-ssdlite-voc

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


class Noop : public ncnn::Layer {};
DEFINE_LAYER_CREATOR(Noop)

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static int detect_mobilenetv2(const cv::Mat& bgr, std::vector<Object>& objects)
{
    ncnn::Net mobilenetv2;

    mobilenetv2.register_custom_layer("Silence", Noop_layer_creator);

    // original pretrained model from https://github.com/chuanqi305/MobileNetv2-SSDLite
    // https://github.com/chuanqi305/MobileNetv2-SSDLite/blob/master/ssdlite/voc/deploy.prototxt
    mobilenetv2.load_param("./model_ncnn/mobilenetv2_ssdlite_voc.param");
    mobilenetv2.load_model("./model_ncnn/mobilenetv2_ssdlite_voc.bin");

    const int target_size = 300;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, target_size, target_size);

    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1.0/127.5,1.0/127.5,1.0/127.5};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = mobilenetv2.create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(4);

    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("detection_out",out);

//     printf("%d %d %d\n", out.w, out.h, out.c);
    objects.clear();
    for (int i=0; i<out.h; i++)
    {
        const float* values = out.row(i);

        Object object;
        object.label = values[0];
        object.prob = values[1];
        object.rect.x = values[2] * img_w;
        object.rect.y = values[3] * img_h;
        object.rect.width = values[4] * img_w - object.rect.x;
        object.rect.height = values[5] * img_h - object.rect.y;

        objects.push_back(object);
    }

    return 0;
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
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

        fprintf(stderr, "%s = %.5f at %.2f %.2f %.2f x %.2f\n", class_names[obj.label], obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

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
    imwrite("result.jpg", image);
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

    cv::Mat m = cv::imread(imagepath, CV_LOAD_IMAGE_COLOR);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<Object> objects;

    long time = getTimeUsec();

    detect_mobilenetv2(m, objects);

    time = getTimeUsec() - time;
    printf("detection time: %ld ms\n",time/1000);


    draw_objects(m, objects);

    return 0;
}
