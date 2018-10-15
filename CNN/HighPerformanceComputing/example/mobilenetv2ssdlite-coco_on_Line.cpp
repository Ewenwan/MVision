// Tencent is pleased to support the open source community by making ncnn available.
// mobilenetv2-ssdlite-coco 91类

// 在线运行 =====
// 1. 设置相机参数，打开相机 =================
// 2. 创建检测对象
// 3. 运行检测模型          =================
// 4. 显示检测结果          =================


#include <stdio.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "./ncnn/include/net.h"


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

static int detect_mobilenetv2(const cv::Mat& bgr, 
                              std::vector<Object>& objects, 
                              ncnn::Net& mobilenetv2)
{
    // ncnn::Net mobilenetv2;

    //mobilenetv2.register_custom_layer("Silence", Noop_layer_creator);

    // original pretrained model from https://github.com/chuanqi305/MobileNetv2-SSDLite
    // https://github.com/chuanqi305/MobileNetv2-SSDLite/blob/master/ssdlite/voc/deploy.prototxt
    //mobilenetv2.load_param("./model_ncnn/mobilenetv2_ssdlite_coco.param");
    //mobilenetv2.load_model("./model_ncnn/mobilenetv2_ssdlite_coco.bin");

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
// coco类别
    static const char* class_names[] = {"background",
        "person", "bicycle", "car", "motorcycle",
        "airplane", "bus", "train", "truck", 
        "boat", "traffic light", "fire hydrant", "N/A", 
        "stop sign", "parking meter", "bench", "bird",
        "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", 
        "giraffe", "N/A", "backpack", "umbrella", 
        "N/A", "N/A", "handbag", "tie", 
        "suitcase", "frisbee", "skis", "snowboard", 
        "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle",
        "N/A", "wine glass", "cup", "fork", 
        "knife", "spoon", "bowl", "banana",
        "apple", "sandwich", "orange", "broccoli", 
        "carrot", "hot dog", "pizza", "donut",
        "cake", "chair", "couch", "potted plant", 
        "bed", "N/A", "dining table", "N/A",
        "N/A", "toilet", "N/A", "tv", 
        "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster",
        "sink", "refrigerator", "N/A", "book", 
        "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"};

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size()/2; i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%d %s = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, class_names[obj.label], obj.prob,
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
// 1. 设置相机参数，打开相机 =================
    int cameraId = 0; // 默认相机0
    if (argc >= 2)
    {
        fprintf(stderr, "Usage: %s camera_Id\n", argv[0]);
        cameraId = atoi(argv[1]);
    }

    cv::VideoCapture CapAll(cameraId);  
    if( !CapAll.isOpened() )//在线 检测
    {
        printf("打开摄像头失败\r\n");
        return -1;
    }
    // 设置分辨率 
    CapAll.set(CV_CAP_PROP_FRAME_WIDTH,640);  
    CapAll.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

// 2. 定义模型============================
    ncnn::Net mobilenetv2;
    mobilenetv2.register_custom_layer("Silence", Noop_layer_creator);

    // original pretrained model from https://github.com/chuanqi305/MobileNetv2-SSDLite
    // https://github.com/chuanqi305/MobileNetv2-SSDLite/blob/master/ssdlite/voc/deploy.prototxt
    mobilenetv2.load_param("./model_ncnn/mobilenetv2_ssdlite_coco.param");
    mobilenetv2.load_model("./model_ncnn/mobilenetv2_ssdlite_coco.bin");

    std::vector<Object> objects;

    cv::Mat src_img;
    while(CapAll.read(src_img)) 
    { 
        long time = getTimeUsec();

        detect_mobilenetv2(src_img, objects, mobilenetv2);

        time = getTimeUsec() - time;
        printf("detection time: %ld ms\n",time/1000);
        draw_objects(src_img, objects);
        //char c = cv::waitKey();  
        //sprintf(filename_l, "%s%02d%s","left", i,".jpg");  
	//imwrite(filename_l, src_imgl);  
        //if( c == 27 || c == 'q' || c == 'Q' )//按ESC/q/Q退出  
	//    break; 
    }

    return 0;
}
