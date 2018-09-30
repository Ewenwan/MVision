// Tencent is pleased to support the open source community by making ncnn available.
// mobilenetv2-ssdlite-voc- camera on line

// 在线运行 =====
// 1. 设置相机参数，打开相机 =================
// 2. 初始化模型            =================
// 3. 格式化输入图像        =================
// 4. 运行检测模型          =================
// 5. 显示检测结果          =================

#include <stdio.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "net.h"

#include <sys/time.h>
#include <unistd.h>

static const char* class_names[] = {"background",
"aeroplane", "bicycle", "bird", "boat",
"bottle", "bus", "car", "cat", "chair",
"cow", "diningtable", "dog", "horse",
"motorbike", "person", "pottedplant",
"sheep", "sofa", "train", "tvmonitor"};


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

// 初始化模型 ======================
void init_detect_mobilenetv2(ncnn::Net& det_net);

// 格式化网络输入 ================================
void process_input(const cv::Mat& bgr_img, ncnn::Mat& net_in, int& src_img_w, int& src_img_h);

// 网络运行获取结果 ==============================
void net_run(ncnn::Net& det_net, ncnn::Mat& net_in, std::vector<Object>& objects, int img_w, int img_h);

// 在原图上显示结果===============================
static void draw_objects(const cv::Mat& bgr_img, const std::vector<Object>& objects);


// main 函数==============================
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
  
    cv::Mat src_img;

// 2. 初始化模型            =================
    ncnn::Net det_net_mobile;
    init_detect_mobilenetv2(det_net_mobile); 


    ncnn::Mat net_in;            // 模型输入
    std::vector<Object> objects; // 检测结果 
    int img_w, img_h;            // 原图像尺寸
    while(CapAll.read(src_img)) 
    { 
// 3. 格式化输入图像        =================
	    process_input(src_img, net_in, img_w, img_h);// 格式化输入===
// 4. 运行检测模型          =================
	    long time = getTimeUsec();// 开始计时
            net_run(det_net_mobile, net_in, objects, img_w, img_h);  // 模型运行
	    time = getTimeUsec() - time;// 结束计时
	    printf("detection time: %ld ms\n",time/1000); // 显示检测时间
// 5. 显示检测结果          =================
	    draw_objects(src_img, objects); // 显示检测结果

            char c = cv::waitKey(1);  
	        //sprintf(filename_l, "%s%02d%s","left", i,".jpg");  
		//imwrite(filename_l, src_imgl);  
	    if( c == 27 || c == 'q' || c == 'Q' )//按ESC/q/Q退出  
		break;  
    }

    return 0;
}


// 初始化模型 ======================
void init_detect_mobilenetv2(ncnn::Net& det_net)
{
    det_net.register_custom_layer("Silence", Noop_layer_creator);// 避免在log中打印并没有使用的blobs的信息。
    // original pretrained model from https://github.com/chuanqi305/MobileNetv2-SSDLite
    // https://github.com/chuanqi305/MobileNetv2-SSDLite/blob/master/ssdlite/voc/deploy.prototxt
    det_net.load_param("./model_ncnn/mobilenetv2_ssdlite_voc.param");
    det_net.load_model("./model_ncnn/mobilenetv2_ssdlite_voc.bin");

}

// 格式化网络输入 ================================
void process_input(const cv::Mat& bgr_img, ncnn::Mat& net_in, int& src_img_w, int& src_img_h)
{

    const int target_size = 300;

    src_img_w = bgr_img.cols;
    src_img_h = bgr_img.rows;

    net_in = ncnn::Mat::from_pixels_resize(bgr_img.data, ncnn::Mat::PIXEL_BGR, bgr_img.cols, bgr_img.rows, target_size, target_size);

    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1.0/127.5,1.0/127.5,1.0/127.5};
    net_in.substract_mean_normalize(mean_vals, norm_vals);// 去均值 归一化
}

// 网络运行获取结果 ================================
void net_run(ncnn::Net& det_net, ncnn::Mat& net_in, std::vector<Object>& objects, int img_w, int img_h)
{
    ncnn::Extractor net_extractor = det_net.create_extractor();// 从 网络模型差创建模型提取器

    net_extractor.set_light_mode(true);        
    // 开启轻模式省内存，可以在每一层运算后自动回收中间结果的内存

    net_extractor.set_num_threads(4);// omp 线程数

    // printf("run ... ");
    net_extractor.input("data", net_in);
    ncnn::Mat out;
    net_extractor.extract("detection_out",out);
    //printf("%d %d %d\n", out.w, out.h, out.c);
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
}

// 在原图上显示结果=============================================================
static void draw_objects(const cv::Mat& bgr_img, const std::vector<Object>& objects)
{
    cv::Mat image = bgr_img.clone(); // 深拷贝===

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%s = %.5f at %.2f %.2f %.2f x %.2f\n", class_names[obj.label], obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));// 矩阵

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);// 显示的字符

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);// 字符大小

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;
        
        // 图上显示矩阵框 ====
        cv::rectangle(image, cv::Rect(cv::Point(x, y),
                                      cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), CV_FILLED);
        // 图像显示文字 =====
        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
    // imwrite("result.jpg", image);
     cv::imshow("image", image);
     // cv::waitKey(0);
}
