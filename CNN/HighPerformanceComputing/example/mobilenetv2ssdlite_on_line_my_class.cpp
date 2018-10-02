// Tencent is pleased to support the open source community by making ncnn available.
// mobilenetv2-ssdlite-voc- camera on line

// 在线运行 =====
// 1. 设置相机参数，打开相机 =================
// 2. 创建检测对象
// 3. 运行检测模型          =================
// 4. 显示检测结果          =================

#include <stdio.h>
//#include <vector>
//#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// #include "net.h"

#include <sys/time.h>
#include <unistd.h>

#include "detector.h" // 检测类
using object_analytics_nodelet::detector::Detector;
// 计时
long getTimeUsec()
{
    
    struct timeval t;
    gettimeofday(&t,0);
    return (long)((long)t.tv_sec*1000*1000 + t.tv_usec);
}

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
    Detector* det_ptr = new(Detector);

    std::vector<Object> objects; // 检测结果 
    while(CapAll.read(src_img)) 
    { 
// 3. 运行检测模型          =================
	    long time = getTimeUsec();// 开始计时
            det_ptr->Run(src_img, objects);  // 模型运行
	    time = getTimeUsec() - time;// 结束计时
	    printf("detection time: %ld ms\n",time/1000); // 显示检测时间
// 4. 显示检测结果          =================
	    draw_objects(src_img, objects); // 显示检测结果

            char c = cv::waitKey(1);  
	        //sprintf(filename_l, "%s%02d%s","left", i,".jpg");  
		//imwrite(filename_l, src_imgl);  
	    if( c == 27 || c == 'q' || c == 'Q' )//按ESC/q/Q退出  
		break;  
    }

    return 0;
}


// 在原图上显示结果=============================================================
static void draw_objects(const cv::Mat& bgr_img, const std::vector<Object>& objects)
{
    cv::Mat image = bgr_img.clone(); // 深拷贝===

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%s = %.5f at %.2f %.2f %.2f x %.2f\n", obj.object_name.c_str(), obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));// 矩阵

        char text[256];
        sprintf(text, "%s %.1f%%", obj.object_name.c_str(), obj.prob * 100);// 显示的字符

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
