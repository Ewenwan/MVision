// Tencent is pleased to support the open source community by making ncnn available.
// mobilenetv2-ssdlite-voc- camera on line

// ncnn 检测出结果后， 使用 grabCut() 对roi进行图像分割，绘制前景

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
#include "opencv2/opencv.hpp"
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


// 绘制mask=======
void draw_mask(cv::Mat& bgr_img, const cv::Rect_<float>& rect, cv::Mat mask)// mask : 
{
 unsigned char* color = (unsigned char*)bgr_img.data; //3 通道
 unsigned char* maskD = (unsigned char*)mask.data;//1 通道
 int beg = (int)rect.x + ((int)rect.y-1)*bgr_img.cols - 1;// 2d框起点
 int ind = 0;
 for(int k=0; k< (int)rect.height; k++) // 每一行
 {   
   int start = beg + k*bgr_img.cols; // 起点
   int end   = start + (int)rect.width;// 终点
   for(int j=start; j<end; j++)//每一列
   {
       if(maskD[ind])// 前景
       {
        color[j*3+0] = 255;// blue 蓝色
        color[j*3+1] = 0;  // green
        color[j*3+2] = 0;  // red
       } 
       ind++; // 
   }
 }
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

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));// 画矩形框矩阵

        cv::Mat tmpROI;// 原图像处 对应 的目标框处的 子图像，对子图像进行gc 图像分割 获取前景后，填充颜色
        bgr_img(obj.rect).copyTo(tmpROI);

        cv::Rect rect(0,0, tmpROI.rows, tmpROI.cols);//左上坐标（X,Y）和 高 宽  宽度不能一样!!!!!
        cv::Mat mask, bgdm, fgdm;
        grabCut(tmpROI, mask, rect, bgdm, fgdm, 1, cv::GC_INIT_WITH_RECT);// 图分割2次
        mask = mask & 1;

        draw_mask(image, obj.rect, mask); // 画上 mask

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
    imwrite("result.jpg", image);
     //cv::imshow("image", image);
     // cv::waitKey(0);
}
