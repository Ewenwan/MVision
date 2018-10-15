```c
#include "opencv2/opencv.hpp"
using namespace cv;

int main()
{
    Mat src = imread("./dog2.jpeg");// 682 * 1023  512*512
    Rect rect(0,0, src.rows, src.cols-5);//左上坐标（X,Y）和 高 宽  宽度不能一样!!!!!
    Mat result, bgdm, fgdm;

    grabCut(src, result, rect, bgdm, fgdm, 3, GC_INIT_WITH_RECT);// bgdm背景模型，fgdm前景模型，内部使用
                                                                 // 1 为迭代次数
                                                                 // GC_INIT_WITH_RECT 表示 使用矩形框初始化 MASK
    //imshow("grab", result);
    /*threshold(result, result, 2, 255, CV_THRESH_BINARY);
    imshow("threshold", result);*/
    //std::cout << "hight: "<< result.rows << "width:" <<result.cols <<std::endl;
    //compare(result, GC_PR_FGD, result, CMP_EQ);//result和GC_PR_FGD对应像素相等时，目标图像该像素值置为255 白色
    //compare(result, GC_FGD, result, CMP_EQ);//result和GC_FGD对应像素相等时，目标图像该像素值置为255 白色
                                               // result 为生成的 mask，
                                               // GC_BGD = 0 ,   // 确定属于背景的
                                               // GC_FGD = 1;    // 确定属于前景的
                                               // GC_PR_BGD = 2; // 可能属于背景的 
                                               // GC_PR_FGD = 3; // 可能属于前景的
    //imshow("result",result);
    result = result & 1;     //  1 和 3 变为1  0和2变为0
    //imshow("result",result);
    Mat foreground(src.size(), CV_8UC3, Scalar(255, 255, 255));
    src.copyTo(foreground, result);//copyTo有两种形式，此形式表示result为mask
    imshow("foreground", foreground);
    waitKey(0);
    return 0;
}

```
