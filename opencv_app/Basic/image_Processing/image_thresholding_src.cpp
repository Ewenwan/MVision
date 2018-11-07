// Copyright 2018 Zeyu Zhong
// Lincese(MIT)
// Author: Zeyu Zhong
// Date: 2018.5.3
// https://github.com/zhearing/image_measurement/blob/master/3-image-thresholding-and-image-refinement/src/image_thresholding.cpp


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

int main(int argc, char **argv) {
    Mat image = imread("../../3-image-thresholding-and-image-refinement/images/smoothed.bmp", -1);
    Mat image_binary;
    image.copyTo(image_binary);

    int s1[image.rows][image.cols];
    for (int m = 1; m < image.rows - 1; m++) 
    {
        for (int n = 1; n < image.cols - 1; n++) 
        {
            int D = 0;
            for (int k1 = -1; k1 < 2; k1++) 
            { // -1 0 1
                for (int k2 = -1; k2 < 2; k2++) 
                { // -1 0 1   
                  // 3*3周围9个点===============
                    if (image.at<uchar>(m, n) > image.at<uchar>(m + k1, n + k2))// 比外圈大
                        D = (image.at<uchar>(m, n) - image.at<uchar>(m + k1, n + k2) + D);// 大的值求和
                    else
                        D = 0 + D;// 小的不加
                }
            }
            s1[m][n] = D;
        }
    }

    int C1 = 0;
    int threshold1 = 0;
    for (int i = 0; i < 256; i++) 
    {// 0~255像素值
        int S = 0;
        for (int m = 1; m < image.rows - 1; m++) 
        {
            for (int n = 1; n < image.cols - 1; n++) 
            {
                if (image.at<uchar>(m, n) == i)// 为1像素的点 为2像素的点 ...
                    S = S + s1[m][n];
            }
        }
        
        if (S > C1) {
            C1 = S;// 最大
            threshold1 = i;// 对应像素阈值
        }
    }
    for (int m = 1; m < image.rows - 1; m++)
        for (int n = 1; n < image.cols - 1; n++) 
        {
            int D = 0;
            for (int k1 = -1; k1 < 2; k1++) // -1 0 1
                for (int k2 = -1; k2 < 2; k2++)  // -1 0 1
                {// 3*3周围9个点===============
                    if (image.at<uchar>(m, n) < image.at<uchar>(m + k1, n + k2))
                        D = (image.at<uchar>(m, n) - image.at<uchar>(m + k1, n + k2)) + D;
                }
            s1[m][n] = D;
        }
    int C2;
    int threshold2 = 0;
    for (int i = 0; i < 256; i++) {
        int S = 0;
        for (int m = 1; m < image.rows - 1; m++)
            for (int n = 1; n < image.cols - 1; n++) 
            {
                if (image.at<uchar>(m, n) == i)// 为1像素的点 为2像素的点 ...
                    S = S + s1[m][n];
            }
        if (S != 0 || S > C2) {
            C2 = S;
            threshold2 = i;
        }
    }

    int threshold_final = (threshold2 - threshold1) / 2;// 阈值

    for (int m = 0; m < image_binary.rows; m++)
        for (int n = 0; n < image_binary.cols; n++) {
        // 阈值二值化==============================================
            if (image_binary.at<uchar>(m, n) >= threshold_final)
                image_binary.at<uchar>(m, n) = 255;
            else
                image_binary.at<uchar>(m, n) = 0;
        }

    namedWindow("image_binary", WINDOW_AUTOSIZE);
    imshow("image_binary", image_binary);
    imwrite("../../3-image-thresholding-and-image-refinement/output/image_binary.bmp", image_binary);
    waitKey(0);
    return 0;
}
