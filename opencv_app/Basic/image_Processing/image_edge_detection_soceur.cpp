// Copyright 2018 Zeyu Zhong
// Lincese(MIT)
// Author: Zeyu Zhong
// Date: 2018.5.3
// https://github.com/zhearing/image_measurement/blob/master/2-image-edge-detection/src/image_edge_detection.cpp

// 源码 边缘检测
// 
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

int main(int argc, char **argv) {
    Mat image = imread("../../2-image-edge-detection/images/mid_image_enhancement.bmp", -1);
    Mat priwitt, smoothed;
    image.copyTo(priwitt);
    image.copyTo(smoothed);

    for (int i = 1; i < image.rows - 1; i++)
        for (int j = 1; j < image.cols - 1; j++) {
            int f[8];
            
//     [3][4][5]
//     [2][ ][6]
//     [1][0][7]
            f[0] = image.at<uchar>(i, j + 1);
            f[1] = image.at<uchar>(i - 1, j + 1);
            f[2] = image.at<uchar>(i - 1, j);
            f[3] = image.at<uchar>(i - 1, j - 1);
            f[4] = image.at<uchar>(i, j - 1);
            f[5] = image.at<uchar>(i + 1, j - 1);
            f[6] = image.at<uchar>(i + 1, j);
            f[7] = image.at<uchar>(i + 1, j + 1);
            int w1, w2, w;
            w1 = f[3] + f[4] + f[2] - f[0] - f[6] - f[7]; // 左上角 - 右下角
            w2 = f[2] + f[1] + f[0] - f[4] - f[5] - f[6]; // 左下角 - 右上角
            w = abs(w1) + abs(w2);
            if (w > 80)
                priwitt.at<uchar>(i, j) = 255;// 角点
            else
                priwitt.at<uchar>(i, j) = 0;
            int Dx, Dy;
            Dx = f[1] + f[0] + f[7] - f[3] - f[4] - f[5];// 下 - 上
            Dy = f[3] + f[2] + f[1] - f[5] - f[6] - f[7];// 左 - 右
            smoothed.at<uchar>(i, j) = abs(Dx) + abs(Dy);
        }

    namedWindow("priwitt", WINDOW_AUTOSIZE);
    imshow("priwitt", priwitt);
    imwrite("../../2-image-edge-detection/output/priwitt.bmp", priwitt);
    namedWindow("smoothed", WINDOW_AUTOSIZE);
    imshow("smoothed", smoothed);
    imwrite("../../2-image-edge-detection/output/smoothed.bmp", smoothed);
    waitKey(0);
    return 0;
}
