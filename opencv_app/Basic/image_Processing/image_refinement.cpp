// Copyright 2018 Zeyu Zhong
// Lincese(MIT)
// Author: Zeyu Zhong
// Date: 2018.5.3

// https://github.com/zhearing/image_measurement/blob/master/3-image-thresholding-and-image-refinement/src/image_refinement.cpp

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

int main(int argc, char **argv) {
    Mat image = imread("../../3-image-thresholding-and-image-refinement/output/image_binary.bmp", -1);
    Mat image_refine;
    image.copyTo(image_refine);

    for (int i = 0; i < image.rows; i++)
        for (int j = 0; j < image.cols; j++) {
            if (image.at<uchar>(i, j) == 255)// 白
                image.at<uchar>(i, j) = 1;// 边黑
        }

    for (int i = 1; i < image_refine.rows - 1; i++)
        for (int j = 1; j < image_refine.cols - 1; j++) {
            int f[9];
            int a = 0;
            int b = 0;
            if (image.at<uchar>(i, j) == 0) {
                continue;// 跳过纯黑
            } 
            else
            {
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
                f[8] = f[0];
            }
            for (int n = 0; n < 8; n++) {
                a = abs(f[i + 1] - f[i]) + 0;
                b = f[i] + b;
            }
            if ((a == 0 || a == 2 || a == 4) && (b != 1)) 
            {
                if (((f[0] && f[2] && f[4]) == 0) && ((f[0] && f[2] && f[6]) == 0)) {
                    if (a != 4)
                        image_refine.at<uchar>(i, j) = 0;
                    else if (((f[0] && f[6]) == 1) && ((f[1] || f[5]) == 1) && ((f[2] || f[3] || f[4] || f[7]) == 0))
                        image_refine.at<uchar>(i, j) = 0;
                    else if (((f[0] && f[2]) == 1) && ((f[3] || f[7]) == 1) && ((f[1] || f[4] || f[5] || f[6]) == 0))
                        image_refine.at<uchar>(i, j) = 0;
                } else if (((f[2] && f[4] && f[6]) == 0) && ((f[4] && f[6] && f[0]) == 0)) {
                    if (a != 4)
                        image_refine.at<uchar>(i, j) = 0;
                    else if (((f[4] && f[2]) == 1) && ((f[5] || f[1]) == 1) && ((f[0] || f[3] || f[6] || f[7]) == 0))
                        image_refine.at<uchar>(i, j) = 0;
                    else if (((f[6] && f[4]) == 1) && ((f[7] || f[3]) == 1) && ((f[0] || f[5] || f[2] || f[1]) == 0))
                        image_refine.at<uchar>(i, j) = 0;
                }
            }
        }
    namedWindow("image_refine", WINDOW_AUTOSIZE);
    imshow("image_refine", image_refine);
    imwrite("../../3-image-thresholding-and-image-refinement/output/image_refine.bmp", image_refine);
    waitKey(0);
    return 0;
}
