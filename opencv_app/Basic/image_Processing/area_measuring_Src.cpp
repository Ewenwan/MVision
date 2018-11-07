// Copyright 2018 Zeyu Zhong
// Lincese(MIT)
// Author: Zeyu Zhong
// Date: 2018.5.4

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
// #include<opencv2/imgproc/imgproc.hpp>
// https://github.com/zhearing/image_measurement/blob/master/4-area-measuring/src/area_measuring.cpp

using namespace cv;

int main(int argc, char **argv) {
    Mat image = imread("../../4-area-measuring/images/origin.bmp", -1);
    //    cvtColor(image,image,CV_BGR2GRAY);
    //    threshold(image,image,145,255,THRESH_BINARY);
    //    imwrite("../../4-area-measuring/output/origin.bmp", image);
    Mat L;
    image.copyTo(L);

    for (int i = 0; i < L.rows - 1; i++)
        for (int j = 0; j < L.cols - 1; j++) {
            L.at<uchar>(i, j) = 0;//清零。。。。？
        }
    int nl = 0;
    int T[90000] = {0};
    for (int i = 0; i < image.rows; i++)
        for (int j = 0; j < image.cols; j++) {
            if (static_cast<int>image.at<uchar>(i, j) == 0) {
                continue;
            } else {
                int X[4];
// [0][3]
// [1][]
// [2]
                X[0] = L.at<uchar>(i - 1, j - 1);//左上点
                X[1] = L.at<uchar>(i - 1, j);
                X[2] = L.at<uchar>(i - 1, j + 1);
                X[3] = L.at<uchar>(i, j - 1);
                int t = 0;
                int L1[4];
                int L2[8];
                for (int k = 0; k < 4; k++) {
                    if (T[X[k]] != 0) {
                        L1[t] = T[X[k]];
                        t++;
                    }
                }
                int n = 0;
                if (t == 0) {
                    n = 0;
                } else {
                    int tem;
                    for (int p = 0; p < t; p++) {
                        for (int q = 0; q < t - p - 1; q++) {
                            if (L1[q] > L1[q + 1]) {
                                tem = L1[q];
                                L1[q] = L1[q + 1];
                                L1[q + 1] = tem;
                            }
                        }
                    }
                    int d = L1[0];
                    for (int w = 1; w < t; w++) {
                        if (L1[w] != d) {
                            L2[n] = d;
                            n++;
                            d = L1[w];
                        }
                    }
                    if (L1[t - 1] == d)
                        L2[n] = d;
                    n = n + 1;
                }
                switch (n) {
                case 0:
                    nl = nl + 1;
                    T[nl] = nl;
                    L.at<uchar>(i, j) = nl;
                    continue;
                case 1:
                    L.at<uchar>(i, j) = L2[0];
                    continue;
                case 2:
                    L.at<uchar>(i, j) = L2[0];
                    for (int k = 2; k < nl + 1; k++) {
                        if (T[k] == L2[1])
                            T[k] = L2[0];
                    }
                    continue;
                }
            }
        }
    int T1[100];
    int T2[100];
    for (int k1 = 1; k1 < nl + 1; k1++)
        T1[k1] = T[k1];
    int tem;
    for (int p = 1; p < nl + 1; p++) {
        for (int q = 1; q < nl - p + 1; q++) {
            if (T1[q] > T1[q + 1]) {
                tem = T1[q];
                T1[q] = T1[q + 1];
                T1[q + 1] = tem;
            }
        }
    }
    int d = T1[1];
    int n0 = 1;
    for (int w = 2; w < nl + 1; w++) {
        if (T1[w] != d) {
            T2[n0] = d;
            n0++;
            d = T1[w];
        }
    }
    if (T1[nl] == d)
        T2[n0] = d;
    for (int i = 1; i < n0 + 1; i++) {
        for (int k1 = 1; k1 < nl + 1; k1++) {
            if (T[k1] == T2[i])
                T[k1] = i;
        }
    }
    for (int i = 0; i < image.rows; i++)
        for (int j = 0; j < image.cols; j++) {
            if (L.at<uchar>(i, j) > 0)
                L.at<uchar>(i, j) = T[L.at<uchar>(i, j)];
        }
    int area[100] = {0};
    for (int m = 0; m < n0 + 1; m++) {
        for (int i = 0; i < image.rows; i++)
            for (int j = 0; j < image.cols; j++) {
                if (L.at<uchar>(i, j) == m)
                    area[m] = area[m] + 1;
            }
    }

    for (int k1 = 0; k1 < n0 + 1; k1++)
        std::cout << "area" << k1 << " " << area[k1] << "\n\r";

    waitKey(0);
    return 0;
}
