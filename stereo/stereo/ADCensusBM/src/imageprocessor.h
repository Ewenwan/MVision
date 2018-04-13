/*
图像预处理类

*/
#ifndef IMAGEPROCESSOR_H
#define IMAGEPROCESSOR_H
#include <opencv2/opencv.hpp>
#include "common.h"

using namespace cv;

class ImageProcessor//图像预处理类
{
public:
    ImageProcessor(float percentageOfDeletion);
// 图像明亮度增强
// BGR--->YCrCb
// Y通道强度直方图
// 像素数量阈值比率 ---> 强度阈值 min  max
// 
    Mat stretchHistogram(Mat image);
// Unsharp Mask(USM)锐化算法  图像进行平滑，也可以叫做模糊。
    Mat unsharpMasking(Mat image, std::string blurMethod, int kernelSize, float alpha, float beta);
// Laplacian 算子 mask
    Mat laplacianSharpening(Mat image, int kernelSize, float alpha, float beta);
private:
    float percentageOfDeletion;
};

#endif // IMAGEPROCESSOR_H
