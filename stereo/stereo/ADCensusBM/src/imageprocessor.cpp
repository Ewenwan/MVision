/*
图像预处理

*/
#include "imageprocessor.h"
#include <limits>

ImageProcessor::ImageProcessor(float percentageOfDeletion)
{
    this->percentageOfDeletion = percentageOfDeletion;//像素数量阈值比率
}
// 图像明亮度增强
// BGR--->YCrCb
// Y通道强度直方图
// 像素数量阈值比率 ---> 强度阈值 min  max
// 
Mat ImageProcessor::stretchHistogram(Mat image)
{
    Size imgSize = image.size();//图像尺寸
    std::vector<Mat> channels;
    Mat output(imgSize, CV_8UC3);
/*
以下是标准公式
RGB 转换成 YUV
Y      = (0.257 * R) + (0.504 * G) + (0.098 * B) +16
Cr    = V =(0.439 * R) -  (0.368 * G) - (0.071 * B) +128
Cb   = U = -(0.148 * R) - (0.291 * G) + (0.439 * B) + 128

YUV 转换成 RGB
B =1.164(Y - 16) + 2.018(U - 128)
G =1.164(Y - 16) - 0.813(V - 128) - 0.391(U - 128)
R =1.164(Y - 16) + 1.596(V - 128)
*/
    uint pixelThres = percentageOfDeletion * imgSize.height * imgSize.width;
    std::vector<uint> hist;//直方图
    hist.resize(std::numeric_limits<uchar>::max() + 1);

// 统计Y通道直方图　以及最大最小值
    cvtColor(image, output, CV_BGR2YCrCb); //change the color image from BGR to YCrCb format
    split(output, channels); //分割通道　split the image into channels
    uchar min = std::numeric_limits<uchar>::max();
    uchar max = 0;
    for(size_t h = 0; h < imgSize.height; h++)//每一行
    {
        for(size_t w = 0; w < imgSize.width; w++)//每一列
        {
            uchar intensity = channels[0].at<uchar>(h, w);//Y通道　“Y”表示明亮度
            if(intensity < min)//最小值
                min = intensity;
            else if(intensity > max)//最大值
                max = intensity;
            hist[intensity]++;//强度直方图　像素数量统计
        }
    }
/*            *   *
           *          *
         *                 *
     *   |                 |    *  
  *      |                 |        *
*        |                 |           *
------------------------------------
         min               max

*/
    //update minimum
    bool foundMin = false;
    uint pixels = 0;
    for(size_t i = 0; i < hist.size() && !foundMin; i++)
    {
        pixels += hist[i];//直方图像素数量和
        if(pixels <= pixelThres)
            min = i;//记录此时的强度值　从小开始
        else
            foundMin = true;
    }
    //update maximum
    bool foundMax = false;
    pixels = 0;
    for(size_t i = hist.size() - 1; i > 0 && !foundMax; i--)
    {
        pixels += hist[i];//直方图像素数量和
        if(pixels <= pixelThres)
            max = i;//记录此时的强度值　从大开始
        else
            foundMax = true;
    }

    for(size_t h = 0; h < imgSize.height; h++)//每一行
    {
        for(size_t w = 0; w < imgSize.width; w++)//每一列
        {
            uchar intensity = channels[0].at<uchar>(h, w);//Y通道
            uchar newIntensity;
            // Y通道值
            newIntensity = (intensity <= min)
                           ? 0
                           : (intensity >= max)
                             ? std::numeric_limits<uchar>::max()
                             : (std::numeric_limits<uchar>::max() / (float)(max - min)) * (intensity - min);

            channels[0].at<uchar>(h, w) = newIntensity;
        }
    }

    merge(channels,output); //合并通道
    cvtColor(output, output, CV_YCrCb2BGR); //再转换成　BGR格式

    return output;
}
// 非锐化掩蔽
// Unsharp Mask(USM)锐化算法  图像进行平滑，也可以叫做模糊。
Mat ImageProcessor::unsharpMasking(Mat image, std::string blurMethod, int kernelSize, float alpha, float beta)
{
    Mat tempImage, output;
    float gamma = 0.0;
    float gaussianDevX = 0.0;
    float gaussianDevY = 0.0;

    if (blurMethod == "gauss")//　"高斯平滑" 也叫 "高斯模糊" 或 "高斯过滤"
    {// ksize - 核大小
        GaussianBlur(image, tempImage, cv::Size(kernelSize, kernelSize), gaussianDevX, gaussianDevY);
        addWeighted(image, alpha, tempImage, beta, gamma, output);// 图像 叠加 dst=α⋅src1+β⋅src2+ γ
    }
    else if (blurMethod == "median")// 中值滤波
    {
        medianBlur(image, tempImage, kernelSize);
        addWeighted(image, alpha, tempImage, beta, gamma, output);
    }
// 双边滤波  bilateralFilter ( src, dst, i, i*2, i/2 );

    return output;
}
// Laplacian 算子 mask
Mat ImageProcessor::laplacianSharpening(Mat image, int kernelSize, float alpha, float beta)
{
    Mat laplacianRes, abs_dst, output;
    int scale = 0;
    int delta = 0;
    float gamma = 0.0;

    Laplacian(image, laplacianRes, CV_8UC3, kernelSize, scale, delta, BORDER_DEFAULT);
    convertScaleAbs(laplacianRes, abs_dst);
    addWeighted(image, alpha, abs_dst, beta, gamma, output);

    return output;
}
