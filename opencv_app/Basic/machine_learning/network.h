#ifndef __NETWORK_H
#define __NETWORK_H

#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace ml;

int networkInit(void);//初始化一个用于训练的神经网络
int networkInit(const char *fileName);//初始化一个用于分类的神经网络
int detectFaceFormImg(const char *imgFilename,Mat *faceRoi);//从指定图像中检测人脸 级联分类器
//训练神经网络并保存到指定的文件
int networkTrain(std::vector<Mat> *posFaceVector,std::vector<Mat> *negFaceVector,const char *fileName);
//用训练好的神经网络进行分类
int networkClassify(Mat *image, Rect& face_dec );

#endif /* __NETWORK_H */
