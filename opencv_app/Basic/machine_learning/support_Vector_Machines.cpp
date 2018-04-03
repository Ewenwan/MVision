/*
支持向量机(SVM) 
Support Vector Machines 

支持向量机 (SVM) 是一个类分类器，正式的定义是一个能够将不同类样本
在样本空间分隔的超平面(n-1 demension)。
 换句话说，给定一些标记(label)好的训练样本 (监督式学习), 
SVM算法输出一个最优化的分隔超平面。

假设给定一些分属于两类的2维点，这些点可以通过直线分割， 我们要找到一条最优的分割线.

|
|
|              +
|            +    +
|               +
|  *               +
|                        +
|   *   *           +
|   *                 + 
|  *      *             +
|   *    *
——————————————————————————————————>

w * 超平面 = 0  

w * x+ + b >= 1
w * x- + b <= -1
  
w * x+0 + b =  1          y = +1
w * x-0 + b = -1          y = -1


(x+0 - x-0) * w/||w|| =
 
x+0 * w/||w||  - x-0 * w/||w|| = 

(1-b) /||w|| - (-1-b)/||w||  = 

2 * /||w||   maxmize
=======>

min 1/2 * ||w||^2

这是一个拉格朗日优化问题

L = 1/2 * ||w||^2 - SUM(a * yi *( w* xi +b) - 1)

dL/dw  = ||w||  - sum(a * yi * xi)   subjext  to 0    calculate the  w
  
dL/db  =  SUM(a * yi)                subjext  to 0     

====>
||w||  = sum(a * yi * xi)
SUM(a * yi)  = 0
=====>

L =  -1/2  SUM(SUM(ai*aj * yi*yj * xi*xi))


*/


#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
using namespace cv;
using namespace cv::ml;
int main(int, char**)
{
    // Data for visual representation
    int width = 512, height = 512;
    Mat image = Mat::zeros(height, width, CV_8UC3);

    // 建立训练样本 training data
    int labels[4] = {1, -1, -1, -1};// 由分属于两个类别的2维点组成， 其中一类包含一个样本点，另一类包含三个点。
    float trainingData[4][2] = { {501, 10}, {255, 10}, {501, 255}, {10, 501} };
    Mat trainingDataMat(4, 2, CV_32FC1, trainingData);
    Mat labelsMat(4, 1, CV_32SC1, labels);

    // Train the SVM
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);// SVM类型.  C_SVC 该类型可以用于n-类分类问题 (n \geq 2)
    svm->setKernel(SVM::LINEAR);// SVM 核类型.  将训练样本映射到更有利于可线性分割的样本集
    // 算法终止条件.   最大迭代次数和容许误差
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    // 训练支持向量
    svm->train(trainingDataMat, ROW_SAMPLE, labelsMat);

    // SVM区域分割
    Vec3b green(0,255,0), blue (255,0,0);
    for (int i = 0; i < image.rows; ++i)
        for (int j = 0; j < image.cols; ++j)
        {
            Mat sampleMat = (Mat_<float>(1,2) << j,i);// generate data
            float response = svm->predict(sampleMat);// 分类类别
            if (response == 1)// 绿色表示标记为1的点，蓝色表示标记为-1的点。
                image.at<Vec3b>(i,j)  = green;
            else if (response == -1)
                image.at<Vec3b>(i,j)  = blue;
        }

    // Show the training data
    int thickness = -1;
    int lineType = 8;
    circle( image, Point(501,  10), 5, Scalar(  0,   0,   0), thickness, lineType );
    circle( image, Point(255,  10), 5, Scalar(255, 255, 255), thickness, lineType );
    circle( image, Point(501, 255), 5, Scalar(255, 255, 255), thickness, lineType );
    circle( image, Point( 10, 501), 5, Scalar(255, 255, 255), thickness, lineType );

    // 支持向量
    thickness = 2;
    lineType  = 8;
    Mat sv = svm->getUncompressedSupportVectors();
    for (int i = 0; i < sv.rows; ++i)
    {
        const float* v = sv.ptr<float>(i);
        circle( image,  Point( (int) v[0], (int) v[1]),   6,  Scalar(128, 128, 128), thickness, lineType);
    }

    imwrite("result.png", image);        // save the image
    imshow("SVM Simple Example", image); // show it to the user
    waitKey(0);
}






