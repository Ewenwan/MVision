/*
支持向量机对线性不可分数据的处理

为什么需要将支持向量机优化问题扩展到线性不可分的情形？ 
在多数计算机视觉运用中，我们需要的不仅仅是一个简单的SVM线性分类器， 
我们需要更加强大的工具来解决 训练数据无法用一个超平面分割 的情形。

我们以人脸识别来做一个例子，训练数据包含一组人脸图像和一组非人脸图像(除了人脸之外的任何物体)。 
这些训练数据超级复杂，以至于为每个样本找到一个合适的表达 (特征向量) 以让它们能够线性分割是非常困难的。


最优化问题的扩展

还记得我们用支持向量机来找到一个最优超平面。 既然现在训练数据线性不可分，
我们必须承认这个最优超平面会将一些样本划分到错误的类别中。 在这种情形下的优化问题，
需要将 错分类(misclassification) 当作一个变量来考虑。
新的模型需要包含原来线性可分情形下的最优化条件，即最大间隔(margin), 
以及在线性不可分时分类错误最小化。


比如，我们可以最小化一个函数，该函数定义为在原来模型的基础上再加上一个常量乘以样本被错误分类的次数:
L = 1/2 * ||w||^2 - SUM(a * yi *( w* xi +b) - 1)  +  c * ( 样本被错误分类的次数)

它没有考虑错分类的样本距离同类样本所属区域的大小。 因此一个更好的方法是考虑 错分类样本离同类区域的距离:

L = 1/2 * ||w||^2 - SUM(a * yi *( w* xi +b) - 1)  +  c * ( 错分类样本离同类区域的距离)


   C比较大时分类错误率较小，但是间隔也较小。 在这种情形下，
错分类对模型函数产生较大的影响，既然优化的目的是为了最小化这个模型函数，
那么错分类的情形必然会受到抑制。

   C比较小时间隔较大，但是分类错误率也较大。 在这种情形下，
模型函数中错分类之和这一项对优化过程的影响变小，
优化过程将更加关注于寻找到一个能产生较大间隔的超平面。


*/
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

#define NTRAINING_SAMPLES   100 // Number of training samples per class

#define FRAC_LINEAR_SEP     0.9f// Fraction of samples which compose the linear separable part

using namespace cv;
using namespace cv::ml;
using namespace std;

static void help()
{
    cout<< "\n--------------------------------------------------------------------------" << endl
        << "This program shows Support Vector Machines for Non-Linearly Separable Data. " << endl
        << "Usage:"                                                               << endl
        << "./non_linear_svms" << endl
        << "--------------------------------------------------------------------------"   << endl
        << endl;
}
int main()
{
    help();

    // Data for visual representation
    const int WIDTH = 512, HEIGHT = 512;
    Mat I = Mat::zeros(HEIGHT, WIDTH, CV_8UC3);

//=== 1. Set up training data randomly=======================================
    Mat trainData(2*NTRAINING_SAMPLES, 2, CV_32FC1);
    Mat labels   (2*NTRAINING_SAMPLES, 1, CV_32SC1);
    RNG rng(100); // Random value generation class
    // Set up the linearly separable part of the training data
    int nLinearSamples = (int) (FRAC_LINEAR_SEP * NTRAINING_SAMPLES);

//===Generate random points for the class 1=========
    Mat trainClass = trainData.rowRange(0, nLinearSamples);
    // The x coordinate of the points is in [0, 0.4)
    Mat c = trainClass.colRange(0, 1);
    rng.fill(c, RNG::UNIFORM, Scalar(1), Scalar(0.4 * WIDTH));
    // The y coordinate of the points is in [0, 1)
    c = trainClass.colRange(1,2);
    rng.fill(c, RNG::UNIFORM, Scalar(1), Scalar(HEIGHT));

//=====Generate random points for the class 2==========
    trainClass = trainData.rowRange(2*NTRAINING_SAMPLES-nLinearSamples, 2*NTRAINING_SAMPLES);
    // The x coordinate of the points is in [0.6, 1]
    c = trainClass.colRange(0 , 1);
    rng.fill(c, RNG::UNIFORM, Scalar(0.6*WIDTH), Scalar(WIDTH));
    // The y coordinate of the points is in [0, 1)
    c = trainClass.colRange(1,2);
    rng.fill(c, RNG::UNIFORM, Scalar(1), Scalar(HEIGHT));

//=====Set up the non-linearly separable part of the training data ===========
    // Generate random points for the classes 1 and 2
    trainClass = trainData.rowRange(  nLinearSamples, 2*NTRAINING_SAMPLES-nLinearSamples);
    // The x coordinate of the points is in [0.4, 0.6)
    c = trainClass.colRange(0,1);
    rng.fill(c, RNG::UNIFORM, Scalar(0.4*WIDTH), Scalar(0.6*WIDTH));
    // The y coordinate of the points is in [0, 1)
    c = trainClass.colRange(1,2);
    rng.fill(c, RNG::UNIFORM, Scalar(1), Scalar(HEIGHT));

//=========Set up the labels for the classes ===============
    labels.rowRange(                0,   NTRAINING_SAMPLES).setTo(1);  // Class 1
    labels.rowRange(NTRAINING_SAMPLES, 2*NTRAINING_SAMPLES).setTo(2);  // Class 2


//============= 2. Set up the support vector machines parameters ===============
    //=======3. Train the svm =======
    cout << "Starting training process" << endl;
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);// SVM类型.  C_SVC 该类型可以用于n-类分类问题 (n \geq 2)
    svm->setC(0.1);//  错分类样本离同类区域的距离 的权重
    svm->setKernel(SVM::LINEAR);// SVM 核类型.  将训练样本映射到更有利于可线性分割的样本集
    // 算法终止条件.   最大迭代次数和容许误差
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, (int)1e7, 1e-6));
    svm->train(trainData, ROW_SAMPLE, labels);
    cout << "Finished training process" << endl;

//================ 4. Show the decision regions ==================
    Vec3b green(0,100,0), blue (100,0,0);
    for (int i = 0; i < I.rows; ++i)
        for (int j = 0; j < I.cols; ++j)
        {
            Mat sampleMat = (Mat_<float>(1,2) << i, j);
            float response = svm->predict(sampleMat);
            if      (response == 1)    I.at<Vec3b>(j, i)  = green;
            else if (response == 2)    I.at<Vec3b>(j, i)  = blue;
        }
//================5. Show the training data ====================
    int thick = -1;
    int lineType = 8;
    float px, py;
 // Class 1===============
    for (int i = 0; i < NTRAINING_SAMPLES; ++i)
    {
        px = trainData.at<float>(i,0);
        py = trainData.at<float>(i,1);
        circle(I, Point( (int) px,  (int) py ), 3, Scalar(0, 255, 0), thick, lineType);
    }
  // Class 2======================
    for (int i = NTRAINING_SAMPLES; i <2*NTRAINING_SAMPLES; ++i)
    {
        px = trainData.at<float>(i,0);
        py = trainData.at<float>(i,1);
        circle(I, Point( (int) px, (int) py ), 3, Scalar(255, 0, 0), thick, lineType);
    }

//============== 6. Show support vectors =====================
    thick = 2;
    lineType  = 8;
    Mat sv = svm->getSupportVectors();
    for (int i = 0; i < sv.rows; ++i)
    {
        const float* v = sv.ptr<float>(i);
        circle( I,  Point( (int) v[0], (int) v[1]), 6, Scalar(128, 128, 128), thick, lineType);
    }
    imwrite("result.png", I);                      // save the Image
    imshow("SVM for Non-Linear Training Data", I); // show it to the user
    waitKey(0);
}






