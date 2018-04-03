/*
人工神经网络(ANN) 简称神经网络(NN)，
能模拟生物神经系统对物体所作出的交互反应，
是由具有适应性的简单单元(称为神经元)组成的广泛并行互连网络。
 神经元
 y = w  * x + b   线性变换   （旋转 平移 伸缩 升/降维）
 z = a(y)         非线性变换 激活函数

常用 Sigmoid 函数作激活函数

y = sigmod(x)  = 1/(1+exp(-x))  映射到0 ～ 1之间
 
 OpenCV 中使用的激活函数是另一种形式，

f(x) = b *  (1 - exp(-c*x)) / (1 + exp(-c*x))

当 α = β = 1 时
f(x) =(1 - exp(-x)) / (1 + exp(-x))
该函数把可能在较大范围内变化的输入值，“挤压” 到 (-1, 1) 的输出范围内

// 设置激活函数，目前只支持 ANN_MLP::SIGMOID_SYM
virtual void cv::ml::ANN_MLP::setActivationFunction(int type, double param1 = 0, double param2 = 0); 

神经网络
2.1  感知机 (perceptron)
  感知机由两层神经元组成，输入层接收外界输入信号，而输出层则是一个 M-P 神经元。
  实际上，感知机可视为一个最简单的“神经网络”，用它可很容易的实现逻辑与、或、非等简单运算。

2.2 层级结构
  常见的神经网络，可分为三层：输入层、隐含层、输出层。
  输入层接收外界输入，隐层和输出层负责对信号进行加工，输出层输出最终的结果。

2.3  层数设置
  	OpenCV 中，设置神经网络层数和神经元个数的函数为 setLayerSizes(InputArray _layer_sizes)，
	// (a) 3层，输入层神经元个数为 4，隐层的为 6，输出层的为 4
	Mat layers_size = (Mat_<int>(1,3) << 4,6,4);

	// (b) 4层，输入层神经元个数为 4，第一个隐层的为 6，第二个隐层的为 5，输出层的为 4
	Mat layers_size = (Mat_<int>(1,4) << 4,6,5,4);

1)  创建
	static Ptr<ANN_MLP> cv::ml::ANN_MLP::create();  // 创建空模型

2) 设置参数

// 设置神经网络的层数和神经元数量
virtual void cv::ml::ANN_MLP::setLayerSizes(InputArray _layer_sizes);

// 设置激活函数，目前只支持 ANN_MLP::SIGMOID_SYM
virtual void cv::ml::ANN_MLP::setActivationFunction(int type, double param1 = 0, double param2 = 0); 

// 设置训练方法，默认为 ANN_MLP::RPROP，较常用的是 ANN_MLP::BACKPROP
// 若设为 ANN_MLP::BACKPROP，则 param1 对应 setBackpropWeightScale()中的参数,
// param2 对应 setBackpropMomentumScale() 中的参数
virtual void cv::ml::ANN_MLP::setTrainMethod(int method, double param1 = 0, double param2 = 0);
virtual void cv::ml::ANN_MLP::setBackpropWeightScale(double val); // 默认值为 0.1
virtual void cv::ml::ANN_MLP::setBackpropMomentumScale(double val); // 默认值为 0.1
 
// 设置迭代终止准则，默认为 TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 0.01)
virtual void cv::ml::ANN_MLP::setTermCriteria(TermCriteria val);

3)  训练

// samples - 训练样本; layout - 训练样本为 “行样本” ROW_SAMPLE 或 “列样本” COL_SAMPLE; response - 对应样本数据的分类结果
virtual bool cv::ml::StatModel::train(InputArray samples,int layout,InputArray responses);  

4)  预测

// samples，输入的样本书数据；results，输出矩阵，默认不输出；flags，标识，默认为 0
virtual float cv::ml::StatModel::predict(InputArray samples, OutputArray results=noArray(),int flags=0) const;　　　　　　 

5) 保存训练好的神经网络参数
    bool trained = ann->train(tData);  
    if (trained) {
          ann->save("ann_param");
     }

6) 载入训练好的神经网络
      Ptr<ml::ANN_MLP> ann = ml::ANN_MLP::load("ann_param");

neuralNetwork = Algorithm::load<ANN_MLP>(fileName);

*/
#include<opencv2/opencv.hpp>  
#include <iostream>    
#include <string>    
  
using namespace std;  
using namespace cv;  
using namespace ml;  
int main()  
{  
    // 标签
    float labels[10][2]={ { 0.9,0.1 },
			  { 0.1,0.9 },
			  { 0.9,0.1 },
			  { 0.1,0.9 },
			  { 0.9,0.1 },
			  { 0.9,0.1 },
			  { 0.1,0.9 },
			  { 0.1,0.9 },
			  { 0.9,0.1 },
			  { 0.9,0.1 } };  
    //这里对于样本标记为0.1和0.9而非0和1，主要是考虑到sigmoid函数的输出为一般为0和1之间的数，
    // 只有在输入趋近于-∞和+∞才逐渐趋近于0和1，而不可能达到。  
    Mat labelsMat(10, 2, CV_32FC1, labels);  
    // 训练集
    float trainingData[10][2] = { { 11,12   },
				  { 111,112 },
				  { 21,22   },
			 	  { 211,212 },
				  { 51,32   },
				  { 71,42   },
				  { 441,412 },
				  { 311,312 },
				  { 41,62   },
				  { 81,52   } };  
    Mat trainingDataMat(10, 2, CV_32FC1, trainingData);  

    // BP 神经网络 模型  创建和参数设置
    Ptr<ANN_MLP> ann = ANN_MLP::create();
    //5层：输入层，3层隐藏层和输出层，每层均为两个神经元perceptron    
    Mat layerSizes = (Mat_<int>(1, 5) << 2, 2, 2, 2, 2); 
    ann->setLayerSizes(layerSizes);//  
    // 设置激活函数，目前只支持 ANN_MLP::SIGMOID_SYM 
    ann->setActivationFunction(ANN_MLP::SIGMOID_SYM,1.0,1.0);//
    // 设置迭代结束条件 
    //  ann->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 300, FLT_EPSILON));  
    // 设置训练方法   反向传播  动量 下山发
    ann->setTrainMethod(ANN_MLP::BACKPROP,0.1,0.9);  
    // 训练数据
    Ptr<TrainData> tData = TrainData::create(trainingDataMat, ROW_SAMPLE, labelsMat);
    // 执行训练  
    bool trained = ann->train(tData);  
    // 保存训练好的神经网络参数
    if (trained) {
          ann->save("ann_param");
     }

    // 载入训练好的神经网络
    //    Ptr<ml::ANN_MLP> ann = ml::ANN_MLP::load("ann_param");

    //  512 x 512 零矩阵   
    int width = 512, height = 512;  
    Mat image = Mat::zeros(height, width, CV_8UC3);  

    Vec3b green(0, 255, 0), blue(255, 0, 0);  
    for (int i = 0; i < image.rows; ++i)  
    {  
        for (int j = 0; j < image.cols; ++j)  
        {  
            Mat sampleMat = (Mat_<float>(1, 2) << i, j);//生成数
            Mat responseMat;//预测结果  我前面定义的 输出为两维
            ann->predict(sampleMat, responseMat);  
            float* p = responseMat.ptr<float>(0);  
            if (p[0] > p[1])  
            {  
                image.at<Vec3b>(j, i) = green;// 正样本 绿色
            }  
            else  
            {  
                image.at<Vec3b>(j, i) = blue; // 负样本 蓝色
            }  
        }  
    }  
    // 画出训练样本数据  
    int thickness = -1;  
    int lineType = 8;
    int r = 8;  //半径
    circle(image, Point(111, 112), r, Scalar(0, 0, 0), thickness, lineType);  
    circle(image, Point(211, 212), r, Scalar(0, 0, 0), thickness, lineType);  
    circle(image, Point(441, 412), r, Scalar(0, 0, 0), thickness, lineType);  
    circle(image, Point(311, 312), r, Scalar(0, 0, 0), thickness, lineType);  
    circle(image, Point(11, 12), r, Scalar(255, 255, 255), thickness, lineType);  
    circle(image, Point(21, 22), r, Scalar(255, 255, 255), thickness, lineType);  
    circle(image, Point(51, 32), r, Scalar(255, 255, 255), thickness, lineType);  
    circle(image, Point(71, 42), r, Scalar(255, 255, 255), thickness, lineType);  
    circle(image, Point(41, 62), r, Scalar(255, 255, 255), thickness, lineType);  
    circle(image, Point(81, 52), r, Scalar(255, 255, 255), thickness, lineType);  
  
    imwrite("result.png", image);       //  保存训练的结果  
  
    imshow("BP Simple Example", image); //    
    waitKey(0);  
  
    return 0;  
}  
