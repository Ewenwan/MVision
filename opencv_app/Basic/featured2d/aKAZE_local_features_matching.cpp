/*非线性尺度空间的KAZE特征提取
【1】非线性尺度空间 AKAZE特征检测 
【2】检测 + 描述
【3】汉明距离匹配器
【4】进行描述子匹配
【5】最近距离的匹配/次近距离的匹配 小于一个阈值 才认为是 初级匹配点对 
【6】一个匹配点 单应变换后 和 其匹配点的 差平方和 开根号 < 阈值  为最终的匹配点对
【7】显示匹配

提供图1到图2的单应变换矩阵 3*3


基于非线性尺度空间的KAZE特征提取方法以及它的改进AKATE
https://blog.csdn.net/chenyusiyuan/article/details/8710462

KAZE是日语‘风’的谐音，寓意是就像风的形成是空气在空间中非线性的流动过程一样，
KAZE特征检测是在图像域中进行非线性扩散处理的过程。

传统的SIFT、SURF等特征检测算法都是基于 线性的高斯金字塔 进行多尺度分解来消除噪声和提取显著特征点。
但高斯分解是牺牲了局部精度为代价的，容易造成边界模糊和细节丢失。

非线性的尺度分解有望解决这种问题，但传统方法基于正向欧拉法（forward Euler scheme）
求解非线性扩散（Non-linear diffusion）方程时迭代收敛的步长太短，耗时长、计算复杂度高。

由此，KAZE算法的作者提出采用加性算子分裂算法(Additive Operator Splitting, AOS)
来进行非线性扩散滤波，可以采用任意步长来构造稳定的非线性尺度空间。


非线性扩散滤波
	Perona-Malik扩散方程:
		具体地，非线性扩散滤波方法是将图像亮度（L）在不同尺度上的变化视为某种形式的
		流动函数（flow function）的散度（divergence），可以通过非线性偏微分方程来描述：
	AOS算法:
		由于非线性偏微分方程并没有解析解，一般通过数值分析的方法进行迭代求解。
		传统上采用显式差分格式的求解方法只能采用小步长，收敛缓慢。

KAZE特征检测与描述

KAZE特征的检测步骤大致如下：
1) 首先通过AOS算法和可变传导扩散（Variable  Conductance  Diffusion）（[4,5]）方法来构造非线性尺度空间。
2) 检测感兴趣特征点，这些特征点在非线性尺度空间上经过尺度归一化后的Hessian矩阵行列式是局部极大值（3×3邻域）。
3) 计算特征点的主方向，并且基于一阶微分图像提取具有尺度和旋转不变性的描述向量。

特征点检测
KAZE的特征点检测与SURF类似，是通过寻找不同尺度归一化后的Hessian局部极大值点来实现的。

*/

#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

// 全局变量 
const float inlier_threshold = 2.5f; // 单应变换后 和匹配点差值阈值
const float nn_match_ratio = 0.8f;   // 最近距离/次近距离 < 阈值 

int main(int argc, char** argv)
{
/// 加载源图像
    string imageName1("../../common/data/graf1.png"); // 图片文件名路径（默认值）
    string imageName2("../../common/data/graf3.png"); // 图片文件名路径（默认值）
    if( argc > 2)
    {
        imageName1 = argv[1];//如果传递了文件 就更新
	imageName2 = argv[2];//如果传递了文件 就更新
    }
    Mat img1 = imread( imageName1, CV_LOAD_IMAGE_GRAYSCALE );
    Mat img2  = imread( imageName2, CV_LOAD_IMAGE_GRAYSCALE );
    if( img1.empty() || img2.empty() )
    { 
        cout <<  "can't load image " << endl;
	return -1;  
    }
    // 单应变换矩阵
    Mat homography;
    FileStorage fs("../../common/data/H1to3p.xml", FileStorage::READ);
    fs.getFirstTopLevelNode() >> homography;


    vector<KeyPoint> kpts1, kpts2;//关键点
    Mat desc1, desc2;//描述子
    // 非线性尺度空间 AKAZE特征检测
    Ptr<AKAZE> akaze = AKAZE::create();
    // 检测 + 描述
    akaze->detectAndCompute(img1, noArray(), kpts1, desc1);
    akaze->detectAndCompute(img2, noArray(), kpts2, desc2);
    // 汉明距离匹配器
    BFMatcher matcher(NORM_HAMMING);
    // 匹配点对 二维数组 一个点检测多个匹配点(按距离远近)
    vector< vector<DMatch> > nn_matches;
    // 进行描述子匹配
    matcher.knnMatch(desc1, desc2, nn_matches, 2);

    vector<KeyPoint> matched1, matched2, inliers1, inliers2;
     // matched1, matched2, 最近距离的匹配/次近距离的匹配的匹配点
     // inliers1, inliers2 一个匹配点 单应变换后 和 其匹配点的 差平方和 开根号 小于 阈值的 更好的匹配点
    vector<DMatch> good_matches;//较好的匹配

   //====最近距离的匹配/次近距离的匹配 小于一个阈值 才认为是 匹配点对 ================
    for(size_t i = 0; i < nn_matches.size(); i++) {
        DMatch first = nn_matches[i][0];
        float dist1 = nn_matches[i][0].distance;//最近距离的匹配
        float dist2 = nn_matches[i][1].distance;//次近距离的匹配 
        if(dist1 < nn_match_ratio * dist2) {
            matched1.push_back(kpts1[first.queryIdx]);// 对应图1的关键点坐标
            matched2.push_back(kpts2[first.trainIdx]);// 对应图2的关键点坐标
        }
    }

   //==========一个匹配点 单应变换后 和 其匹配点的 差平方和 开根号 < 阈值 为最终的匹配点对
    for(unsigned i = 0; i < matched1.size(); i++) {
        Mat col = Mat::ones(3, 1, CV_64F);//图1 点 的 其次表达方式 (u,v,1)
        col.at<double>(0) = matched1[i].pt.x;
        col.at<double>(1) = matched1[i].pt.y;
        col = homography * col;// 单应变换 后
        col /= col.at<double>(2);// 将第三维归一化（x,y,z）-> （x/z,y/z,z/z）->(u',v',1)
        // 计算 一个匹配点 单应变换后 和 其匹配点的 差平方和 开根号
        double dist = sqrt( pow(col.at<double>(0) - matched2[i].pt.x, 2) +
                            pow(col.at<double>(1) - matched2[i].pt.y, 2));

        if(dist < inlier_threshold) {
            int new_i = static_cast<int>(inliers1.size());
            inliers1.push_back(matched1[i]);// 内点 符合 单应变换的   点
            inliers2.push_back(matched2[i]);// 是按 匹配点对方式存入的 
            good_matches.push_back(DMatch(new_i, new_i, 0));//所以匹配点对 1-1 2-2 3-3 4-4 5-5
        }
    }

// ======= 显示 ===============
    Mat res;//最后的图像
    drawMatches(img1, inliers1, img2, inliers2, good_matches, res, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    imshow("res.png", res);//显示匹配后的图像
    
    double inlier_ratio = inliers1.size() * 1.0 / matched1.size();
    cout << "A-KAZE Matching Results" << endl;
    cout << "*******************************" << endl;
    cout << "# Keypoints 1:                        \t" << kpts1.size() << endl;
    cout << "# Keypoints 2:                        \t" << kpts2.size() << endl;
    cout << "# Matches:                            \t" << matched1.size() << endl;
    cout << "# Inliers:                            \t" << inliers1.size() << endl;
    cout << "# Inliers Ratio:                      \t" << inlier_ratio << endl;
    cout << endl;
    return 0;
}
