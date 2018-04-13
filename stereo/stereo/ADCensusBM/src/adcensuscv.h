/*
ADCensus 代价计算  = AD＋Census
     1.  AD即 Absolute Difference 三通道颜色差值绝对值之和求均值
     2. Census Feature特征原理很简单
　　　　　　　         在指定窗口内比较周围亮度值与中心点的大小，匹配距离用汉明距表示。
　　　　　　　　　　　　　   Census保留周边像素空间信息，对光照变化有一定鲁棒性。

就是在给定的窗口内，比较中心像素与周围邻居像素之间的大小关系，大了就为1，小了就为0，
然后每个像素都对应一个二值编码序列，然后通过海明距离来表示两个像素的相似程度，

     3. 信息结合
　　　　　　　　　　　　　　　　cost = r(Cad , lamd1) + r(Cces, lamd2)
        r(C , lamd) = 1 - exp(- c/ lamd)
   　 　　Cross-based 代价聚合:
        自适应窗口代价聚合，
		在设定的最大窗口范围内搜索，
　　　　　　　　　　　　　满足下面三个约束条件确定每个像素的十字坐标，完成自适应窗口的构建。
　　　　　　　　　　　　　　　Scanline 代价聚合优化
*/
#ifndef ADCENSUSCV_H
#define ADCENSUSCV_H

#include <opencv2/opencv.hpp>

using namespace cv;

class ADCensusCV
{
public:
    ADCensusCV(const Mat &leftImage, const Mat &rightImage, Size censusWin, float lambdaAD, float lambdaCensus);
// 对应像素差的绝对值 3通道均值  Absolute Differences
    float ad(int wL, int hL, int wR, int hR) const;
// census值
    float census(int wL, int hL, int wR, int hR) const;
// ad + census值 信息结合
    float adCensus(int wL, int hL, int wR, int hR) const;
private:
    Mat leftImage;
    Mat rightImage;
    Size censusWin;
    float lambdaAD;
    float lambdaCensus;
};

#endif // ADCENSUSCV_H
