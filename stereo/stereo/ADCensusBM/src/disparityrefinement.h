/*
视差优化

和其他文献相同，作者也采用left-right-check的方法将像素点分类成为三种：
遮挡点，非稳定点，稳定点。
对于遮挡点和非稳定点，只能基于稳定点的视差传播来解决，
本文在视差传播这一块采用了两种方法，一个是迭代区域投票法，
另一个是16方向极线插值法，后者具体来说：
沿着点p的16个方向开始搜索，如果p是遮挡点，那么遇到的第一个稳定点的视差就是p的视差，
如果p是非稳定点，那么遇到第一个与p点颜色相近的稳定点的视差作为p的视差。

针对于视差图的边缘，作者直接提取两侧的像素的代价，
如果有一个代价比较小，那么就采用这个点的视差值作为边缘点的视差值，
至于边缘点是如何检测出来的，很简单，对视差图随便应用于一个边缘检测算法即可。

做完这些之后，别忘记亚像素求精，这和WTA一样，是必不可少的。
最后再来一个中值滤波吧，因为大家都这么玩，属于后处理潜规则。

开始----->输入左右视差图----> 不稳定点检测 ------> 视差传播－迭代区域投票法------> 
视差传播-16方向极线插值算法 -----> 边缘矫正 ------> 亚像素求精 ------->  中值滤波 ------> 结束

*/
#ifndef DISPARITYREFINEMENT_H
#define DISPARITYREFINEMENT_H

#include <opencv2/opencv.hpp>
#include "adcensuscv.h"
#include "common.h"

using namespace cv;
using namespace std;

class DisparityRefinement
{
public:
    DisparityRefinement(uint dispTolerance, int dMin, int dMax,
                        uint votingThreshold, float votingRatioThreshold, uint maxSearchDepth,
                        uint blurKernelSize, uint cannyThreshold1, uint cannyThreshold2, uint cannyKernelSize);
    Mat outlierElimination(const Mat &leftDisp, const Mat &rightDisp);
    void regionVoting(Mat &disparity, const vector<Mat> &upLimits, const vector<Mat> &downLimits,
                      const vector<Mat> &leftLimits, const vector<Mat> &rightLimits, bool horizontalFirst);
    void properInterpolation(Mat &disparity, const Mat &leftImage);
    void discontinuityAdjustment(Mat &disparity, const vector<vector<Mat> > &costs);
    Mat subpixelEnhancement(Mat &disparity, const vector<vector<Mat> > &costs);


    static const int DISP_OCCLUSION;
    static const int DISP_MISMATCH;
private:
    int colorDiff(const Vec3b &p1, const Vec3b &p2);
    Mat convertDisp2Gray(const Mat &disparity);

    int occlusionValue;
    int mismatchValue;
    uint dispTolerance;
    int dMin;
    int dMax;
    uint votingThreshold;
    float votingRatioThreshold;
    uint maxSearchDepth;
    uint blurKernelSize;
    uint cannyThreshold1;
    uint cannyThreshold2;
    uint cannyKernelSize;
};

#endif // DISPARITYREFINEMENT_H
