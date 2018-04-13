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
#include "disparityrefinement.h"

const int DisparityRefinement::DISP_OCCLUSION = 1;
const int DisparityRefinement::DISP_MISMATCH = 2;

DisparityRefinement::DisparityRefinement(uint dispTolerance, int dMin, int dMax,
                                         uint votingThreshold, float votingRatioThreshold, uint maxSearchDepth,
                                         uint blurKernelSize, uint cannyThreshold1, uint cannyThreshold2, uint cannyKernelSize)
{
    this->occlusionValue = dMin - DISP_OCCLUSION;
    this->mismatchValue = dMin - DISP_MISMATCH;
    this->dispTolerance = dispTolerance;// Parameters for outlier detection　外点检测
    this->dMin = dMin;//最小搜索视差
    this->dMax = dMax;//最大搜索视差
    this->votingThreshold = votingThreshold;
    this->votingRatioThreshold = votingRatioThreshold;
    this->maxSearchDepth = maxSearchDepth;
    this->blurKernelSize = blurKernelSize;
    this->cannyThreshold1 = cannyThreshold1;
    this->cannyThreshold2 = cannyThreshold2;
    this->cannyKernelSize = cannyKernelSize;
}
//三通道颜色差值最大的一个
int DisparityRefinement::colorDiff(const Vec3b &p1, const Vec3b &p2)
{
    int colorDiff, diff = 0;

    for(uchar color = 0; color < 3; color++)
    {
        colorDiff = std::abs(p1[color] - p2[color]);
        diff = (diff > colorDiff)? diff: colorDiff;
    }

    return diff;
}
// 遮挡点，非稳定点，稳定点。对于遮挡点和非稳定点，只能基于稳定点的视差传播来解决
// 外点估计 不稳定点检测
Mat DisparityRefinement::outlierElimination(const Mat &leftDisp, const Mat &rightDisp)
{
    Size dispSize = leftDisp.size();
    Mat disparityMap(dispSize, CV_32S);

    //#pragma omp parallel for
    for(int h = 0; h < dispSize.height; h++)
    {
        for(int w = 0; w < dispSize.width; w++)
        {
            int disparity = leftDisp.at<int>(h, w);//左图视差

            // if the point is an outlier, differentiate it between occlusion and mismatch
            if(w - disparity < 0 || abs(disparity - rightDisp.at<int>(h, w - disparity)) > dispTolerance)
            {// 外点：不稳定点 + 遮挡点
                bool occlusion = true;// 遮挡点
                for(int d = dMin; d <= dMax && occlusion; d++)
                {
                    if(w - d >= 0 && d == rightDisp.at<int>(h, w - d))
                    {
                        occlusion = false;
                    }
                }
                disparity = (occlusion)? occlusionValue: mismatchValue;
            }

            disparityMap.at<int>(h, w) = disparity;
        }
    }

    return disparityMap;
}
// 　视差传播  迭代区域投票法
// 它的目的是对outlier进行填充处理，一般来说outlier遮挡点居多
// 填充最常用的方法就是用附近的稳定点就行了  但是不利于并行处理
void DisparityRefinement::regionVoting(Mat &disparity, const vector<Mat> &upLimits, const vector<Mat> &downLimits,
                                       const vector<Mat> &leftLimits, const vector<Mat> &rightLimits, bool horizontalFirst)
{
    // temporary disparity map that avoids too fast correction
    Size dispSize = disparity.size();
    Mat dispTemp(dispSize, CV_32S);

    // histogram for voting
    vector<int> hist(dMax - dMin + 1, 0);//每个区域求取视差直方图（不要归一化）
// 例如，得到的直方图共有15个bin，最大的bin值是8，那么outlier的视差就由这个8来决定，
// 但是稳定点的个数必须得比较多，比较多才有统计稳定性，
    const Mat *outerLimitsA, *outerLimitsB;
    const Mat *innerLimitsA, *innerLimitsB;

    if(horizontalFirst)
    {
        outerLimitsA = &upLimits[0];
        outerLimitsB = &downLimits[0];
        innerLimitsA = &leftLimits[0];
        innerLimitsB = &rightLimits[0];
    }
    else
    {
        outerLimitsA = &leftLimits[0];
        outerLimitsB = &rightLimits[0];
        innerLimitsA = &upLimits[0];
        innerLimitsB = &downLimits[0];
    }

    // loop on the whole picture
    for(size_t h = 0; h < dispSize.height; h++)
    {
        for(size_t w = 0; w < dispSize.width; w++)
        {
            // if the pixel is not an outlier
            if(disparity.at<int>(h, w) >= dMin)//稳定点
            {
                dispTemp.at<int>(h, w) = disparity.at<int>(h, w);
            }
            else//非稳定点
            {
                int outerLimitA = -outerLimitsA->at<int>(h, w);
                int outerLimitB = outerLimitsB->at<int>(h, w);
                int innerLimitA;
                int innerLimitB;
                int vote = 0;
                for(int outer = outerLimitA; outer <= outerLimitB; outer++)
                {
                    if(horizontalFirst)//先水平
                    {
                        innerLimitA = -innerLimitsA->at<int>(h + outer, w);
                        innerLimitB = innerLimitsB->at<int>(h + outer, w);
                    }
                    else//先垂直
                    {
                        innerLimitA = -innerLimitsA->at<int>(h, w + outer);
                        innerLimitB = innerLimitsB->at<int>(h, w + outer);
                    }


                    for(int inner = innerLimitA; inner <= innerLimitB; inner++)
                    {
                        int height, width;
                        if(horizontalFirst)
                        {
                            height = h + outer;
                            width = w + inner;
                        }
                        else
                        {
                            height = h + inner;
                            width = w + outer;
                        }

                        // if the pixel is an outlier, there is no vote to take into account
                        if(disparity.at<int>(height, width) >= dMin)
                        {
                            // increase the number of votes
                            vote++;
                            // update the histogram　　更新直方图
                            hist[disparity.at<int>(height, width) - dMin] += 1;
                        }
                    }
                }

                if(vote <= votingThreshold)
                {
                    dispTemp.at<int>(h, w) = disparity.at<int>(h, w);
                }
                else
                {
                    int disp = disparity.at<int>(h, w);
                    float voteRatio;
                    float voteRatioMax = 0;
                    for(int d = dMin; d <= dMax; d++)
                    {
                        voteRatio = hist[d - dMin] / (float)vote;
                        if(voteRatio > voteRatioMax)
                        {
                            voteRatioMax = voteRatio;
                            disp = (voteRatioMax > votingRatioThreshold)? d: disp;
                        }
                        hist[d - dMin] = 0;
                    }
                    dispTemp.at<int>(h, w) = disp;
                }
            }
        }
    }

    dispTemp.copyTo(disparity);
}
// 有些outlier由于区域内稳定点个数不满足公式，这样的区域用此方法是处理不来的，
// 只能进一步通过16方向极线插值来进一步填充，二者配合起来能够取得不错的效果
void DisparityRefinement::properInterpolation(Mat &disparity, const Mat &leftImage)
{
    Size dispSize = disparity.size();
    Mat dispTemp(dispSize, CV_32S);

    // look on the 16 different directions
    int directionsW[] = {0, 2, 2, 2, 0, -2, -2, -2, 1, 2, 2, 1, -1, -2, -2, -1};
    int directionsH[] = {2, 2, 0, -2, -2, -2, 0, 2, 2, 1, -1, -2, -2, -1, 1, 2};

    // loop on the whole picture
    for(size_t h = 0; h < dispSize.height; h++)
    {
        for(size_t w = 0; w < dispSize.width; w++)
        {
            // if the pixel is not an outlier
            if(disparity.at<int>(h, w) >= dMin)
            {
                dispTemp.at<int>(h, w) = disparity.at<int>(h, w);
            }
            else
            {
                vector<int> neighborDisps(16, disparity.at<int>(h, w));
                vector<int> neighborDiffs(16, -1);
                for(uchar direction = 0; direction < 16; direction++)
                {
                    int hD = h, wD = w;
                    bool inside = true, gotDisp = false;
                    for(uchar sD = 0; sD < maxSearchDepth && inside && !gotDisp; sD++)
                    {
                        // go one step further
                        if(sD % 2 == 0)
                        {
                            hD += directionsH[direction] / 2;
                            wD += directionsW[direction] / 2;
                        }
                        else
                        {
                            hD += directionsH[direction] - directionsH[direction] / 2;
                            wD += directionsW[direction] - directionsW[direction] / 2;
                        }
                        inside = hD >= 0 && hD < dispSize.height && wD >= 0 && wD < dispSize.width;
                        if(inside && disparity.at<int>(hD, wD) >= dMin)
                        {
                            neighborDisps[direction] = disparity.at<int>(hD, wD);
                            neighborDiffs[direction] = colorDiff(leftImage.at<Vec3b>(h, w), leftImage.at<Vec3b>(hD, wD));
                            gotDisp = true;
                        }
                    }

                }

                if(disparity.at<int>(h, w) == dMin - DISP_OCCLUSION)
                {
                    int minDisp = neighborDisps[0];
                    for(uchar direction = 1; direction < 16; direction++)
                    {
                        if(minDisp > neighborDisps[direction])
                            minDisp = neighborDisps[direction];
                    }
                    dispTemp.at<int>(h, w) = minDisp;
                }
                else
                {
                    int minDisp = neighborDisps[0];
                    int minDiff = neighborDiffs[0];
                    for(uchar dir = 1; dir < 16; dir++)
                    {
                        if(minDiff < 0 || (minDiff > neighborDiffs[dir] && neighborDiffs[dir] > 0))
                        {
                            minDisp = neighborDisps[dir];
                            minDiff = neighborDiffs[dir];
                        }
                    }
                    dispTemp.at<int>(h, w) = minDisp;
                }
            }
        }
    }

    dispTemp.copyTo(disparity);
}
// 边缘矫正
void DisparityRefinement::discontinuityAdjustment(Mat &disparity, const vector<vector<Mat> > &costs)
{
    Size dispSize = disparity.size();
    Mat dispTemp, detectedEdges, dispGray;

    disparity.copyTo(dispTemp);

    //Edge Detection
    dispGray = convertDisp2Gray(disparity);// 深度图转灰度图　　　用于边缘检测
    blur(dispGray, detectedEdges, Size(blurKernelSize, blurKernelSize));// 视差图平滑
    Canny(detectedEdges, detectedEdges, cannyThreshold1, cannyThreshold2, cannyKernelSize);//candy 边缘检测

    int directionsH[] = {-1, 1, -1, 1, -1, 1, 0, 0};
    int directionsW[] = {-1, 1, 0, 0, 1, -1, -1, 1};

    for(size_t h = 1; h < dispSize.height - 1; h++)
    {
        for(size_t w = 1; w < dispSize.width - 1; w++)
        {
            // if pixel is on an edge
            if(detectedEdges.at<uchar>(h, w) != 0)
            {
                int direction = -1;
                if(detectedEdges.at<uchar>(h - 1, w - 1) != 0 && detectedEdges.at<uchar>(h + 1, w + 1) != 0)
                {
                    direction = 0;
                }
                else if(detectedEdges.at<uchar>(h - 1, w + 1) != 0 && detectedEdges.at<uchar>(h + 1, w - 1) != 0)
                {
                    direction = 4;
                }
                else if(detectedEdges.at<uchar>(h - 1, w) != 0 || detectedEdges.at<uchar>(h + 1, w) != 0)
                {
                    if(detectedEdges.at<uchar>(h - 1, w - 1) != 0 || detectedEdges.at<uchar>(h - 1, w) != 0 || detectedEdges.at<uchar>(h - 1, w + 1) != 0)
                        if(detectedEdges.at<uchar>(h + 1, w - 1) != 0 || detectedEdges.at<uchar>(h + 1, w) != 0 || detectedEdges.at<uchar>(h + 1, w + 1) != 0)
                            direction = 2;
                }
                else
                {
                    if(detectedEdges.at<uchar>(h - 1, w - 1) != 0 || detectedEdges.at<uchar>(h, w - 1) != 0 || detectedEdges.at<uchar>(h + 1, w - 1) != 0)
                        if(detectedEdges.at<uchar>(h - 1, w + 1) != 0 || detectedEdges.at<uchar>(h, w + 1) != 0 || detectedEdges.at<uchar>(h + 1, w + 1) != 0)
                            direction = 6;
                }

                if (direction != -1)
                {
                    dispTemp.at<int>(h, w) = dMin - DISP_MISMATCH;

                    int disp = disparity.at<int>(h, w);

                    // select pixels from both sides of the edge
                    direction = (direction + 4) % 8;

                    if(disp >= dMin)
                    {
                        costType cost = costs[0][disp - dMin].at<costType>(h, w);
                        int d1 = disparity.at<int>(h + directionsH[direction], w + directionsW[direction]);
                        int d2 = disparity.at<int>(h + directionsH[direction + 1], w + directionsW[direction + 1]);

                        costType cost1 = (d1 >= dMin)
                                     ? costs[0][d1 - dMin].at<costType>(h + directionsH[direction], w + directionsW[direction])
                                     : -1;

                        costType cost2 = (d2 >= dMin)
                                     ? costs[0][d2 - dMin].at<costType>(h + directionsH[direction + 1], w + directionsW[direction + 1])
                                     : -1;

                        if(cost1 != -1 && cost1 < cost)
                        {
                            disp = d1;
                            cost = cost1;
                        }

                        if(cost2 != -1 && cost2 < cost)
                        {
                            disp = d2;
                        }
                    }

                    dispTemp.at<int>(h, w) = disp;

                }
            }
        }
    }

    dispTemp.copyTo(disparity);
}
//  亚像素求精
Mat DisparityRefinement::subpixelEnhancement(Mat &disparity, const vector<vector<Mat> > &costs)
{
    Size dispSize = disparity.size();
    Mat dispTemp(dispSize, CV_32F);

    for(size_t h = 0; h < dispSize.height; h++)
    {
        for(size_t w = 0; w < dispSize.width; w++)
        {
            int disp = disparity.at<int>(h, w);//视差
            float interDisp = disp;

            if(disp > dMin && disp < dMax)//稳定点
            {
                float cost = costs[0][disp - dMin].at<costType>(h, w) / (float)COST_FACTOR;
                float costPlus = costs[0][disp + 1 - dMin].at<costType>(h, w) / (float)COST_FACTOR;
                float costMinus = costs[0][disp - 1 - dMin].at<costType>(h, w) / (float)COST_FACTOR;

                float diff = (costPlus - costMinus) / (2 * (costPlus + costMinus - 2 * cost));//亚像素

                if(diff > -1 && diff < 1)
                    interDisp -= diff;
            }

            dispTemp.at<float>(h, w) = interDisp;
        }
    }

    medianBlur(dispTemp, dispTemp, 3);//中值滤波
    return dispTemp;
}
// 深度图转灰度图　　　用于边缘检测
Mat DisparityRefinement::convertDisp2Gray(const Mat &disparity)
{
    Size dispSize = disparity.size();
    Mat dispU(dispSize, CV_8U);

    for(size_t h = 0; h < dispSize.height; h++)
    {
        for(size_t w = 0; w < dispSize.width; w++)
        {//小于0的点　提高到0
            dispU.at<uchar>(h, w) = (disparity.at<int>(h, w) < 0)? 0: (uchar)disparity.at<int>(h, w);
        }
    }

    equalizeHist(dispU, dispU);//直方图均衡化
    return dispU;
}
