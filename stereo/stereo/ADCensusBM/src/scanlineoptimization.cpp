/*
扫描线优化 
这个扫描线优化是动态规划的一种方法，在史上最经典立体匹配文献SGM中首次被使用，
具体的思想本文就不详细说了，由于代价聚合的结果不大靠谱，可以考虑将其视作数据项，
建立全局能量函数（公式如下所示），这样便直接过渡到了全局算法。

第一项C就是代价聚合项，后面两项分别考虑到了视差的微变（低纹理区域）和剧烈变化（物体边缘），
优化这个能量函数做采用的方法就是“扫描线优化法”，其公式如下所示，这里和SGM一模一样
https://blog.csdn.net/wsj998689aa/article/details/49403353

*/
#include "scanlineoptimization.h"

#include <iostream>

using namespace std;

ScanlineOptimization::ScanlineOptimization(const Mat &leftImage, const Mat &rightImage, int dMin, int dMax,
                                           uint colorThreshold, float pi1, float pi2)
{
    this->images[0] = leftImage;
    this->images[1] = rightImage;
    this->imgSize = leftImage.size();
    this->dMin = dMin;
    this->dMax = dMax;
    this->colorDifference = colorThreshold;
    this->pi1 = pi1;
    this->pi2 = pi2;
}

void ScanlineOptimization::optimization(vector<Mat> *costMaps, bool rightFirst)
{
//    vector<Mat> optimizedCosts(dMax - dMin + 1);
//    for(int d = 0; d < optimizedCosts.size(); d++)
//        optimizedCosts[d] = Mat::zeros(imgSize, CV_32F);

    // computes the 4 directionnal optimized costs
    // vertical downward
    cout << "[ScanlineOptimization] started 1. vertical computation" << endl;
    verticalComputation(0, 1, costMaps, rightFirst);

    // vertical upward
    cout << "[ScanlineOptimization] started 2. vertical computation" << endl;
    verticalComputation(imgSize.height - 1, -1, costMaps, rightFirst);

    // horizontal leftward
    cout << "[ScanlineOptimization] started 1. horizontal computation" << endl;
    horizontalComputation(0, 1, costMaps, rightFirst);

    // horizontal rightward
    cout << "[ScanlineOptimization] started 2. horizontal computation" << endl;
    horizontalComputation(imgSize.width - 1, -1, costMaps, rightFirst);

//    for(int d = 0; d < optimizedCosts.size(); d++)
//        optimizedCosts[d].copyTo(optCostMaps[d]);
}

void ScanlineOptimization::verticalComputation(int height, int direction, vector<Mat> *costMaps, bool rightFirst)
{
//    vector<Mat> tempCosts(dMax - dMin + 1);

//    // initialize vertical optimized costs
//    for(int disparity = 0; disparity < tempCosts.size(); disparity++)
//    {
//        tempCosts[disparity].create(imgSize, CV_32F);
//        for(int width = 0; width < imgSize.width; width++)
//        {
//            tempCosts[disparity].at<float>(height, width) = costMaps[imageNo][disparity].at<float>(height, width);
//        }
//    }

    // computes vertical optimized costs
    size_t height1;
//    #pragma omp parallel default(shared) private(height1) num_threads(omp_get_max_threads())
//    #pragma omp for schedule(static)
    for(height1 = height + direction; 0 <= height1 && height1 < imgSize.height; height1 += direction)
    {
        verticalOptimization(height1, height1 - direction, costMaps, rightFirst);
    }

//    // add so cost into optimizedCosts
//    for(int disparity = 0; disparity < tempCosts.size(); disparity++)
//    {
//        for(int w = 0; w < imgSize.width; w++)
//        {
//            for(int h = 0; h < imgSize.height; h++)
//            {
//                optimizedCosts[disparity].at<float>(h, w) += tempCosts[disparity].at<float>(h, w);
//            }
//        }
//    }
}

void ScanlineOptimization::verticalOptimization(int height1, int height2, vector<Mat> *costMaps, bool rightFirst)
{
    for(size_t width = 0; width < imgSize.width; width++)
    {
        partialOptimization(height1, height2, width, width, costMaps, rightFirst);
    }
}

void ScanlineOptimization::horizontalComputation(int width, int direction, vector<Mat> *costMaps, bool rightFirst)
{
//    vector<Mat> tempCosts(dMax - dMin + 1);

//    // initialize horizontal optimized costs
//    for(int disparity = 0; disparity < tempCosts.size(); disparity++)
//    {
//        tempCosts[disparity].create(imgSize, CV_32F);
//        for(size_t height = 0; height < imgSize.height; height++)
//        {
//            tempCosts[disparity].at<float>(height, width) = costMaps[imageNo][disparity].at<float>(height, width);
//        }
//    }

    // computes horizontal optimized costs
    size_t width1;
//    #pragma omp parallel default(shared) private(width1) num_threads(omp_get_max_threads())
//    #pragma omp for schedule(static)
    for(width1 = width + direction; 0 <= width1 && width1 < imgSize.width; width1 += direction)
    {
        horizontalOptimization(width1, width1 - direction, costMaps, rightFirst);
    }

//    // add so cost into optimizedCosts
//    for(int disparity = 0; disparity < tempCosts.size(); disparity++)
//    {
//        for(size_t w = 0; w < imgSize.width; w++)
//        {
//            for(size_t h = 0; h < imgSize.height; h++)
//            {
//                optimizedCosts[disparity].at<float>(h, w) += tempCosts[disparity].at<float>(h, w);
//            }
//        }
//    }
}

void ScanlineOptimization::horizontalOptimization(int width1, int width2, vector<Mat> *costMaps, bool rightFirst)
{
    for(size_t height = 0; height < imgSize.height; height++)
    {
        partialOptimization(height, height, width1, width2, costMaps, rightFirst);
    }
}

void ScanlineOptimization::partialOptimization(int height1, int height2, int width1, int width2, vector<Mat> *costMaps, bool rightFirst)
{
    float minOptCost = costMaps->at(0).at<costType>(height2, width2) / (float)COST_FACTOR;

    // find minimal previous optimized cost for a given column index
    for(int disparity = 1; disparity <= dMax-dMin; ++disparity)
    {
        float tmpCost = costMaps->at(disparity).at<costType>(height2, width2) / (float)COST_FACTOR;
        if(minOptCost > tmpCost)
            minOptCost = tmpCost;
    }

    float minkCr = minOptCost;

    for(int disparity = 0; disparity <= dMax-dMin; ++disparity)
    {
        // C1(p,d) - min_k(Cr(p-r,k))
        float cost = costMaps->at(disparity).at<costType>(height1, width1) / (float)COST_FACTOR - minkCr;

        // compute P1 and P2 parameters for better scanline optimization
        float p1, p2;
        computeP1P2(height1, height2, width1, width2, disparity + dMin, p1, p2, rightFirst);

        // compute min(Cr(p-r,d), Cr(p-r, d+-1) + P1, min_k(Cr(p-r,k)+P2))
        minOptCost = minkCr + p2;

        float tmpCost = costMaps->at(disparity).at<costType>(height2, width2) / (float)COST_FACTOR;
        if(minOptCost > tmpCost)
            minOptCost = tmpCost;

        if(disparity != 0)
        {
            tmpCost = costMaps->at(disparity - 1).at<costType>(height2, width2) / (float)COST_FACTOR + p1;
            if(minOptCost > tmpCost)
                minOptCost = tmpCost;
        }

        if(disparity != dMax - dMin)
        {
            tmpCost = costMaps->at(disparity + 1).at<costType>(height2, width2) / (float)COST_FACTOR + p1;
            if(minOptCost > tmpCost)
                minOptCost = tmpCost;
        }

        costMaps->at(disparity).at<costType>(height1, width1) = (costType)(((cost + minOptCost)) / 2 * COST_FACTOR);
    }
}

void ScanlineOptimization::computeP1P2(int height1, int height2, int width1, int width2, int disparity, float &p1, float &p2, bool rightFirst)
{
    int imageNo = 0;
    int otherImgNo = 1;

    if(rightFirst)
    {
        imageNo = 1;
        otherImgNo = 0;
        disparity = -disparity;
    }

    // compute color differences between pixels in the two pictures
    int d1 = colorDiff(images[imageNo].at<Vec3b>(height1, width1), images[imageNo].at<Vec3b>(height2, width2));
    int d2 = colorDifference + 1;

    if(0 <= width1 + disparity && width1 + disparity < imgSize.width
       && 0 <= width2 + disparity && width2 + disparity < imgSize.width)
    {
        d2 = colorDiff(images[otherImgNo].at<Vec3b>(height1, width1 + disparity),
                       images[otherImgNo].at<Vec3b>(height2, width2 + disparity));
    }

    // depending on the color differences computed previously, find the so parameters
    if(d1 < colorDifference)
    {
        if(d2 < colorDifference)
        {
            p1 = pi1;
            p2 = pi2;
        }
        else
        {
            p1 = pi1 / 4.0;
            p2 = pi2 / 4.0;
        }
    }
    else
    {
        if(d2 < colorDifference)
        {
            p1 = pi1 / 4.0;
            p2 = pi2 / 4.0;
        }
        else
        {
            p1 = pi1 / 10.0;
            p2 = pi2 / 10.0;
        }
    }
}

int ScanlineOptimization::colorDiff(const Vec3b &p1, const Vec3b &p2) const
{
    int colorDiff, diff = 0;

    for(uchar color = 0; color < 3; color++)
    {
        colorDiff = std::abs(p1[color] - p2[color]);
        diff = (diff > colorDiff)? diff: colorDiff;
    }

    return diff;
}
