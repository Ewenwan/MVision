/*
自适应窗口代价聚合


*/
#ifndef AGGREGATION_H
#define AGGREGATION_H
#include <opencv2/opencv.hpp>
#include <omp.h>
#include "common.h"

using namespace cv;
using namespace std;

class Aggregation
{
public:
    Aggregation(const Mat &leftImage, const Mat &rightImage, uint colorThreshold1, uint colorThreshold2,
                uint maxLength1, uint maxLength2);
    void aggregation2D(Mat &costMap, bool horizontalFirst, uchar imageNo);
    void getLimits(vector<Mat> &upLimits, vector<Mat> &downLimits, vector<Mat> &leftLimits, vector<Mat> &rightLimits) const;
private:
    Mat images[2];
    Size imgSize;
    uint colorThreshold1, colorThreshold2;
    uint maxLength1, maxLength2;
    vector<Mat> upLimits;
    vector<Mat> downLimits;
    vector<Mat> leftLimits;
    vector<Mat> rightLimits;

    int colorDiff(const Vec3b &p1, const Vec3b &p2);
    int computeLimit(int height, int width, int directionH, int directionW, uchar imageNo);
    Mat computeLimits(int directionH, int directionW, int imageNo);

    Mat aggregation1D(const Mat &costMap, int directionH, int directionW, Mat &windowSizes, uchar imageNo);
};

#endif // AGGREGATION_H
