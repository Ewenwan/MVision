#ifndef SCANLINEOPTIMIZATION_H
#define SCANLINEOPTIMIZATION_H

#include <opencv2/opencv.hpp>
#include <omp.h>
#include "common.h"

using namespace cv;
using namespace std;

class ScanlineOptimization
{
public:
    ScanlineOptimization(const Mat &leftImage, const Mat &rightImage, int dMin, int dMax,
                         uint colorDifference, float pi1, float pi2);
    void optimization(vector<Mat> *costMaps, bool rightFirst);
private:
    Mat images[2];
    Size imgSize;
    int dMin;
    int dMax;
    uint colorDifference;
    float pi1;
    float pi2;

    void verticalComputation(int height, int direction, vector<Mat> *costMaps, bool rightFirst);
    void verticalOptimization(int height1, int height2, vector<Mat> *costMaps, bool rightFirst);

    void horizontalComputation(int width, int direction, vector<Mat> *costMaps, bool rightFirst);
    void horizontalOptimization(int width1, int width2, vector<Mat> *costMaps, bool rightFirst);

    void partialOptimization(int height1, int height2, int width1, int width2, vector<Mat> *costMaps, bool rightFirst);

    void computeP1P2(int height1, int height2, int width1, int width2, int disparity, float &p1, float &p2, bool rightFirst);

    int colorDiff(const Vec3b &p1, const Vec3b &p2) const;
};

#endif // SCANLINEOPTIMIZATION_H
