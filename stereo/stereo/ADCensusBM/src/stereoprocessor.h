#ifndef STEREOPROCESSOR_H
#define STEREOPROCESSOR_H
#include "adcensuscv.h"
#include "aggregation.h"
#include "scanlineoptimization.h"
#include "disparityrefinement.h"
#include <omp.h>
#include "common.h"

using namespace std;

class StereoProcessor
{
public:
    StereoProcessor(uint dMin, uint dMax, Mat leftImage, Mat rightImage, Size censusWin, float defaultBorderCost,
                    float lambdaAD, float lambdaCensus, string savePath, uint aggregatingIterations,
                    uint colorThreshold1, uint colorThreshold2, uint maxLength1, uint maxLength2, uint colorDifference,
                    float pi1, float pi2, uint dispTolerance, uint votingThreshold, float votingRatioThreshold,
                    uint maxSearchDepth, uint blurKernelSize, uint cannyThreshold1, uint cannyThreshold2, uint cannyKernelSize);
    ~StereoProcessor();//析构函数
    bool init(string &error);
    bool compute();//计算
    Mat getDisparity();//得到视差图

private:
    int dMin;
    int dMax;
    Mat images[2];
    Size censusWin;
    float defaultBorderCost;
    float lambdaAD;
    float lambdaCensus;
    string savePath;
    uint aggregatingIterations;
    uint colorThreshold1;
    uint colorThreshold2;
    uint maxLength1;
    uint maxLength2;
    uint colorDifference;
    float pi1;
    float pi2;
    uint dispTolerance;
    uint votingThreshold;
    float votingRatioThreshold;
    uint maxSearchDepth;
    uint blurKernelSize;
    uint cannyThreshold1;
    uint cannyThreshold2;
    uint cannyKernelSize;
    bool validParams, dispComputed;

    vector<vector<Mat> > costMaps;
    Size imgSize;
    ADCensusCV *adCensus;//需要手动删除　指针类对象
    Aggregation *aggregation;
    Mat disparityMap, floatDisparityMap;
    DisparityRefinement *dispRef;

    void costInitialization();
    void costAggregation();
    void scanlineOptimization();
    void outlierElimination();
    void regionVoting();
    void properInterpolation();
    void discontinuityAdjustment();
    void subpixelEnhancement();

    Mat cost2disparity(int imageNo);

    template <typename T>
    void saveDisparity(const Mat &disp, string filename, bool stretch = true);
};

#endif // STEREOPROCESSOR_H
