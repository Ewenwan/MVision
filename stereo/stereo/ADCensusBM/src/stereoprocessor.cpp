/*
双目处理类　
1．ADcensus代价初始化　+　
2. 自适应窗口代价聚合　+　
3. 扫描线全局优化　+ 
4. 代价转视差, 外点(遮挡点+不稳定)检测  + 
5. 视差传播－迭代区域投票法 使用临近好的点为外点赋值　   + 
6. 视差传播-16方向极线插值（对于区域内点数量少的　外点　再优化） +
7. candy边缘矫正 + 
8. 亚像素求精,中值滤波 

        costInitialization();//　代价初始化　AD + census　/　默认　defaultBorderCost * COST_FACTOR
        costAggregation();  // 自适应窗口代价聚合 将中心点附近自适应不规则区域的　adCensus　求和
        scanlineOptimization();// 扫描线全局优化  动态规划的一种方法
// 由于代价聚合的结果不大靠谱，可以考虑将其视作数据项，
// 建立全局能量函数（公式如下所示），这样便直接过渡到了全局算法。
// https://www.cnblogs.com/DaD3zZ-Beyonder/p/5934092.html
        outlierElimination();// 代价转视差　外点(遮挡点+不稳定)检测 
        regionVoting();// 视差传播－迭代区域投票法（外点视差优化）
        properInterpolation();// 视差传播-16方向极线插值（对于区域内点数量少的　外点　再优化）
        discontinuityAdjustment();// candy边缘矫正
        subpixelEnhancement();// 亚像素求精 ＋　中值滤波 


*/
#include "stereoprocessor.h"
#include <limits>

#define DEBUG

StereoProcessor::StereoProcessor(uint dMin, uint dMax, Mat leftImage, 
				 Mat rightImage, Size censusWin, float defaultBorderCost,
                                 float lambdaAD, float lambdaCensus, string savePath, 
				 uint aggregatingIterations, uint colorThreshold1, 
				 uint colorThreshold2, uint maxLength1, uint maxLength2, 
				 uint colorDifference, float pi1, float pi2, uint dispTolerance, 					 uint votingThreshold, float votingRatioThreshold,
				 uint maxSearchDepth, uint blurKernelSize, uint cannyThreshold1, 					 uint cannyThreshold2, uint cannyKernelSize)
{
    this->dMin = dMin;//最小视差
    this->dMax = dMax;//最大视差
    this->images[0] = leftImage;//左图像
    this->images[1] = rightImage;//右图像
    this->censusWin = censusWin;//窗口
    this->defaultBorderCost = defaultBorderCost;//权重
    this->lambdaAD = lambdaAD;
    this->lambdaCensus = lambdaCensus;
    this->savePath = savePath;
    this->aggregatingIterations = aggregatingIterations;
    this->colorThreshold1 = colorThreshold1;
    this->colorThreshold2 = colorThreshold2;
    this->maxLength1 = maxLength1;
    this->maxLength2 = maxLength2;
    this->colorDifference = colorDifference;
    this->pi1 = pi1;
    this->pi2 = pi2;
    this->dispTolerance = dispTolerance;
    this->votingThreshold = votingThreshold;
    this->votingRatioThreshold = votingRatioThreshold;
    this->maxSearchDepth = maxSearchDepth;
    this->blurKernelSize = blurKernelSize;
    this->cannyThreshold1 = cannyThreshold1;
    this->cannyThreshold2 = cannyThreshold2;
    this->cannyKernelSize = cannyKernelSize;
    this->validParams = false;
    this->dispComputed = false;
}
//析构函数
StereoProcessor::~StereoProcessor()
{
    delete adCensus;//需要手动删除　指针类对象
    delete aggregation;
    delete dispRef;
}
//检测参数是否有错误
bool StereoProcessor::init(string &error)
{
    bool valid = true;
    valid &= dMin < dMax;
    valid &= images[0].size().height == images[1].size().height && images[0].size().width == images[1].size().width && images[0].size().height > 2 && images[0].size().width > 2;
    valid &= censusWin.height > 2 && censusWin.width > 2 && 
             censusWin.height % 2 != 0 && censusWin.width % 2 != 0;//奇数窗口
    //权重　0~1
    valid &= defaultBorderCost >= 0 && defaultBorderCost < 1 &&
             lambdaAD >= 0 && lambdaCensus >= 0 && aggregatingIterations > 0;
    valid &= colorThreshold1 > colorThreshold2 && colorThreshold2 > 0;// 阈值
    valid &= maxLength1 > maxLength2 && maxLength2 > 0;
    valid &= colorDifference > 0 && pi1 > 0 && pi2 > 0;
    valid &= votingThreshold > 0 && votingRatioThreshold > 0 && maxSearchDepth > 0;
    valid &= blurKernelSize > 2 && blurKernelSize % 2 != 0;
    valid &= cannyThreshold1 < cannyThreshold2;
    valid &= cannyKernelSize > 2 && cannyKernelSize % 2 != 0;
    if(!valid)
    {
        error = "StereoProcessor needs the following parameters: \n"
                "dMin(uint)                       [dMin(uint) < dMax(uint)]\n"
                "leftImage(string)                [path to image file]\n"
                "rightImage(string)               [path to image file]\n"
                "censusWinH(uint)                 [censusWinH > 2 and odd]\n"
                "censusWinW(uint)                 [censusWinW > 2 and odd]\n"
                "defaultBorderCost(float)        [0 <= defaultBorderCost < 1]\n"
                "lambdaAD(float)                 [lambdaAD >= 0]\n"
                "lambdaCensus(float)             [lambdaCensus >= 0]\n"
                "savePath(string)                 [:-)]\n"
                "aggregatingIterations(uint)      [aggregatingIterations > 0]\n"
                "colorThreshold1(uint)            [colorThreshold1 > colorThreshold2]\n"
                "colorThreshold2(uint)            [colorThreshold2 > 0]\n"
                "maxLength1(uint)                 [maxLength1 > 0]\n"
                "maxLength2(uint)                 [maxLength2 > 0]\n"
                "colorDifference(uint)            [colorDifference > 0]\n"
                "pi1(float)                      [pi1 > 0]\n"
                "pi2(float)                      [pi2 > 0]\n"
                "dispTolerance(uint)              [;-)]\n"
                "votingThreshold(uint)            [votingThreshold > 0]\n"
                "votingRatioThreshold(float)     [votingRatioThreshold > 0]\n"
                "maxSearchDepth(uint)             [maxSearchDepth > 0]\n"
                "blurKernelSize(uint)             [blurKernelSize > 2 and odd]\n"
                "cannyThreshold1(uint)            [cannyThreshold1 < cannyThreshold2]\n"
                "cannyThreshold2(uint)            [:-)]\n"
                "cannyKernelSize(uint)            [cannyKernelSize > 2 and odd]\n";
    }
    else
    {
        error = "";

        this->imgSize = images[0].size();//图像大小
        costMaps.resize(2);
        for (size_t i = 0; i < 2; i++)
        {
            costMaps[i].resize(abs(dMax - dMin) + 1);//视差　map
            for(size_t j = 0; j < costMaps[i].size(); j++)
            {
                costMaps[i][j].create(imgSize, COST_MAP_TYPE);
            }
        }
// 代价计算　cost = r(Cad , lamd1) + r(Cces, lamd2)
        adCensus = new ADCensusCV(images[0], images[1], censusWin, lambdaAD, lambdaCensus);
// 自适应窗口代价聚合 搜索　自适应框时的颜色阈值  搜索范围阈值
        aggregation = new Aggregation(images[0], images[1], colorThreshold1, colorThreshold2, maxLength1, maxLength2);
// 视差优化 输入左右视差图----> 外点(遮挡点+不稳定)检测 ------> 视差传播－迭代区域投票法------>
// 视差传播-16方向极线插值算法 -----> candy边缘矫正 ------> 亚像素求精 ------->  中值滤波 
        dispRef = new DisparityRefinement(dispTolerance, dMin, dMax, votingThreshold,
					  votingRatioThreshold, maxSearchDepth, blurKernelSize,
					  cannyThreshold1, cannyThreshold2, cannyKernelSize);
    }

    validParams = valid;
    return valid;
}

bool StereoProcessor::compute()
{
    if(validParams)
    {
        costInitialization();//　代价初始化　AD + census　/　默认　defaultBorderCost * COST_FACTOR
        costAggregation();  // 自适应窗口代价聚合 将中心点附近自适应不规则区域的　adCensus　求和
    //    scanlineOptimization();// 扫描线优化 
//由于代价聚合的结果不大靠谱，可以考虑将其视作数据项，
//建立全局能量函数（公式如下所示），这样便直接过渡到了全局算法。
        outlierElimination();// 代价转视差　外点(遮挡点+不稳定)检测 
        regionVoting();// 视差传播－迭代区域投票法
        properInterpolation();// 视差传播-16方向极线插值（对于区域内点数量少的　外点　再优化）
        discontinuityAdjustment();// candy边缘矫正
        subpixelEnhancement();// 亚像素求精 ＋　中值滤波 
        dispComputed = true;
    }

    return validParams;
}

Mat StereoProcessor::getDisparity()
{
    return (dispComputed)? floatDisparityMap: Mat();
}

// =============================================================
//　代价初始化　计算左右两幅图像的　代价　不同的视差搜索区域　同一行上 超过范围使用默认　否则使用adCensus
// =============================================================
void StereoProcessor::costInitialization()
{
// census窗口 一半
    Size halfCensusWin(censusWin.width / 2, censusWin.height / 2);
    bool out;
    int d, h, w, wL, wR;
    size_t imageNo;

    cout << "[StereoProcessor] started cost initialization!" << endl;


    for(imageNo = 0; imageNo < 2; ++imageNo)
    {
        #pragma omp parallel default (shared) private(d, h, w, wL, wR, out) num_threads(omp_get_max_threads())
        #pragma omp for schedule(static)
        for(d = 0; d <= dMax - dMin; d++)//最大最小视差范围内搜索
        {
            for(h = 0; h < imgSize.height; h++)//每一行
            {
                for(w = 0; w < imgSize.width; w++)//每一列
                {
                    wL = w;
                    wR = w;

                    if(imageNo == 0)//左图像
                        wR = w - d;
                    else
                        wL = w + d;

                    out = wL - halfCensusWin.width < 0 || 
 			  wL + halfCensusWin.width >= imgSize.width || 
			  wR - halfCensusWin.width < 0 || 
			  wR + halfCensusWin.width >= imgSize.width || 
			  h - halfCensusWin.height < 0 || 
			  h + halfCensusWin.height >= imgSize.height;
//计算左右两幅图像的　代价　不同的视差搜索区域　同一行上 超过范围使用默认　否则使用adCensus
	costMaps[imageNo][d].at<costType>(h, w) = out ? defaultBorderCost * COST_FACTOR : 
					adCensus->adCensus(wL, h, wR, h) / 2 * COST_FACTOR;
                }
            }
            cout << "[StereoProcessor] created disparity no. " << d << " for image no. " << imageNo << endl;
        }
        cout << "[StereoProcessor] created cost maps for image no. " << imageNo << endl;
    }

#ifdef DEBUG
    Mat disp = cost2disparity(0);
    saveDisparity<int>(disp, "01_dispLR.png");

    disp = cost2disparity(1);
    saveDisparity<int>(disp, "01_dispRL.png");
#endif

    cout << "[StereoProcessor] finished cost initialization!" << endl;

}

// =============================================================
// 自适应窗口代价聚合 将中心点附近自适应不规则区域的　adCensus　求和
// =============================================================
void StereoProcessor::costAggregation()
{
    size_t imageNo, i;
    int d;

    cout << "[StereoProcessor] started cost aggregation!" << endl;

    for(imageNo = 0; imageNo < 2; ++imageNo)//左右两张图
    {
#if COST_TYPE_NOT_FLOAT
        #pragma omp parallel default (shared) private(d, i) num_threads(omp_get_max_threads())
        #pragma omp for schedule(static)
        for(d = 0; d <= dMax - dMin; d++)//最大最小视差范围内搜索
        {
            Mat currCostMap(imgSize, CV_32F);

            for(int h = 0; h < imgSize.height; h++)//每一行
            {
                for(int w = 0; w < imgSize.width; w++)//每一列
                {
                    currCostMap.at<float>(h, w) = (float)costMaps[imageNo][d].at<costType>(h, w) / COST_FACTOR;
                }
            }

            bool horizontalFirst = true;//先水平
            for(i = 0; i < aggregatingIterations; i++)//聚合次数
            {
                aggregation->aggregation2D(currCostMap, horizontalFirst, imageNo);
                horizontalFirst = !horizontalFirst;//在垂直
                cout << "[StereoProcessor] aggregation iteration no. " << i << 
		        ", disparity no. " << d << " for image no. " << imageNo << endl;
            }

            for(int h = 0; h < imgSize.height; h++)
            {
                for(int w = 0; w < imgSize.width; w++)
                {
                    costMaps[imageNo][d].at<costType>(h, w) = (costType)(currCostMap.at<float>(h, w) * COST_FACTOR);
                }
            }
        }
#else
        #pragma omp parallel default (shared) private(d, i) num_threads(omp_get_max_threads())
        #pragma omp for schedule(static)
        for(d = 0; d <= dMax - dMin; d++)
        {
            bool horizontalFirst = true;
            for(i = 0; i < aggregatingIterations; i++)
            {
                aggregation->aggregation2D(costMaps[imageNo][d], horizontalFirst, imageNo);
                horizontalFirst = !horizontalFirst;
                cout << "[StereoProcessor] aggregation iteration no. " << i << 
			", disparity no. " << d << " for image no. " << imageNo << endl;
            }
        }
#endif
    }

#ifdef DEBUG
    Mat disp = cost2disparity(0);
    saveDisparity<int>(disp, "02_dispLR_agg.png");

    disp = cost2disparity(1);
    saveDisparity<int>(disp, "02_dispRL_agg.png");
#endif

    cout << "[StereoProcessor] finished cost aggregation!" << endl;
}
// =============================================================
// 扫描线优化 
// =============================================================
void StereoProcessor::scanlineOptimization()
{
    ScanlineOptimization sO(images[0], images[1], dMin, dMax, colorDifference, pi1, pi2);
    int imageNo;

    cout << "[StereoProcessor] started scanline optimization!" << endl;

    #pragma omp parallel default (shared) private(imageNo) num_threads(omp_get_max_threads())
    #pragma omp for schedule(static)
    for(imageNo = 0; imageNo < 2; ++imageNo)
    {
        sO.optimization(&costMaps[imageNo], (imageNo == 1));
    }

#ifdef DEBUG
    Mat disp = cost2disparity(0);
    saveDisparity<int>(disp, "03_dispLR_so.png");

    disp = cost2disparity(1);
    saveDisparity<int>(disp, "03_dispRL_so.png");
#endif

    cout << "[StereoProcessor] finished scanline optimization!" << endl;
}
// =============================================================
// 代价转视差　外点(遮挡点+不稳定)检测 
// =============================================================
void StereoProcessor::outlierElimination()
{
    cout << "[StereoProcessor] started outlier elimination!" << endl;

    Mat disp0 = cost2disparity(0);//代价转到视差图
    Mat disp1 = cost2disparity(1);

    disparityMap = dispRef->outlierElimination(disp0, disp1);//找出外点(遮挡点+ 不稳定)

#ifdef DEBUG
    saveDisparity<int>(disparityMap, "04_dispBoth_oe.png");
#endif

    cout << "[StereoProcessor] finished outlier elimination!" << endl;
}
// =============================================================
// 视差传播－迭代区域投票法 使用临近好的点　为　外点　赋值　
// =============================================================
void StereoProcessor::regionVoting()
{
    vector<Mat> upLimits, downLimits, leftLimits, rightLimits;

    cout << "[StereoProcessor] started region voting!" << endl;

    aggregation->getLimits(upLimits, downLimits, leftLimits, rightLimits);

    bool horizontalFirst = false;
    for(int i = 0; i < 5; i++)//5次迭代优化
    {
        cout << "[StereoProcessor] region voting iteration no. " << i << endl;
        dispRef->regionVoting(disparityMap, upLimits, downLimits, leftLimits, rightLimits, horizontalFirst);
        horizontalFirst = ~horizontalFirst;
    }

#ifdef DEBUG
    saveDisparity<int>(disparityMap, "05_dispBoth_rv.png");
#endif

    cout << "[StereoProcessor] finished region voting!" << endl;
}
// =============================================================
// 有些outlier由于区域内稳定点个数不满足公式，这样的区域用此方法是处理不来的，只能进一步通过16方向极线插值来进一步填充
// 视差传播-16方向极线插值（对于区域内点数量少的　外点　再优化）
// =======================================================================
void StereoProcessor::properInterpolation()
{
    cout << "[StereoProcessor] started proper interpolation!" << endl;
    dispRef->properInterpolation(disparityMap, images[0]);

#ifdef DEBUG
    saveDisparity<int>(disparityMap, "06_dispBoth_pi.png");
#endif

    cout << "[StereoProcessor] finished proper interpolation!" << endl;
}
// =============================================================
// candy边缘矫正
// =============================================================
void StereoProcessor::discontinuityAdjustment()
{
    cout << "[StereoProcessor] started discontinuity adjustment!" << endl;
    dispRef->discontinuityAdjustment(disparityMap, costMaps);

#ifdef DEBUG
    saveDisparity<int>(disparityMap, "07_dispBoth_da.png");
#endif

    cout << "[StereoProcessor] finished discontinuity adjustment!" << endl;
}
// =============================================================
// 亚像素求精 ＋　中值滤波 
// =============================================================
void StereoProcessor::subpixelEnhancement()
{
    cout << "[StereoProcessor] started subpixel enhancement!" << endl;
    floatDisparityMap = dispRef->subpixelEnhancement(disparityMap, costMaps);

#ifdef DEBUG
    saveDisparity<float>(floatDisparityMap, "08_dispBoth_se.png");
#endif

    cout << "[StereoProcessor] finished subpixel enhancement!" << endl;
}

// 代价转成　视差（代价小的对应点的坐标差值为视差）
Mat StereoProcessor::cost2disparity(int imageNo)
{
    Mat disp(imgSize, CV_32S);
    Mat lowCost(imgSize, COST_MAP_TYPE, Scalar(std::numeric_limits<costType>::max()));

    for(int d = 0; d <= dMax - dMin; d++)
    {
        for(size_t h = 0; h < imgSize.height; h++)
        {
            for(size_t w = 0; w < imgSize.width; w++)
            {
                if (lowCost.at<costType>(h, w) > costMaps[imageNo][d].at<costType>(h, w))//代价较小
                {
                    lowCost.at<costType>(h, w) = costMaps[imageNo][d].at<costType>(h, w);
                    disp.at<int>(h, w) = d + dMin;//保存该代价对应的视差
                }
            }
        }
    }

    return disp;
}
//　保存视差
template <typename T>
void StereoProcessor::saveDisparity(const Mat &disp, string filename, bool stretch)
{
    Mat output(imgSize, CV_8UC3);
    String path(savePath);
    T min, max;

    if(stretch)
    {
        min = std::numeric_limits<T>::max();
        max = 0;
        for(size_t h = 0; h < imgSize.height; h++)
        {
            for(size_t w = 0; w < imgSize.width; w++)
            {
                T disparity = disp.at<T>(h, w);

                if(disparity < min && disparity >= 0)
                    min = disparity;
                else if(disparity > max)
                    max = disparity;
            }
        }
    }

    for(size_t h = 0; h < imgSize.height; h++)
    {
        for(size_t w = 0; w < imgSize.width; w++)
        {
            Vec3b color;
            T disparity = disp.at<T>(h, w);

            if (disparity >= dMin)
            {
                if(stretch)
                    disparity = (255 / (max - min)) * (disparity - min);

                color[0] = (uchar)disparity;
                color[1] = (uchar)disparity;
                color[2] = (uchar)disparity;

            }
            else if(disparity == dMin - DisparityRefinement::DISP_OCCLUSION)
            {
                color[0] = 255;
                color[1] = 0;
                color[2] = 0;
            }
            else
            {
                color[0] = 0;
                color[1] = 0;
                color[2] = 255;
            }

            output.at<Vec3b>(h, w) = color;
        }
    }

    path += filename;
    imwrite(path, output);
}
