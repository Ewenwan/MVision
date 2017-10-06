#ifndef _EXTRA_H_
#define _EXTRA_H_
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>


using namespace cv;

void decomposeEssentialMat( InputArray _E, OutputArray _R1, OutputArray _R2, OutputArray _t );

int recoverPose( InputArray E, InputArray _points1, InputArray _points2, OutputArray _R,
                     OutputArray _t, double focal, Point2d pp=Point2d(0, 0), InputOutputArray _mask=noArray());

cv::Mat findEssentialMat( InputArray _points1, InputArray _points2, double focal, Point2d pp);

#endif