/*
 显示 包围盒
 显示 状态  
 打印 状态  
 返回 二维像素坐标点
*/
#ifndef UTILS_H
#define UTILS_H

#include <opencv2/core.hpp>
#include <vector>
#include "stats.h"

using namespace std;
using namespace cv;
// 显示 包围盒
void drawBoundingBox(Mat image, vector<Point2f> bb);
// 显示状态 
void drawStatistics(Mat image, const Stats& stats);
// 打印状态
void printStatistics(string name, Stats stats);
// 返回 二维像素坐标点
vector<Point2f> Points(vector<KeyPoint> keypoints);

// 显示 包围盒
void drawBoundingBox(Mat image, vector<Point2f> bb)
{
    for(unsigned i = 0; i < bb.size() - 1; i++) {
        line(image, bb[i], bb[i + 1], Scalar(0, 0, 255), 2);
    }
    line(image, bb[bb.size() - 1], bb[0], Scalar(0, 0, 255), 2);
}
// 显示状态 
void drawStatistics(Mat image, const Stats& stats)
{
    static const int font = FONT_HERSHEY_PLAIN;
    stringstream str1, str2, str3;

    str1 << "Matches: " << stats.matches;// 匹配点 数量
    str2 << "Inliers: " << stats.inliers;// 内点数量
    str3 << "Inlier ratio: " << setprecision(2) << stats.ratio;//比率

    putText(image, str1.str(), Point(0, image.rows - 90), font, 2, Scalar::all(255), 3);
    putText(image, str2.str(), Point(0, image.rows - 60), font, 2, Scalar::all(255), 3);
    putText(image, str3.str(), Point(0, image.rows - 30), font, 2, Scalar::all(255), 3);
}
//打印状态
void printStatistics(string name, Stats stats)
{
    cout << name << endl;
    cout << "----------" << endl;

    cout << "Matches " << stats.matches << endl;
    cout << "Inliers " << stats.inliers << endl;
    cout << "Inlier ratio " << setprecision(2) << stats.ratio << endl;
    cout << "Keypoints " << stats.keypoints << endl;
    cout << endl;
}
// 返回 二维像素坐标点
vector<Point2f> Points(vector<KeyPoint> keypoints)
{
    vector<Point2f> res;
    for(unsigned i = 0; i < keypoints.size(); i++) {
        res.push_back(keypoints[i].pt);
    }
    return res;
}

#endif // UTILS_H
