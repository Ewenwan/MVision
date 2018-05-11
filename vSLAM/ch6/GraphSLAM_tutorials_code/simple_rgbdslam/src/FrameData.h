#ifndef FRAMEDATA_H
#define FRAMEDATA_H

#include <iostream>
#include <string>
#include <fstream>

#include <pcl/io/pcd_io.h>

#include <opencv2/opencv.hpp>
struct FrameRGBD
{
    int frameID;
    cv::Mat rgbimg;
    pcl::PointCloud< pcl::PointXYZRGBA>::Ptr pointcloudPtr;
};

class FrameData
{
private:
    int _frameID;
    float factor;
    float cx;
    float cy;
    float fx;
    float fy;
    std::ifstream _rgbfile, _dptfile;
    std::string _rgbline, _dptline;
    std::string _rootpath,_framergbpath,_framedptpath;

public :
    std::string subFileName(const char *line);
    void showMsg();
    bool LoadConfig(const char *filepath);

    bool nextFrame(FrameRGBD &temp);
    FrameData(const char *rp);
    //~FrameData();
};

#endif

