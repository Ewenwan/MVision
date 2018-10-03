#ifndef PERCIPIO_SAMPLE_COMMON_RGBPOINT_CLOUD_VIEWER_HPP_
#define PERCIPIO_SAMPLE_COMMON_RGBPOINT_CLOUD_VIEWER_HPP_

#include <opencv2/opencv.hpp>
#include <string>

class RGBPointCloudViewerImpl;

class RGBPointCloudViewer {
public:
    RGBPointCloudViewer();
    ~RGBPointCloudViewer();

    void show(const cv::Mat &pointCloud, const cv::Mat &color, const std::string &windowName);
    bool isStopped(const std::string &windowName);

private:
    RGBPointCloudViewerImpl* impl;// 具体实现类
};

#endif
