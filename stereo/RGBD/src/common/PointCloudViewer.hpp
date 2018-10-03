#ifndef PERCIPIO_SAMPLE_COMMON_POINT_CLOUD_VIEWER_HPP_
#define PERCIPIO_SAMPLE_COMMON_POINT_CLOUD_VIEWER_HPP_

#include <opencv2/opencv.hpp>
#include <string>

class PointCloudViewerImpl;

class PointCloudViewer {
public:
    PointCloudViewer();
    ~PointCloudViewer();

    void show(const cv::Mat &pointCloud, const std::string &windowName);
    bool isStopped(const std::string &windowName);

private:
    PointCloudViewerImpl* impl;
};


enum{
    PC_FILE_FORMAT_XYZ = 0,
};
void writePointCloud(const cv::Point3f* pnts, size_t n, const char* file, int format);



#endif
