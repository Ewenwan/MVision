#include <stdexcept>
#include "RGBPointCloudViewer.hpp"
#include <stdio.h>

#ifdef HAVE_PCL
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/cloud_viewer.h>

static const std::string helpText[] = {
    "Left Button + Slide  Left/Right",
    "Left Button + Slide  Up/Down",
    "Left Button + CTRL + Slide Left/Right",
    "Left Button + SHIFT",
    "Mouse Wheel Up/Down",
    "Mouse Wheel PressDown"
};
static int helpTextSize = 11;
static int WinWidth = 640;
static int WinHeight = 480;
static void viewerOneOff(pcl::visualization::PCLVisualizer& viewer)
{
    viewer.setBackgroundColor(0.0, 0.0, 0.0);
    viewer.setCameraPosition(0, 0, 0, 0, 0, 1, 0, -1, 0);
    viewer.resetCamera();
    //viewer.addCoordinateSystem(1.0);
    viewer.setPosition(0, 0);
    viewer.setSize(WinWidth, WinHeight);

    for (size_t i=0; i<sizeof(helpText)/sizeof(helpText[0]); i++)
        viewer.addText(helpText[i], 3, WinHeight-(i+1)*helpTextSize, helpTextSize, 1.0, 1.0, 1.0);
}
#endif // HAVE_PCL


class RGBPointCloudViewerImpl {
public:
    void show(const cv::Mat &pointCloud, const cv::Mat &color,  const std::string &windowName)
    {
#ifdef HAVE_PCL
        if(!(pointCloud.type() == CV_32FC3)){
            throw std::runtime_error("pcshow: pointCloud should be (type=CV_32FC3)");
        }
        
        std::map<std::string, pcl::visualization::CloudViewer*>::iterator it = m_viewerMap.find(windowName);
        
        bool reset_view = false;
        if(m_viewerMap.end() == it){
            pcl::visualization::CloudViewer* viewer = new pcl::visualization::CloudViewer(windowName);
            std::pair<std::map<std::string, pcl::visualization::CloudViewer*>::iterator, bool> ret;
            ret = m_viewerMap.insert(std::pair<std::string, pcl::visualization::CloudViewer*>(windowName, viewer));
            if(!ret.second){
                // LOGE("pcshow: insert viewer %s failed.\n", windowName.c_str());
                return;
            }
            it = ret.first;
            reset_view = true;
        }
        
        // PCL display
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

        genPointCloudXYZRGBFromVec3f((float*)pointCloud.data, (unsigned char*)color.data,
                                     pointCloud.rows , pointCloud.cols, cloud);

        it->second->showCloud(cloud);
        if(reset_view){
            it->second->runOnVisualizationThreadOnce(viewerOneOff);
        }
#endif // HAVE_PCL
    }


    bool isStopped(const std::string &windowName)
    {
        bool ret = true;
#ifdef HAVE_PCL
        std::map<std::string, pcl::visualization::CloudViewer*>::iterator it = m_viewerMap.find(windowName);
        ret = it->second->wasStopped(0);
#endif // HAVE_PCL
        return ret;
    }


#ifdef HAVE_PCL
    void genPointCloudXYZRGBFromVec3f(const float* data, unsigned char* color, int rows, int cols, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud)
    {
        
        cloud->resize(rows*cols);
	cloud->width    =  cols;  
	cloud->height   =  rows;
        cloud->is_dense = false;// 非稠密点云，会有不好的点
        
        for(int i = 0; i < cloud->points.size(); i++)
         {
            //cloud->push_back(pcl::PointXYZRGB(data[i*3+0],  data[i*3+1],  data[i*3+2], 
            //                                  color[i*3+0], color[i*3+1], color[i*3+2] ));
	    cloud->points[i].x = data[i*3+0];
	    cloud->points[i].y = data[i*3+1];
	    cloud->points[i].z = data[i*3+2];
	    cloud->points[i].r = color[i*3+0];
	    cloud->points[i].g = color[i*3+1];
            cloud->points[i].b = color[i*3+2];
        }
    }

private:
    std::map<std::string, pcl::visualization::CloudViewer*> m_viewerMap;
#endif // HAVE_PCL

};

///////////////////////////////////////////////////////////////

RGBPointCloudViewer::RGBPointCloudViewer()
{
    impl = new RGBPointCloudViewerImpl;
}

RGBPointCloudViewer::~RGBPointCloudViewer()
{
    delete impl;
}

bool RGBPointCloudViewer::isStopped(const std::string &windowName)
{
    return impl->isStopped(windowName);
}

void RGBPointCloudViewer::show(const cv::Mat &pointCloud, const cv::Mat &color, const std::string &windowName)
{
    impl->show(pointCloud, color, windowName);
}


