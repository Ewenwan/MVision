#include "FrameData.h"
#include <pcl/visualization/cloud_viewer.h>

using namespace std;
using namespace cv;
using namespace pcl;

const char *rgbtxt = "rgb.txt";
const char *dpttxt = "depth.txt";

bool FrameData::LoadConfig(const char *filepath)
{
    return true;
}
FrameData::FrameData(const char *rp)
{
    _frameID = 0;
    factor = 5000.0;
    cx = 319.0;
    cy = 239.0;
    fx = 525.0;
    fy = 525.0;

    _rootpath = rp;

    _framergbpath = _rootpath + rgbtxt;
    _framedptpath = _rootpath + dpttxt;
    cout<< "rgb.txt & depth.txt path: "<<endl;
    cout<<_framergbpath <<" "<<_framedptpath<<endl;

    _rgbfile.open(_framergbpath.c_str());
    _dptfile.open(_framedptpath.c_str());

    //_rgbfile.open("/home/hyj/rgbddata/rgb.txt");
    //_dptfile.open("/home/hyj/rgbddata/depth.txt");

    if(!_rgbfile.is_open() && !_dptfile.is_open())
    {
        cerr<<"rgb.txt & depth.txt file is not exit"<<endl;
    }
    else
    {
        // remove the comment line in txt file
        for(int i =0;i<3;i++)
        {
            getline(_rgbfile,_rgbline);
            getline(_dptfile,_dptline);
        }
    }

}

string FrameData::subFileName(const char* line)
{
    const char* spacekey = strchr(line,' ');
    int index = (int)(spacekey - line);

    string l_str = line;
    string result = l_str.substr(index+1);
    return result;

}

bool FrameData::nextFrame(FrameRGBD &temp)
{
    PointCloud<PointXYZRGBA>::Ptr cloudPtr(new PointCloud<PointXYZRGBA>);
    
    if(!_rgbfile.eof() && !_dptfile.eof())
    {
        getline(_rgbfile,_rgbline);
        string dat = subFileName(_rgbline.c_str());
        cout << "path: " << _rootpath+dat<<endl;

        temp.rgbimg = imread(_rootpath+dat);
        
        getline(_dptfile,_dptline);
        dat = subFileName(_dptline.c_str());
        cv::Mat current_dpt;
        current_dpt = imread(_rootpath+dat,-1);

        cloudPtr->width = current_dpt.cols;
        cloudPtr->height = current_dpt.rows;
        cloudPtr->is_dense = false;
        cloudPtr->points.resize(cloudPtr->width * cloudPtr->height);
       
        for(int v = 0; v < current_dpt.rows; v++)
        {
            const uchar *rgbptr = temp.rgbimg.ptr<uchar>(v);
            for(int u = 0; u< current_dpt.cols; u++)
            {
                PointXYZRGBA xyzrgb;

                // rgb
                const uchar *pixel = rgbptr;
                xyzrgb.b = pixel[0];
                xyzrgb.g = pixel[1];
                xyzrgb.r = pixel[2];
                rgbptr += 3;

                // depth
                float Z = current_dpt.ptr<ushort>(v)[u]/factor;
                float X = (u - cx)*Z/fx;
                float Y = (v - cy)*Z/fy;
                
                if(Z == 0.0)
                {
                    xyzrgb.x = xyzrgb.y = xyzrgb.z = numeric_limits<float>::quiet_NaN();
                    
                }
                else
                {
                  //   cout<<"x:"<< X <<" "<<" y: "<<" "<<Y<<" "<<"z:"<<Z<<endl;
                    xyzrgb.z = Z;
                    xyzrgb.x = X;
                    xyzrgb.y = Y;

                }
                cloudPtr->at(u,v) = xyzrgb;
               // cout<< cloudPtr->at(u,v)<<endl;

            }
        }
    }
    else
    {
        return false;
    }
    _frameID += 1;
    temp.frameID = _frameID;
    temp.pointcloudPtr = cloudPtr;

    return true;
}
