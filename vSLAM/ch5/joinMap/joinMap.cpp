#include <iostream>
#include <fstream>
using namespace std;
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Geometry>    // 变换矩阵　Ｔ
#include <boost/format.hpp>  // 格式化字符串 for formating strings 处理图像文件格式
//点云数据处理
#include <pcl/point_types.h> 
#include <pcl/io/pcd_io.h> 
#include <pcl/visualization/pcl_visualizer.h>
/*
 现实世界物体坐标　—(外参数 变换矩阵Ｔ变换)—>  相机坐标系　—(同/Z)—>归一化平面坐标系——>径向和切向畸变纠正——>(内参数平移　Cx Cy 缩放焦距Fx Fy)
 ——> 图像坐标系下　像素坐标
 u=Fx *X/Z + Cx 　　像素列位置坐标　
 v=Fy *Y/Z + Cy 　　像素列位置坐标　
 
 反过来
 X=(u- Cx)*Z/Fx
 Y=(u- Cy)*Z/Fy
 Z轴归一化
 X=(u- Cx)*Z/Fx/depthScale
 Y=(u- Cy)*Z/Fy/depthScale
 Z=Z/depthScale
 
外参数　T
世界坐标　
pointWorld = T*[X Y Z]
 
 
 */
int main( int argc, char** argv )
{
    vector<cv::Mat> colorImgs, depthImgs;    // 彩色图和深度图
    vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> poses;         // 相机位姿
    /*
     pose.txt
     x,y,z,Qx,Qy,Qz,Qw  Qw为四元数实部
     -0.228993 0.00645704 0.0287837 -0.0004327 -0.113131 -0.0326832 0.993042
     -0.50237 -0.0661803 0.322012 -0.00152174 -0.32441 -0.0783827 0.942662
     -0.970912 -0.185889 0.872353 -0.00662576 -0.278681 -0.0736078 0.957536
     -1.41952 -0.279885 1.43657 -0.00926933 -0.222761 -0.0567118 0.973178
     -1.55819 -0.301094 1.6215 -0.02707 -0.250946 -0.0412848 0.966741
     分别为五张图相机的位置和姿态
     */
    ifstream fin("pose.txt");
    if (!fin)
    {
        cerr<<"请在有pose.txt的目录下运行此程序"<<endl;
        return 1;
    }
    
    for ( int i=0; i<5; i++ )
    {
        boost::format fmt( "./%s/%d.%s" ); //图像文件格式
        colorImgs.push_back( cv::imread( (fmt%"color"%(i+1)%"png").str() ));//color/1.png~5.png
        depthImgs.push_back( cv::imread( (fmt%"depth"%(i+1)%"pgm").str(), -1 ));//depth/1.pgm~5.pgm 使用-1读取原始图像
        
        double data[7] = {0};
        for ( auto& d:data )//txt文件的每一行数据
            fin>>d;
        Eigen::Quaterniond q( data[6], data[3], data[4], data[5] );//data[6]为四元数实部
        Eigen::Isometry3d T(q);//按四元数旋转
        T.pretranslate( Eigen::Vector3d( data[0], data[1], data[2] ));//加上平移
        poses.push_back( T );//保存每一行的旋转平移　变换矩阵T
    }
    
    // 计算点云并拼接
    // 相机内参 
    double cx = 325.5;//图像像素　原点平移
    double cy = 253.5;
    double fx = 518.0;//焦距和缩放  等效
    double fy = 519.0;
    double depthScale = 1000.0;//归一化平面用到?
    
    cout<<"正在将图像转换为点云..."<<endl;
    
    // 定义点云使用的格式：这里用的是XYZRGB　即　空间位置和RGB色彩像素对
    typedef pcl::PointXYZRGB PointT; //点云中的点对象
    typedef pcl::PointCloud<PointT> PointCloud;//整个点云对象
    
    // 新建一个点云
    PointCloud::Ptr pointCloud( new PointCloud ); 
    for ( int i=0; i<5; i++ )
    {
        cout<<"转换图像中: "<<i+1<<endl; 
        cv::Mat color = colorImgs[i]; //彩色图像
        cv::Mat depth = depthImgs[i];//深度图像
        Eigen::Isometry3d T = poses[i];//每个图像对应的摄像机位资
        for ( int v=0; v<color.rows; v++ )//每一行
            for ( int u=0; u<color.cols; u++ )//每一列
            {
	      //内参数　转换
                unsigned int d = depth.ptr<unsigned short> ( v )[u]; // 深度值
                if ( d==0 ) continue; // 为0表示没有测量到
                Eigen::Vector3d point; 
                point[2] = double(d)/depthScale; 
                point[0] = (u-cx)*point[2]/fx;
                point[1] = (v-cy)*point[2]/fy; 
		
                Eigen::Vector3d pointWorld = T*point;//位于世界坐标系中的实际位置  x,y,z
              //外参数  转换 
                PointT p ; //XYZRGB
                p.x = pointWorld[0];
                p.y = pointWorld[1];
                p.z = pointWorld[2];
                p.b = color.data[ v*color.step+u*color.channels() ];//注意opencv彩色图像通道的顺序为 bgr
                p.g = color.data[ v*color.step+u*color.channels()+1 ];
                p.r = color.data[ v*color.step+u*color.channels()+2 ];
                pointCloud->points.push_back( p );
            }
    }
    
    pointCloud->is_dense = false;
    cout<<"点云共有"<<pointCloud->size()<<"个点."<<endl;
    pcl::io::savePCDFileBinary("map.pcd", *pointCloud );
    //  可在命令行内使用　pcl_viewer map.pcd 查看点云数据
    return 0;
}
