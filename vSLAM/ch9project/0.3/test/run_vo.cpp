// -------------- test the visual odometry -------------
/*
 *myslam::VisualOdometry
 * ORB特征点 描述子 汉明字符串距离匹配得到 特征点对
 * 其中 一幅图像的2D像素点 根据 深度信息和 相机内参数 转化成 3D点，组成2D-3D点对
 * 使用随机采样序列的 PnP算法求解 转换矩阵 根据 符合 转换矩阵的内点数量判断 求解得到的 转换矩阵的好坏
 * 把第一帧图像设为  世界坐标系原点
 * 以后的每一帧通过 与上一帧的转换矩阵 转换到世界坐标系下
 * 根据得到的 转换矩阵，若 旋转向量 和 平移矩阵 的大小都超过一定限度，则认为该帧是一个关键帧
 * 保存3D坐标(转化到相机坐标系系下) 和 对应的描述子
 * */
#include <fstream>
#include <boost/timer.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/viz.hpp> 

#include "myslam/config.h"
#include "myslam/visual_odometry.h"

int main ( int argc, char** argv )
{
    if ( argc != 2 )
    {
        cout<<"用法: run_vo parameter_file"<<endl;
        return 1;
    }

    myslam::Config::setParameterFile ( argv[1] );//读取配置文件
    myslam::VisualOdometry::Ptr vo ( new myslam::VisualOdometry );//视觉里程计

    string dataset_dir = myslam::Config::get<string> ( "dataset_dir" );
    cout<<"dataset: "<<dataset_dir<<endl;
    ifstream fin ( dataset_dir+"/associate.txt" );
    if ( !fin )
    {
        cout<<"please generate the associate file called associate.txt!"<<endl;
        return 1;
    }

    vector<string> rgb_files, depth_files;//文件名
    vector<double> rgb_times, depth_times;//时间序列 容器 vector 实为 数组实现  内存 连续     
    while ( !fin.eof() )//读取数据集列表
    {
        string rgb_time, rgb_file, depth_time, depth_file;
        fin>>rgb_time>>rgb_file>>depth_time>>depth_file;
        rgb_times.push_back ( atof ( rgb_time.c_str() ) );
        depth_times.push_back ( atof ( depth_time.c_str() ) );
        rgb_files.push_back ( dataset_dir+"/"+rgb_file );
        depth_files.push_back ( dataset_dir+"/"+depth_file );

        if ( fin.good() == false )
            break;
    }

    myslam::Camera::Ptr camera ( new myslam::Camera );
    
    // visualization 可视化
    cv::viz::Viz3d vis("Visual Odometry");
    cv::viz::WCoordinateSystem world_coor(1.0), camera_coor(0.5);
    cv::Point3d cam_pos( 0, -1.0, -1.0 ), cam_focal_point(0,0,0), cam_y_dir(0,1,0);
    cv::Affine3d cam_pose = cv::viz::makeCameraPose( cam_pos, cam_focal_point, cam_y_dir );
    vis.setViewerPose( cam_pose );
    
    world_coor.setRenderingProperty(cv::viz::LINE_WIDTH, 2.0);
    camera_coor.setRenderingProperty(cv::viz::LINE_WIDTH, 1.0);
    vis.showWidget( "World", world_coor );
    vis.showWidget( "Camera", camera_coor );

    cout<<"read total "<<rgb_files.size() <<" entries"<<endl;
    for ( int i=0; i<rgb_files.size(); i++ )
    {
        Mat color = cv::imread ( rgb_files[i] );//彩色图
        Mat depth = cv::imread ( depth_files[i], -1 );//深度图
        if ( color.data==nullptr || depth.data==nullptr )
            break;
        myslam::Frame::Ptr pFrame = myslam::Frame::createFrame();//帧
        pFrame->camera_ = camera;
        pFrame->color_ = color;
        pFrame->depth_ = depth;
        pFrame->time_stamp_ = rgb_times[i];

        boost::timer timer;
        vo->addFrame ( pFrame );
        cout<<"VO costs time: "<<timer.elapsed()<<endl;
        
        if ( vo->state_ == myslam::VisualOdometry::LOST )//vo视觉里程计
            break;
        SE3 Tcw = pFrame->T_c_w_.inverse();
        
        // show the map and the camera pose  照相机位姿
        cv::Affine3d M(//相机位姿
            cv::Affine3d::Mat3( 
                Tcw.rotation_matrix()(0,0), Tcw.rotation_matrix()(0,1), Tcw.rotation_matrix()(0,2),
                Tcw.rotation_matrix()(1,0), Tcw.rotation_matrix()(1,1), Tcw.rotation_matrix()(1,2),
                Tcw.rotation_matrix()(2,0), Tcw.rotation_matrix()(2,1), Tcw.rotation_matrix()(2,2)
            ), 
            cv::Affine3d::Vec3(
                Tcw.translation()(0,0), Tcw.translation()(1,0), Tcw.translation()(2,0)
            )
        );
        cv::imshow("image", color );//显示图片
        //cv::waitKey(1);
	cv::waitKey(0);
        vis.setWidgetPose( "Camera", M);//显示相机位姿
        vis.spinOnce(1, false);
    }

    return 0;
}
