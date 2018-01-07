/*
 * LK光流法跟踪特征点
 * ./useLK ../../data
 * 
 * 1】灰度不变假设
 * I(x+dx, y+dy, t+dt) = I(x, y, t) 同一个空间点的像素灰度值 在各个时间点的图像上是固定不变的 假设
 *  泰勒展开  I(x+dx, y+dy, t+dt) = I(x, y, t)   +  I对x的偏导数 * dx + I对y的偏导数 * dy + I对t的偏导数 * dt =  I(x, y, t) 
 *  得到 I对x的偏导数 * dx + I对y的偏导数 * dy + I对t的偏导数 * dt = 0
 *  两边同时除以 dt   得到
 * I对x的偏导数 * dx/dt + I对y的偏导数 * dy/dt  = - I对t的偏导数
 * 
 * 其中 dx/dt  = 像素运动的速度 x轴分量 u； dy/dt为 像素运动的速度 y轴分量 v
 *         I对x的偏导数   为像素在该点 x方向 梯度 Ix   ；  I对y的偏导数   为像素在该点 y方向 梯度  Iy
 *         I对t的偏导数   为两个时间点图像 对应点 灰度变化 It
 *                                                | u | 
 * 写成 向量形式有  [ Ix   Iy] *  | v |  = - It   我们是想计算 像素的运动速度 即可由前一 像素坐标得到后一像素坐标
 *                                
 * 2】同区域 像素具有相同的运动速度 假设 w×w的大小的窗口内 像素的运动速度相同
 * 则    [ Ix   Iy]1		    u       |It1
 *         [ Ix   Iy]2             *  v   = -|It2
 *          ...				     |...
 *         [ Ix   Iy]w*w  		     |Itw*w 
 * 写成矩阵形式 
 *     A *   |u|   =  -b
 *              |v|
 * 
 *   | u|   =  -  A逆* b  =  - A逆 *  A转置 的 逆 * A转置 * b = -( A 转置 * A)逆 * A转置 * b 
 *   | v|
 * 得到像素 在图像空间运动速度 u v 
 * 
 */
#include <iostream>//输入输出流
#include <fstream>//文件数据流
#include <list>//列表
#include <vector>//容器
#include <chrono>//计时
using namespace std; 
//opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>//二维特征
#include <opencv2/video/tracking.hpp>//跟踪算法

int main( int argc, char** argv )
{
    if ( argc != 2 )
    {
        cout<<"用法（TMU数据集）: useLK path_to_dataset"<<endl;
        return 1;
    }
    /*
     TMU数据集:
     rgb.txt       记录了RGB图像的采集时间 和对应的文件名
     depth.txt   记录了深度图像的采集时间 和对应的文件名
     /rgb           存放 rgb图像  png格式彩色图为八位3通道
     /depth       存放深度图像  深度图为 16位单通道图像
     groundtruth.txt 为外部运动捕捉系统采集到的相机位姿  time,tx,ty,tz,qx,qy,qz,qw
     RGB图像和 深度图像采集独立的 时间不同时，需要对数据进行一次时间上的对齐，
     时间间隔相差一个阈值认为是同一匹配图
     可使用 associate.py脚步完成   python associate.py rgb.txt   depth.txt  > associate.txt
     
     */
    string path_to_dataset = argv[1];//数据集路径
    string associate_file = path_to_dataset + "/associate.txt";//匹配好的图像
    
    ifstream fin( associate_file );//读取文件   ofstream 输出文件流
    if ( !fin ) //打开文件失败
    {
        cerr<<"找不到 associate.txt!"<<endl;
        return 1;
    }
    
    string rgb_file, depth_file, time_rgb, time_depth;//字符串流
    list<cv::Point2f > keypoints;      // 因为要删除跟踪失败的点，使用list
    cv::Mat color, depth, last_color;
    
    for ( int index=0; index<100; index++ )
    {
        fin>>time_rgb>>rgb_file>>time_depth>>depth_file;
	//rgb图像对应时间 rgb图像 深度图像对应时间 深度图像
        color = cv::imread( path_to_dataset+"/"+rgb_file );//彩色图像
        depth = cv::imread( path_to_dataset+"/"+depth_file, -1 );//原始图像
        if (index ==0 )//第一帧图像
        {
            // 对第一帧提取FAST特征点
            vector<cv::KeyPoint> kps;//关键点容器
            cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();//检测器
            detector->detect( color, kps );//检测关键点 放入容器内
            for ( auto kp:kps )
                keypoints.push_back( kp.pt );//存入 关键点 二维坐标  列表内
            last_color = color;
            continue;// index ==0 时 执行到这里 以下不执行
        }
        
        if ( color.data==nullptr || depth.data==nullptr )//图像读取错误 跳过
            continue;
	
        // 对其他帧用LK跟踪特征点
        vector<cv::Point2f> next_keypoints; //下一帧关键点
        vector<cv::Point2f> prev_keypoints; //上一帧关键点
        for ( auto kp:keypoints )
            prev_keypoints.push_back(kp);//上一帧关键点
        vector<unsigned char> status;// 关键点 跟踪状态标志
        vector<float> error; //信息
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();//计时 开始
        cv::calcOpticalFlowPyrLK( last_color, color, prev_keypoints, next_keypoints, status, error );
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();//计时 结束
        chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
        cout<<"LK Flow use time："<<time_used.count()<<" seconds."<<endl;
        // 把跟丢的点删掉
        int i=0; 
        for ( auto iter=keypoints.begin(); iter!=keypoints.end(); i++) 
        {
            if ( status[i] == 0 )//状态为0  跟踪 失败
            {
                iter = keypoints.erase(iter);// list列表 具有管理表格元素能力  删除跟踪失败的点
                continue;//跳过执行 后面两步
            }
            *iter = next_keypoints[i];// s
            iter++;// 迭代
        }
        
        cout<<"跟踪的关键点数量 tracked keypoints: "<<keypoints.size()<<endl;
        if (keypoints.size() == 0)
        {
            cout<<"所有关键点已丢失 all keypoints are lost."<<endl;
            break; 
        }
        // 画出 keypoints
        cv::Mat img_show = color.clone();//复制图像 开辟新的内存空间
        for ( auto kp:keypoints )
            cv::circle(img_show, kp, 10, cv::Scalar(0, 240, 0), 1);//画圆  图像  中心点  半径 颜色 粗细
        cv::imshow("corners", img_show);//显示
        cv::waitKey(0);//等待按键按下 遍历下一张图片
        last_color = color;
    }
    return 0;
}
