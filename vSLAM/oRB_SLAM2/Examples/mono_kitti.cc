/**
* This file is part of ORB-SLAM2.
* 单目 相机 kitti数据集
*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>//时间
#include<iomanip>

#include<opencv2/core/core.hpp>

#include"System.h"

using namespace std;

// 读取图片目录  根据序列文件   返回 文件名字符串容器 和 对于时间戳序列
void LoadImages(const string &strSequence, vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps);

int main(int argc, char **argv)
{
    if(argc != 4)
    {                            // 用法 ./mono_kitti 词典路径  设置文件路径    数据集
        cerr << endl << "用法: ./mono_kitti path_to_vocabulary path_to_settings path_to_sequence" << endl;
        return 1;
    }

    // 更加序列文件  得到 图片文件路径  和 对于的序列
    vector<string> vstrImageFilenames;// 文件名 字符串 容器
    vector<double> vTimestamps;//图片 时间戳 double 容器
    LoadImages(string(argv[3]), vstrImageFilenames, vTimestamps);
    int nImages = vstrImageFilenames.size();//图片数量

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    // 创建 系统对象 完成了 字典读取 跟踪线程 建图线程  回环检测线程  可视化线程的 初始化启动
    // 初始化系统 传入字典   配置文件路径
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::MONOCULAR,true);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;// 跟踪线程  每一帧图像 跟踪处理的时间
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "开始处理 图片序列 ..." << endl;
    cout << "总图片数量:  " << nImages << endl << endl;

    // Main loop
    cv::Mat im;// 图像
    for(int ni=0; ni<nImages; ni++)//循环读取图片序列内的图像数据
    {
        // 读取图像 Read image from file
        im = cv::imread(vstrImageFilenames[ni],CV_LOAD_IMAGE_UNCHANGED);//读取图片
        double tframe = vTimestamps[ni];// 此帧图像对应的时间戳
        if(im.empty())
        {
            cerr << endl << "未能载入图像: " << vstrImageFilenames[ni] << endl;
            return 1;
        }

		// 时间记录 开始
	#ifdef COMPILEDWITHC11
		std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
	#else
		std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
	#endif

        // 讲图像传给 SLAM系统 Pass the image to the SLAM system
        SLAM.TrackMonocular(im,tframe);//单目 跟踪
	// 时间记录 结束
	#ifdef COMPILEDWITHC11
		std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
	#else
		std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
	#endif
        // 单目跟踪 一帧图像时间
        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
	 
	// 保持跟踪处理 的时间
        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];//两帧时间戳之差

        if(ttrack<T)// 跟踪时间 小于图像帧率时间  休息一会
            usleep((T-ttrack)*1e6);
    }

    // 关闭所有线程 Stop all threads
    SLAM.Shutdown();

    //每一帧跟踪时间排序  Tracking time statistics
    sort(vTimesTrack.begin(), vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];// 跟踪总时间
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;// 每一帧 跟踪 时间 中值 
    cout << "mean tracking time: " << totaltime/nImages << endl;// 均值

    // 保存 相机轨迹  Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");    

    return 0;
}

// 根据图片序列文件 生成 图片文件路径 容器 和 其 对应时间戳 容器 
void LoadImages(const string &strPathToSequence, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
	ifstream fTimes;
	string strPathTimeFile = strPathToSequence + "/times.txt";
	fTimes.open(strPathTimeFile.c_str());//打开文件
	while(!fTimes.eof())//到文件末尾
	{
	    string s;
	    getline(fTimes,s);//每一行
	    if(!s.empty())
	    {
		stringstream ss;
		ss << s;
		double t;
		ss >> t;//时间戳
		vTimestamps.push_back(t);// 存入时间戳容器
	    }
	}

	string strPrefixLeft = strPathToSequence + "/image_0/";//图片 父目录

	const int nTimes = vTimestamps.size();//总数量
	vstrImageFilenames.resize(nTimes);//

	for(int i=0; i<nTimes; i++)
	{
	    stringstream ss;
	    ss << setfill('0') << setw(6) << i;// 宽度6位  填充0 
	    vstrImageFilenames[i] = strPrefixLeft + ss.str() + ".png";// 图片文件完整路径
	}
}
