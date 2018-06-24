/*
auther: wanyouwen
date: 2018.6.24

双目示例:
双目相机app程序框架：

1. 读取相机配置文件(内参数 畸变矫正参数 双目对齐变换矩阵) =======================
   cv::FileStorage fsSettings(setting_filename, cv::FileStorage::READ);
   fsSettings["LEFT.K"] >> K_l;//内参数
   fsSettings["LEFT.D"] >> D_l;// 畸变矫正
   fsSettings["LEFT.P"] >> P_l;// P_l,P_r --左右相机在校准后坐标系中的投影矩阵 3×4
   fsSettings["LEFT.R"] >> R_l;// R_l,R_r --左右相机校准变换（旋转）矩阵  3×3
2. 计算双目矫正映射矩阵========================================================
   cv::Mat M1l,M2l,M1r,M2r;
       cv::initUndistortRectifyMap(K_l,D_l,R_l,P_l.rowRange(0,3).colRange(0,3),cv::Size(cols_l,rows_l),CV_32F,M1l,M2l);
       cv::initUndistortRectifyMap(K_r,D_r,R_r,P_r.rowRange(0,3).colRange(0,3),cv::Size(cols_r,rows_r),CV_32F,M1r,M2r);
3. 创建双目系统=============================================================== 
   ORB_SLAM2::System SLAM(vocabulary_filepath, setting_filename, ORB_SLAM2::System::STEREO, true);
4. 从双目设备捕获图像,设置分辨率捕获图像========================================
   cv::VideoCapture CapAll(deviceid); //打开相机设备 
   //设置分辨率   1280*480  分成两张 640*480  × 2 左右相机
       CapAll.set(CV_CAP_PROP_FRAME_WIDTH,1280);  
       CapAll.set(CV_CAP_PROP_FRAME_HEIGHT, 480); 
    5. 获取左右相机图像===========================================================
   CapAll.read(src_img);
   imLeft  = src_img(cv::Range(0, 480), cv::Range(0, 640));   
       imRight = src_img(cv::Range(0, 480), cv::Range(640, 1280));   
6. 使用2步获取的双目矫正映射矩阵 矫正 左右相机图像==============================
   cv::remap(imLeft,imLeftRect,M1l,M2l,cv::INTER_LINEAR);
       cv::remap(imRight,imRightRect,M1r,M2r,cv::INTER_LINEAR);
7. 记录时间戳 time ，并计时===================================================
	#ifdef COMPILEDWITHC11
	   std::chrono::steady_clock::time_point    t1 = std::chrono::steady_clock::now();
	#else
	   std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
	#endif
    8. 把左右图像和时间戳 传给 SLAM系统===========================================
    SLAM.TrackStereo(imLeftRect, imRightRect, time);
9. 计时结束，计算时间差，处理时间============================================= 
10.循环执行 5-9步===========================================================
11.结束，关闭slam系统，关闭所有线程===========================================   
12.保存相机轨迹============================================================== 

*/
#include<iostream>
#include<algorithm>
#include<fstream>
#include<iomanip>
#include<chrono>//chrono是一个time library, 源于boost，现在已经是C++标准。

#include<opencv2/core/core.hpp>
#include<System.h>

using namespace std;
using namespace cv;

#include <boost/format.hpp>  // 格式化字符串 for formating strings 处理图像文件格式
#include <boost/thread/thread.hpp>

static void print_help()
{
    printf("\n orbslam2 stereo test : \n");//生成视差和点云图
    printf("\n Usage: my_stereo [-v=<path_to_vocabulary>] [-s=<path_to_settings>]\n"
           "\n [--d=<camera_id>] \n");
}


int main(int argc, char **argv)
{
    //std::uint32_t timestamp;
    double time=0.0, ttrack=0;

    //vector<IMUData> imudatas;
    std::string setting_filename = "";    //配置文件
    std::string vocabulary_filepath = ""; //关键帧数据库 词典 重定位  
    int deviceid = 1;                     //相机设备id
    cv::CommandLineParser parser(argc, argv,
        "{help h||}{d|1|}{v|../../Vocabulary/ORBvoc.bin|}{s|my_stereo.yaml|}");
//=======打印帮助信息============
    if(parser.has("help"))
    {
        print_help();
        return 0;
    }
    if( parser.has("d") )
        deviceid = parser.get<int>("d");//相机设备id
    if( parser.has("s") )
        setting_filename = parser.get<std::string>("s");//
    if( parser.has("v") )
        vocabulary_filepath = parser.get<std::string>("v");//
    if (!parser.check()) {
        parser.printErrors();
        return 1;
    }
//if(setting_filename.empty())//{

 //1.  读取相机配置文件(内参数 畸变矫正参数 双目对齐变换矩阵 ) ====================
    cv::FileStorage fsSettings(setting_filename, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        cerr << "ERROR: Wrong path to setting file : " << setting_filename << endl;
        return -1;
    }

    cv::Mat K_l, K_r, D_l, D_r, P_l, P_r, R_l, R_r;
    fsSettings["LEFT.K"] >> K_l;//内参数 
    if(K_l.empty())  cout << "K_l missing "<<endl;
    fsSettings["RIGHT.K"] >> K_r;//
    if(K_r.empty()) cout << "K_r missing "<<endl;
    fsSettings["LEFT.D"] >> D_l;// 畸变矫正
    if(D_l.empty()) cout << "D_l missing "<<endl;
    fsSettings["RIGHT.D"] >> D_r;
    if(D_r.empty()) cout << "D_r missing "<<endl;

    fsSettings["LEFT.P"] >> P_l;// P_l,P_r --左右相机在校准后坐标系中的投影矩阵 3×4
    if(P_l.empty()) cout << "P_l missing "<<endl;
    fsSettings["RIGHT.P"] >> P_r;
    if(K_r.empty()) cout << "P_r missing "<<endl;

    fsSettings["LEFT.R"] >> R_l;// R_l,R_r --左右相机校准变换（旋转）矩阵  3×3
    if(R_l.empty()) cout << "R_l missing "<<endl;
    fsSettings["RIGHT.R"] >> R_r;
    if(R_r.empty()) cout << "R_r missing "<<endl;

    int rows_l = fsSettings["LEFT.height"];
    int cols_l = fsSettings["LEFT.width"];
    int rows_r = fsSettings["RIGHT.height"];
    int cols_r = fsSettings["RIGHT.width"];

    if(K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() || R_l.empty() || R_r.empty() || D_l.empty() || D_r.empty() || rows_l==0 || rows_r==0 || cols_l==0 || cols_r==0)
    {
        cerr << "ERROR: Calibration parameters to rectify stereo are missing!" << endl;
        return -1;
    }
//}

//外参数
// Mat R, T, R1, P1, R2, P2;
//fs["R"] >> R;
//fs["T"] >> T;	
//图像矫正摆正 映射计算  
//stereoRectify( K_l, D_l, K_r, D_r, cv::Size(cols_l,rows_l), R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, cv::Size(cols_l,rows_l), &roi1, &roi2 );

// 2. 计算双目矫正映射矩阵================================================
    cv::Mat M1l,M2l,M1r,M2r;
    cv::initUndistortRectifyMap(K_l,D_l,R_l,P_l.rowRange(0,3).colRange(0,3),cv::Size(cols_l,rows_l),CV_32F,M1l,M2l);
    cv::initUndistortRectifyMap(K_r,D_r,R_r,P_r.rowRange(0,3).colRange(0,3),cv::Size(cols_r,rows_r),CV_32F,M1r,M2r);

// 3. 创建双目系统 ORB_SLAM2::System======================================
    ORB_SLAM2::System SLAM(vocabulary_filepath, setting_filename, ORB_SLAM2::System::STEREO, true);


    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
// 4. 从双目设备捕获图像设置分辨率捕获图像 ===================================================
    cv::VideoCapture CapAll(deviceid); //打开相机设备 
    if( !CapAll.isOpened() ) 
    { 
       printf("打开摄像头失败\r\n");
       printf("再试一次\r\n");
       //sleep(1);//延时1秒 
       cv::VideoCapture CapAll(1); //打开相机设备 
       if( !CapAll.isOpened() ) { 
         printf("打开摄像头失败\r\n");
	 printf("再试一次..\r\n");
          //sleep(1);//延时1秒 
          cv::VideoCapture CapAll(1); //打开相机设备 
          if( !CapAll.isOpened() ) { 
	  printf("打开摄像头失败\r\n");
	   return -1;
         }
       }
   }
    //设置分辨率   1280*480  分成两张 640*480  × 2 左右相机
    CapAll.set(CV_CAP_PROP_FRAME_WIDTH,1280);  
    CapAll.set(CV_CAP_PROP_FRAME_HEIGHT, 480);  

    cv::Mat src_img, imLeft, imRight, imLeftRect, imRightRect;

    while(CapAll.read(src_img)) 
     {
// 5. 获取左右相机图像====================================================
        imLeft  = src_img(cv::Range(0, 480), cv::Range(0, 640));   
        imRight = src_img(cv::Range(0, 480), cv::Range(640, 1280)); 

// 6. 矫正左右相机图像====================================================
        cv::remap(imLeft,imLeftRect,M1l,M2l,cv::INTER_LINEAR);
        cv::remap(imRight,imRightRect,M1r,M2r,cv::INTER_LINEAR);

// 7. 记录时间戳 tframe ，并计时==========================================
#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point        t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif	
        time += ttrack ;

// 8. 把左右图像和时间戳 传给 SLAM系统====================================
        SLAM.TrackStereo(imLeftRect, imRightRect, time);

// 9. 计时结束，计算时间差，处理时间======================================	
#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point        t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
	
  //  if(ttrack<T)
//   usleep((T-ttrack)*1e6); //sleep
	
    }

// 10. 结束，关闭slam系统，关闭所有线程===================================
    SLAM.Shutdown();
// 11. 保存相机轨迹======================================================
    SLAM.SaveTrajectoryKITTI("myCameraTrajectory.txt");

    CapAll.release();
    cv::destroyAllWindows();
    return 0;
}
