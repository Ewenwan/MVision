/**
* This file is part of ORB-SLAM2.
* 图漾 rgbd orbslam2 示例
* 运行 ： ./ty_rgbd ../../Vocabulary/ORBvoc.bin ./my_rgbd_ty_api_adj.yaml
*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include<opencv2/core/core.hpp>

#include<System.h>


#include "./common/common.hpp" // 图漾的头文件

using namespace std;

// 1. 参数初始化===============================
static char buffer[1024 * 1024 * 20];// 存储 sdk 版本等信息
static int  n;// usb设备号
static volatile bool exit_main;

static int fps_counter = 0;
static clock_t fps_tm = 0;
const char* IP = NULL; // 指向只读字符的 指针IP 网络设备IP地址
TY_DEV_HANDLE hDevice; // 设备 handle
//int32_t color = 1, ir = 0, depth = 1; // rgb图像 ir红外图像  depth深度图像
//int i = 1; // 捕获图片index

char img_color_file_name[15]; 

char* frameBuffer[2]; // 帧 缓冲区

double slam_sys_time=0.0, ttrack=0; // 时间戳

// ORB_SLAM2::System SLAM();

int capture_ok_flag = 0;

cv::Mat color_img, depth_img;// 彩色图 和 灰度图=====
cv::Mat  undistort_result;   // 彩色矫正图
cv::Mat newDepth;            // 配置后的深度图

// 回调函数数据 结构体===================
struct CallbackData {
    int             index;
    TY_DEV_HANDLE   hDevice;
    DepthRender*    render;
    TY_CAMERA_DISTORTION color_dist;// 畸变参数
    TY_CAMERA_INTRINSIC color_intri;// 内参数
};
CallbackData cb_data;     // 回调函数数据=====

int get_fps(); // 帧数/s
void eventCallback(TY_EVENT_INFO *event_info, void *userdata);// 事件回调函数
void frameHandler(TY_FRAME_DATA* frame, void* userdata, ORB_SLAM2::System & SLAM2);// 帧回调函数
void frameHandler(TY_FRAME_DATA* frame, void* userdata);// 帧回调函数
int ty_RGBD_init(void); // 图漾相机初始化

int main(int argc, char **argv)
{
// 1. 检测命令行输入
    if(argc != 3)
    {
        cerr << endl << "Usage: ./my_rgbd path_to_vocabulary path_to_settings" << endl;
        return 1;
    }

// 2. 图漾相机初始化================================
    int init_flag = ty_RGBD_init();
    if(init_flag == -1) 
    {
	LOGD("=== camera init failed===");
        return -1;
    }


// 3. 创建双目系统 ORB_SLAM2::System======================================
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::RGBD,true);
   //SLAM(argv[1],argv[2],ORB_SLAM2::System::RGBD,true);


    cout << endl << "-------" << endl;
    cout << "Start processing ..." << endl;
    
// 7. 启动采集==============================
    LOGD("=== Start capture");
    ASSERT_OK( TYStartCapture(hDevice) );


// 7.5 获取相机参数=======
    LOGD("=== Read color rectify matrix");
    {
        TY_CAMERA_DISTORTION color_dist;// 相机 畸变参数
        TY_CAMERA_INTRINSIC color_intri;// 相机 内参数
        TY_STATUS ret = TYGetStruct(hDevice, TY_COMPONENT_RGB_CAM, 
                                    TY_STRUCT_CAM_DISTORTION, &color_dist, sizeof(color_dist));
        ret |= TYGetStruct(hDevice, TY_COMPONENT_RGB_CAM, 
                           TY_STRUCT_CAM_INTRINSIC, &color_intri, sizeof(color_intri));
        if (ret == TY_STATUS_OK)
        {
            cb_data.color_intri = color_intri;// 相机 内参数
            cb_data.color_dist= color_dist;   // 相机 畸变参数
        }
        else
        { //reading data from device failed .set some default values....
            memset(cb_data.color_dist.data, 0, 12 * sizeof(float));
            // 畸变参数 k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4
            memset(cb_data.color_intri.data, 0, 9 * sizeof(float));// 内参数
            cb_data.color_intri.data[0] = 1000.f;// fx
            cb_data.color_intri.data[4] = 1000.f;// fy
            cb_data.color_intri.data[2] = 600.f;// cx
            cb_data.color_intri.data[5] = 450.f;// cy
        }
    }


    LOGD("=== While loop to fetch frame");
    exit_main = false;
    TY_FRAME_DATA frame; // 每一帧数据
    
    while(!exit_main) 
    {
// 8. 主动获取帧数据(主动获取模式下不调用)===========
        int err = TYFetchFrame(hDevice, &frame, -1);// 获取一帧数据
        if( err != TY_STATUS_OK ) 
        {
            LOGD("... Drop one frame");
        } 
        else 
        { // 处理获取到的帧数据============
            //frameHandler(&frame, &cb_data, SLAM);
              frameHandler(&frame, &cb_data);
        }
        
        if(capture_ok_flag){
                LOGD("... tracking ... ");
		capture_ok_flag = 0;
	    // 7. 记录时间戳 tframe ，并计时==========================================
	#ifdef COMPILEDWITHC11
		std::chrono::steady_clock::time_point    t1 = std::chrono::steady_clock::now();
	#else
		std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
	#endif	
		slam_sys_time += ttrack ;

	// 8. 把左右图像和时间戳 传给 SLAM系统====================================
		SLAM.TrackRGBD(color_img, depth_img, slam_sys_time);

	// 9. 计时结束，计算时间差，处理时间======================================	
	#ifdef COMPILEDWITHC11
		std::chrono::steady_clock::time_point        t2 = std::chrono::steady_clock::now();
	#else
		std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
	#endif

		ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
		
        }
	
	
    }

// 10. 结束，关闭slam系统，关闭所有线程===================================
    // Stop all threads
    SLAM.Shutdown();

// 9. 设备停止采集=================
    ASSERT_OK( TYStopCapture(hDevice) );
// 10. 关闭设备=================
    ASSERT_OK( TYCloseDevice(hDevice) );
// 11. 释放 API=================
    ASSERT_OK( TYDeinitLib() );
    // MSLEEP(10); // sleep to ensure buffer is not used any more

// 12. 释放内存空间==============
    delete frameBuffer[0];
    delete frameBuffer[1];

    LOGD("=== Main done!");

// 11. 保存相机轨迹======================================================
    // Save camera trajectory
    SLAM.SaveTrajectoryTUM("my_ty_rgbd_CameraTrajectory.txt");
    //SLAM.SaveKeyFrameTrajectoryTUM("my_ty_rgbd_KeyFrameTrajectory.txt");   

    return 0;
}


//================================================
//=== 计算帧数 fps
//================================================
#ifdef _WIN32
// 帧数===================
int get_fps() {
   const int kMaxCounter = 250;
   fps_counter++;
   if (fps_counter < kMaxCounter) {
     return -1;
   }
   int elapse = (clock() - fps_tm);
   int v = (int)(((float)fps_counter) / elapse * CLOCKS_PER_SEC);
   fps_tm = clock();

   fps_counter = 0;
   return v;
 }

#else
int get_fps() {
  const int kMaxCounter = 200;
  struct timeval start;
  fps_counter++;
  if (fps_counter < kMaxCounter) {
    return -1;
  }

  gettimeofday(&start, NULL);
  int elapse = start.tv_sec * 1000 + start.tv_usec / 1000 - fps_tm;
  int v = (int)(((float)fps_counter) / elapse * 1000);
  gettimeofday(&start, NULL);
  fps_tm = start.tv_sec * 1000 + start.tv_usec / 1000;

  fps_counter = 0;
  return v;
}
#endif


//=========================================================
//=== 处理相机帧的 回调函数
//=========================================================
void frameHandler(TY_FRAME_DATA* frame, void* userdata, ORB_SLAM2::System & SLAM2) {
    CallbackData* pData = (CallbackData*) userdata;
    // LOGD("=== Get frame %d", ++pData->index);// 帧id

   // int ret = get_fps();
   //   if (ret > 0)
   //   printf("fps: %d\n", ret);

    // cv::Mat depth, irl, irr, color;
    // parseFrame(*frame, &depth, &irl, &irr, &color, 0);
    // cv::Mat color_img, depth_img;
    parseFrame(*frame, &depth_img, 0, 0, &color_img, 0);

    //if(!depth.empty()){
    //    cv::Mat colorDepth = pData->render->Compute(depth);
    //    cv::imshow("ColorDepth", colorDepth);// 彩色深度图
    //}
    //if(!irl.empty()){ cv::imshow("LeftIR", irl); }
    //if(!irr.empty()){ cv::imshow("RightIR", irr); }
    // cv::namedWindow("Color", CV_WINDOW_NORMAL);//CV_WINDOW_NORMAL就是0
    //if(!color.empty()){ cv::imshow("Color", color); }

if((!depth_img.empty()) && (!color_img.empty()))
{
    // 7. 记录时间戳 tframe ，并计时==========================================
#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point    t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif	
        slam_sys_time += ttrack ;

// 8. 把左右图像和时间戳 传给 SLAM系统====================================
        SLAM2.TrackStereo(color_img, depth_img, slam_sys_time);

// 9. 计时结束，计算时间差，处理时间======================================	
#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point        t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
}

    int key = cv::waitKey(1);
    switch(key & 0xff) 
    {
    case 0xff:
        break;
    case 'q':
        exit_main = true;
        break;
    default:
        LOGD("Unmapped key %d", key);
    }

    //LOGD("=== Callback: Re-enqueue buffer(%p, %d)", frame->userBuffer, frame->bufferSize);
    ASSERT_OK( TYEnqueueBuffer(pData->hDevice, frame->userBuffer, frame->bufferSize) );
}

//=========================================================
//=== 处理相机帧的 回调函数
//=========================================================
void frameHandler(TY_FRAME_DATA* frame, void* userdata) {
    CallbackData* pData = (CallbackData*) userdata;
    // LOGD("=== Get frame %d", ++pData->index);// 帧id

   // int ret = get_fps();
   //   if (ret > 0)
   //   printf("fps: %d\n", ret);

    // cv::Mat depth_src, irl, irr, color_src, point3D;
    cv::Mat irl, irr, point3D;
    parseFrame(*frame, &depth_img, &irl, &irr, &color_img, &point3D);

    //if(!depth_img.empty()){
    //    cv::Mat colorDepth = pData->render->Compute(depth_img);
    //    cv::imshow("ColorDepth", colorDepth);// 彩色深度图
    //}
    //if(!irl.empty()){ cv::imshow("LeftIR", irl); }
    //if(!irr.empty()){ cv::imshow("RightIR", irr); }
    // cv::namedWindow("Color", CV_WINDOW_NORMAL);//CV_WINDOW_NORMAL就是0
    // if(!color_img.empty()){ cv::imshow("Color", color_img); }

    // 矫正彩色图像=======
    if(!color_img.empty())
    {
        //cv::Mat undistort_result(color_img.size(), CV_8UC3);// 矫正结果
        undistort_result = cv::Mat(color_img.size(), CV_8UC3);// 矫正结果
        TY_IMAGE_DATA dst;        // 目标图像
        dst.width = color_img.cols;   // 宽度 列数
        dst.height = color_img.rows;  // 高度 行数
        dst.size = undistort_result.size().area() * 3;// 3通道 
        dst.buffer = undistort_result.data;
        dst.pixelFormat = TY_PIXEL_FORMAT_RGB; // RGB 格式

        TY_IMAGE_DATA src;        // 源图像=================
        src.width = color_img.cols;
        src.height = color_img.rows;
        src.size = color_img.size().area() * 3;
        src.pixelFormat = TY_PIXEL_FORMAT_RGB;
        src.buffer = color_img.data; 
        //undistort camera image 
        //TYUndistortImage accept TY_IMAGE_DATA from TY_FRAME_DATA , pixel format RGB888 or MONO8
        //you can also use opencv API cv::undistort to do this job.
        ASSERT_OK(TYUndistortImage(&pData->color_intri, &pData->color_dist, NULL, &src, &dst));
        color_img = undistort_result;// 畸变矫正后的图像=== 可能需要 深拷贝 !!!!!!!!!!==也定义成全局变量===

       // cv::Mat resizedColor;// 彩色图缩放到 和 深度图一样大, 本身设定就是一样大的
       // cv::resize(color, resizedColor, depth.size(), 0, 0, CV_INTER_LINEAR);
       // cv::imshow("color", resizedColor);
    }

// 彩色图和深度图配准========3d点云反投影到图像上，获取对应像素点的深度值=====
    // do Registration
    // cv::Mat newDepth; // 定义成全局变量======
    if(!point3D.empty() && !color_img.empty()) 
    {
        ASSERT_OK( TYRegisterWorldToColor2(pData->hDevice, (TY_VECT_3F*)point3D.data, 0, 
                   point3D.cols * point3D.rows, color_img.cols, color_img.rows, (uint16_t*)buffer, sizeof(buffer)
                    ));
        newDepth = cv::Mat(color_img.rows, color_img.cols, CV_16U, (uint16_t*)buffer);
        cv::Mat resized_color;
        cv::Mat temp;
        //you may want to use median filter to fill holes in projected depth image or do something else here
        cv::medianBlur(newDepth,temp,5);// 对3d点云反投影到 彩色图上 获取的带有 孔洞的 深度图 进行均值滤波======
        newDepth = temp;
        //resize to the same size for display
        // cv::resize(newDepth, newDepth, depth_src.size(), 0, 0, 0);// 深度图  0 填充
        // cv::resize(color, resized_color, depth_src.size());// 彩色图

       // cv::Mat depthColor = pData->render->Compute(newDepth);// 彩色深度图
       // cv::imshow("Registrated ColorDepth", depthColor);// 显示 

       // depthColor = depthColor / 2 + resized_color / 2; // c彩色深度图 和 彩色图合并在一起
       // cv::imshow("projected depth", depthColor);// 显示 
    } 

    if((!depth_img.empty()) && (!color_img.empty()) && (!point3D.empty()))
    {
	capture_ok_flag = 1;// 采集成功  ？？？
    }

    int key = cv::waitKey(1);
    switch(key & 0xff) 
    {
    case 0xff:
        break;
    case 'q':
        exit_main = true;
        break;
    default:
        LOGD("Unmapped key %d", key);
    }
    //LOGD("=== Callback: Re-enqueue buffer(%p, %d)", frame->userBuffer, frame->bufferSize);
    ASSERT_OK( TYEnqueueBuffer(pData->hDevice, frame->userBuffer, frame->bufferSize) );
}
//==================================================
//====事件 回调函数
//==================================================
void eventCallback(TY_EVENT_INFO *event_info, void *userdata)
{
    if (event_info->eventId == TY_EVENT_DEVICE_OFFLINE) {
        LOGD("=== Event Callback: Device Offline!");
        // Note: 
        //     Please set TY_BOOL_KEEP_ALIVE_ONOFF feature to false if you need to debug with breakpoint!
    }
    else if (event_info->eventId == TY_EVENT_LICENSE_ERROR) {
        LOGD("=== Event Callback: License Error!");
    }
}


//===================================================
//======图漾相机初始化
//===================================================
int ty_RGBD_init(void)
{
    //cv::namedWindow("Color", CV_WINDOW_NORMAL);//CV_WINDOW_NORMAL就是0
// 1. 初始化 API ==============
    LOGD("=== Init lib");
    ASSERT_OK( TYInitLib() );
    TY_VERSION_INFO* pVer = (TY_VERSION_INFO*)buffer;
    ASSERT_OK( TYLibVersion(pVer) );// 获取sdk 软件版本信息========
    LOGD("     - lib version: %d.%d.%d", pVer->major, pVer->minor, pVer->patch);

// 2. 打开设备=================
    if(IP) // 打开网络设备
    {
        LOGD("=== Open device %s", IP);
        ASSERT_OK( TYOpenDeviceWithIP(IP, &hDevice) );
    } 
    else   // 打开 USB设备
    {
        LOGD("=== Get device info");
        ASSERT_OK( TYGetDeviceNumber(&n) );// 获取设备号
        LOGD("     - device number %d", n);

        TY_DEVICE_BASE_INFO* pBaseInfo = (TY_DEVICE_BASE_INFO*)buffer;
        ASSERT_OK( TYGetDeviceList(pBaseInfo, 100, &n) );

        if(n == 0) {
            LOGD("=== No device got");
            return -1;
        }

        LOGD("=== Open device 0");
        ASSERT_OK( TYOpenDevice(pBaseInfo[0].id, &hDevice) );
    }

// 3. 操作组件===================
    int32_t allComps;
    ASSERT_OK( TYGetComponentIDs(hDevice, &allComps) );// 查询组件状态
    // TYGetEnabledComponents(hDevice, &allComps) 查询使能状态的组件
    if(allComps & TY_COMPONENT_RGB_CAM) 
    { // 使能 RGB 组件功能==
        LOGD("=== Has RGB camera, open RGB cam");
        ASSERT_OK( TYEnableComponents(hDevice, TY_COMPONENT_RGB_CAM) );
    }

/*
    int32_t componentIDs = 0;
    LOGD("=== Configure components, open depth cam");
    if (depth) 
    {// 深度图像
        componentIDs = TY_COMPONENT_DEPTH_CAM;
    }

    if (ir) 
    {// 红外图像
        componentIDs |= TY_COMPONENT_IR_CAM_LEFT;
    }

    if (depth || ir) 
    {
        ASSERT_OK( TYEnableComponents(hDevice, componentIDs) );
    }
*/
    LOGD("=== Configure components");
    int32_t componentIDs = TY_COMPONENT_POINT3D_CAM | TY_COMPONENT_RGB_CAM;// 开启3d点云组件+RGB组件，相当于开启了所有
    ASSERT_OK( TYEnableComponents(hDevice, componentIDs) );
    


// 4. 配置参数 feature 分辨率等 =========================
    LOGD("=== Configure feature, set resolution to 640x480.");
    LOGD("Note: DM460 resolution feature is in component TY_COMPONENT_DEVICE,");
    LOGD("      other device may lays in some other components.");
    TY_FEATURE_INFO info;

    TY_STATUS ty_status = TYGetFeatureInfo(hDevice, TY_COMPONENT_DEPTH_CAM, TY_ENUM_IMAGE_MODE, &info);
    if ((info.accessMode & TY_ACCESS_WRITABLE) && (ty_status == TY_STATUS_OK)) 
    { 
      // 设置分辨率 深度图======
      int err = TYSetEnum(hDevice,            TY_COMPONENT_DEPTH_CAM, 
                          TY_ENUM_IMAGE_MODE, TY_IMAGE_MODE_640x480);
        ASSERT(err == TY_STATUS_OK || err == TY_STATUS_NOT_PERMITTED);   
    } 
    
    ty_status = TYGetFeatureInfo(hDevice, TY_COMPONENT_RGB_CAM, TY_ENUM_IMAGE_MODE, &info);
    if ((info.accessMode & TY_ACCESS_WRITABLE) && (ty_status == TY_STATUS_OK)) 
    { 
      // 设置分辨率 彩色图======
      int err = TYSetEnum(hDevice,            TY_COMPONENT_RGB_CAM, 
                          TY_ENUM_IMAGE_MODE, TY_IMAGE_MODE_640x480);
        ASSERT(err == TY_STATUS_OK || err == TY_STATUS_NOT_PERMITTED);   
    }

// 5. 分配内存空间
    LOGD("=== Prepare image buffer");
    
    // 查询当前配置下每个 framebuffer 的大小
    int32_t frameSize;
    //frameSize = 1280 * 960 * (3 + 2 + 2);// 彩色图 默认为 1280*960
    // 若配置为 640*480 则: frameSize = 640 * 480 * (3 + 2 + 2)
    ASSERT_OK( TYGetFrameBufferSize(hDevice, &frameSize) );
    LOGD("     - Get size of framebuffer, %d", frameSize);
    // ASSERT( frameSize >= 640 * 480 * 2 );
    
    // 分配 framebuffer 并压入驱动内的缓冲队列
    LOGD("     - Allocate & enqueue buffers");
    // char* frameBuffer[2];
    frameBuffer[0] = new char[frameSize];
    frameBuffer[1] = new char[frameSize];
    LOGD("     - Enqueue buffer (%p, %d)", frameBuffer[0], frameSize);
    ASSERT_OK( TYEnqueueBuffer(hDevice, frameBuffer[0], frameSize) );
    LOGD("     - Enqueue buffer (%p, %d)", frameBuffer[1], frameSize);
    ASSERT_OK( TYEnqueueBuffer(hDevice, frameBuffer[1], frameSize) );

// 6. 注册回调函数(主动获取模式下不调用)。
    LOGD("=== Register callback");
    LOGD("Note: Callback may block internal data receiving,");
    LOGD("      so that user should not do long time work in callback.");
    LOGD("      To avoid copying data, we pop the framebuffer from buffer queue and");
    LOGD("      give it back to user, user should call TYEnqueueBuffer to re-enqueue it.");
    DepthRender render;
    // CallbackData cb_data;
    cb_data.index = 0;
    cb_data.hDevice = hDevice;
    cb_data.render = &render;
    //ASSERT_OK( TYRegisterCallback(hDevice, frameHandler, &cb_data) );

    LOGD("=== Register event callback");
    LOGD("Note: Callback may block internal data receiving,");
    LOGD("      so that user should not do long time work in callback.");
    ASSERT_OK(TYRegisterEventCallback(hDevice, eventCallback, NULL));
    
    // 取消 主动发送数据模式
    LOGD("=== Disable trigger mode");
    ty_status = TYGetFeatureInfo(hDevice, TY_COMPONENT_DEVICE, TY_BOOL_TRIGGER_MODE, &info);
    if ((info.accessMode & TY_ACCESS_WRITABLE) && (ty_status == TY_STATUS_OK)) {
        ASSERT_OK(TYSetBool(hDevice, TY_COMPONENT_DEVICE, TY_BOOL_TRIGGER_MODE, false));
    }
    return 0;
}
