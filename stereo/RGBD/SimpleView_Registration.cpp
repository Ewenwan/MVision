#include "../common/common.hpp"

static char buffer[1024*1024*20];
static int  n;
static volatile bool exit_main;
static volatile bool save_frame;

// 用户数据
struct CallbackData {
    int             index;    // index
    TY_DEV_HANDLE   hDevice;  // 设备
    DepthRender*    render;   // 

    TY_CAMERA_DISTORTION color_dist;// 畸变数据
    TY_CAMERA_INTRINSIC color_intri;// 内参数
};

void handleFrame(TY_FRAME_DATA* frame, void* userdata)
{
    CallbackData* pData = (CallbackData*) userdata;
    // LOGD("=== Get frame %d", ++pData->index); // 帧+1

    cv::Mat depth, irl, irr, color, point3D;
    parseFrame(*frame, &depth, &irl, &irr, &color, &point3D);
    if(!depth.empty())
    {
        cv::Mat colorDepth = pData->render->Compute(depth);
        cv::imshow("ColorDepth", colorDepth); // 未配准的 深度图
    }
    if(!irl.empty()){ cv::imshow("LeftIR", irl); }
    if(!irr.empty()){ cv::imshow("RightIR", irr); }

// 矫正彩色图像=======
    if(!color.empty())
    {
        cv::Mat undistort_result(color.size(), CV_8UC3);
        TY_IMAGE_DATA dst;        // 目标图像
        dst.width = color.cols;   // 宽度 列数
        dst.height = color.rows;  // 高度 行数
        dst.size = undistort_result.size().area() * 3;// 3通道 
        dst.buffer = undistort_result.data;
        dst.pixelFormat = TY_PIXEL_FORMAT_RGB; // RGB 格式
        TY_IMAGE_DATA src;        // 源图像=================
        src.width = color.cols;
        src.height = color.rows;
        src.size = color.size().area() * 3;
        src.pixelFormat = TY_PIXEL_FORMAT_RGB;
        src.buffer = color.data; 
        //undistort camera image 
        //TYUndistortImage accept TY_IMAGE_DATA from TY_FRAME_DATA , pixel format RGB888 or MONO8
        //you can also use opencv API cv::undistort to do this job.
        ASSERT_OK(TYUndistortImage(&pData->color_intri, &pData->color_dist, NULL, &src, &dst));
        color = undistort_result;// 畸变矫正后的图像==========================

        cv::Mat resizedColor;// 彩色图缩放到 和 深度图一样大
        cv::resize(color, resizedColor, depth.size(), 0, 0, CV_INTER_LINEAR);
        cv::imshow("color", resizedColor);
    }

// 彩色图和深度图配准=============
    // do Registration
    cv::Mat newDepth;
    if(!point3D.empty() && !color.empty()) 
    {
        ASSERT_OK( TYRegisterWorldToColor2(pData->hDevice, (TY_VECT_3F*)point3D.data, 0, 
                   point3D.cols * point3D.rows, color.cols, color.rows, (uint16_t*)buffer, sizeof(buffer)
                    ));
        newDepth = cv::Mat(color.rows, color.cols, CV_16U, (uint16_t*)buffer);
        cv::Mat resized_color;
        cv::Mat temp;
        //you may want to use median filter to fill holes in projected depth image or do something else here
        cv::medianBlur(newDepth,temp,5);// 对3d点云反投影到 彩色图上 获取的带有 孔洞的 深度图 进行均值滤波======
        newDepth = temp;
        //resize to the same size for display
        cv::resize(newDepth, newDepth, depth.size(), 0, 0, 0);// 深度图
        cv::resize(color, resized_color, depth.size());// 彩色图

        cv::Mat depthColor = pData->render->Compute(newDepth);// 彩色深度图
        cv::imshow("Registrated ColorDepth", depthColor);// 显示 

        depthColor = depthColor / 2 + resized_color / 2; // c彩色深度图 和 彩色图合并在一起
        cv::imshow("projected depth", depthColor);// 显示 
    }

    if(save_frame){
        LOGD(">>>>>>>>>> write images");
        imwrite("depth.png", newDepth);
        imwrite("color.png", color);
        save_frame = false;
    }

    int key = cv::waitKey(1);
    switch(key){
        case -1:
            break;
        case 'q': 
        case 1048576 + 'q':
            exit_main = true;
            break;
        case 's': 
        case 1048576 + 's':
            save_frame = true;
            break;
        default:
            LOGD("Pressed key %d", key);
    }

   // LOGD("=== Callback: Re-enqueue buffer(%p, %d)", frame->userBuffer, frame->bufferSize);
    ASSERT_OK( TYEnqueueBuffer(pData->hDevice, frame->userBuffer, frame->bufferSize) );
}

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


int main(int argc, char* argv[])
{
    const char* IP = NULL;
    const char* ID = NULL;
    TY_DEV_HANDLE hDevice;
// 1. 网络设备IP====
    for(int i = 1; i < argc; i++){
        if(strcmp(argv[i], "-id") == 0){
            ID = argv[++i];
        }else if(strcmp(argv[i], "-ip") == 0){
            IP = argv[++i];
        }else if(strcmp(argv[i], "-h") == 0){
            LOGI("Usage: SimpleView_Callback [-h] [-ip <IP>]");
            return 0;
        }
    }
// 2. 初始化API===
    LOGD("=== Init lib");
    ASSERT_OK( TYInitLib() );
    TY_VERSION_INFO* pVer = (TY_VERSION_INFO*)buffer;
    ASSERT_OK( TYLibVersion(pVer) );
    LOGD("     - lib version: %d.%d.%d", pVer->major, pVer->minor, pVer->patch);
// 3. 打开本地设备/网络设备 ====
    if(IP) {
        LOGD("=== Open device %s", IP);
        ASSERT_OK( TYOpenDeviceWithIP(IP, &hDevice) );
    } else {
        if(ID == NULL){
            LOGD("=== Get device info");
            ASSERT_OK( TYGetDeviceNumber(&n) );
            LOGD("     - device number %d", n);

            TY_DEVICE_BASE_INFO* pBaseInfo = (TY_DEVICE_BASE_INFO*)buffer;
            ASSERT_OK( TYGetDeviceList(pBaseInfo, 100, &n) );

            if(n == 0){
                LOGD("=== No device got");
                return -1;
            }
            ID = pBaseInfo[0].id;
        }

        LOGD("=== Open device: %s", ID);
        ASSERT_OK( TYOpenDevice(ID, &hDevice) );
    }

// 4. 配置组件 Component=======
    int32_t allComps;
    ASSERT_OK( TYGetComponentIDs(hDevice, &allComps) );
    if(!(allComps & TY_COMPONENT_RGB_CAM)){
        LOGE("=== Has no RGB camera, cant do registration");
        return -1;
    }

    LOGD("=== Configure components");
    int32_t componentIDs = TY_COMPONENT_POINT3D_CAM | TY_COMPONENT_RGB_CAM;
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
    
    LOGD("=== Prepare image buffer");
    int32_t frameSize;

    //frameSize = 1280 * 960 * (3 + 2 + 2);// 彩色图 默认为 1280*960
    // 若配置为 640*480 则: frameSize = 640 * 480 * (3 + 2 + 2)
    ASSERT_OK( TYGetFrameBufferSize(hDevice, &frameSize) );
    LOGD("     - Get size of framebuffer, %d", frameSize);
    LOGD("     - Allocate & enqueue buffers");
    char* frameBuffer[2];
    frameBuffer[0] = new char[frameSize];
    frameBuffer[1] = new char[frameSize];
    LOGD("     - Enqueue buffer (%p, %d)", frameBuffer[0], frameSize);
    ASSERT_OK( TYEnqueueBuffer(hDevice, frameBuffer[0], frameSize) );
    LOGD("     - Enqueue buffer (%p, %d)", frameBuffer[1], frameSize);
    ASSERT_OK( TYEnqueueBuffer(hDevice, frameBuffer[1], frameSize) );
// 5. 配置回调函数====
    LOGD("=== Register callback");
    LOGD("Note: Callback may block internal data receiving,");
    LOGD("      so that user should not do long time work in callback.");
    LOGD("      To avoid copying data, we pop the framebuffer from buffer queue and");
    LOGD("      give it back to user, user should call TYEnqueueBuffer to re-enqueue it.");
    DepthRender render;
    CallbackData cb_data; // 回调数据====
    cb_data.index = 0;
    cb_data.hDevice = hDevice;
    cb_data.render = &render;
    // ASSERT_OK( TYRegisterCallback(hDevice, frameCallback, &cb_data) );

    LOGD("=== Register event callback");
    LOGD("Note: Callback may block internal data receiving,");
    LOGD("      so that user should not do long time work in callback.");
    ASSERT_OK(TYRegisterEventCallback(hDevice, eventCallback, NULL));

// 6. 关闭触发模式=====
    LOGD("=== Disable trigger mode");
    ASSERT_OK( TYSetBool(hDevice, TY_COMPONENT_DEVICE, TY_BOOL_TRIGGER_MODE, false) );
// 7. 启动采集
    LOGD("=== Start capture");
    ASSERT_OK( TYStartCapture(hDevice) );

    LOGD("=== Read color rectify matrix");
    {
        TY_CAMERA_DISTORTION color_dist;// 相机 畸变参数
        TY_CAMERA_INTRINSIC color_intri;// 相机 内参数
        TY_STATUS ret = TYGetStruct(hDevice, TY_COMPONENT_RGB_CAM, TY_STRUCT_CAM_DISTORTION, &color_dist, sizeof(color_dist));
        ret |= TYGetStruct(hDevice, TY_COMPONENT_RGB_CAM, TY_STRUCT_CAM_INTRINSIC, &color_intri, sizeof(color_intri));
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
        int i;
        // 打印 畸变参数 
        std::cout << " camera distortion param: ";
        for (i=0; i<12; i++){
            std::cout << cb_data.color_dist.data[i] << " ";
         }
        std::cout << std::endl;
        // 打印 内参数 
        std::cout << " camera intrinsic param: ";
        for (i=0; i<9; i++){
            std::cout << cb_data.color_intri.data[i] << " ";
         }
        std::cout << std::endl;
    }

    LOGD("=== Wait for callback");
    exit_main = false;
    while(!exit_main){
        TY_FRAME_DATA frame;
        int err = TYFetchFrame(hDevice, &frame, -1);// 捕获 一帧数据
        if( err != TY_STATUS_OK ) 
        {
            LOGE("Fetch frame error %d: %s", err, TYErrorString(err));
            break;
        } 
        else {
            handleFrame(&frame, &cb_data);// 传入用户数据
        }
    }

    ASSERT_OK( TYStopCapture(hDevice) );// 停止采集
    ASSERT_OK( TYCloseDevice(hDevice) );// 关闭设备
    ASSERT_OK( TYDeinitLib() );         // 关闭API
    delete frameBuffer[0];
    delete frameBuffer[1];

    LOGD("=== Main done!");
    return 0;
}
