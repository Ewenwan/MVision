#include "../common/common.hpp"

static char buffer[1024 * 1024];// 存储 sdk 版本等信息
static int  n;// usb设备号
static volatile bool exit_main;

static int fps_counter = 0;
static clock_t fps_tm = 0;

const char* IP = NULL; // 指向只读字符的 指针IP 网络设备IP地址
TY_DEV_HANDLE hDevice; // 设备 handle
int32_t color = 1, ir = 0, depth = 0; // rgb图像 ir红外图像  depth深度图像
int i = 1; // 捕获图片index

char img_color_file_name[15]; 

char* frameBuffer[2]; // 帧 缓冲区

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

// 回调函数数据 结构体===================
struct CallbackData {
    int             index;
    TY_DEV_HANDLE   hDevice;
    DepthRender*    render;
};
CallbackData cb_data;     // 回调函数数据=====



void frameHandler(TY_FRAME_DATA* frame, void* userdata) {
    CallbackData* pData = (CallbackData*) userdata;
    // LOGD("=== Get frame %d", ++pData->index);// 帧id

    int ret = get_fps();
	if (ret > 0)
        printf("fps: %d\n", ret);

    // cv::Mat depth, irl, irr, color;
    // parseFrame(*frame, &depth, &irl, &irr, &color, 0);
    cv::Mat color;
    parseFrame(*frame, 0, 0, 0, &color, 0);
    //if(!depth.empty()){
    //    cv::Mat colorDepth = pData->render->Compute(depth);
    //    cv::imshow("ColorDepth", colorDepth);
    //}
    //if(!irl.empty()){ cv::imshow("LeftIR", irl); }
    //if(!irr.empty()){ cv::imshow("RightIR", irr); }
    // cv::namedWindow("Color", CV_WINDOW_NORMAL);//CV_WINDOW_NORMAL就是0
    if(!color.empty()){ cv::imshow("Color", color); }

    int key = cv::waitKey(1);
    switch(key & 0xff) 
    {
    case 0xff:
        break;
    case 'q':
        exit_main = true;
        break;
    case 's':// 保存图片================================================
        
	sprintf(img_color_file_name, "%s%02d%s","ty_rgb_", i++,".jpg");  
        LOGD("=== save img  %s", img_color_file_name);
	cv::imwrite(img_color_file_name, color); 
    default:
        LOGD("Unmapped key %d", key);
    }

    //LOGD("=== Callback: Re-enqueue buffer(%p, %d)", frame->userBuffer, frame->bufferSize);
    ASSERT_OK( TYEnqueueBuffer(pData->hDevice, frame->userBuffer, frame->bufferSize) );
}

// 事件 回调函数====================
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

// 相机初始化=======================
int ty_RGBD_init(void)
{
   cv::namedWindow("Color", CV_WINDOW_NORMAL);//CV_WINDOW_NORMAL就是0
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
    if(allComps & TY_COMPONENT_RGB_CAM  && color) 
    { // 使能 RGB 组件功能==
        LOGD("=== Has RGB camera, open RGB cam");
        ASSERT_OK( TYEnableComponents(hDevice, TY_COMPONENT_RGB_CAM) );
    }

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

// 4. 配置参数 feature 分辨率等 =========================
    LOGD("=== Configure feature, set resolution to 640x480.");
    LOGD("Note: DM460 resolution feature is in component TY_COMPONENT_DEVICE,");
    LOGD("      other device may lays in some other components.");
    TY_FEATURE_INFO info;
    TY_STATUS ty_status = TYGetFeatureInfo(hDevice, TY_COMPONENT_DEPTH_CAM, TY_ENUM_IMAGE_MODE, &info);

    if ((info.accessMode & TY_ACCESS_WRITABLE) && (ty_status == TY_STATUS_OK)) 
    {
        // 设置分辨率======
        int err = TYSetEnum(hDevice, TY_COMPONENT_DEPTH_CAM, TY_ENUM_IMAGE_MODE, TY_IMAGE_MODE_640x480);
        
        ASSERT(err == TY_STATUS_OK || err == TY_STATUS_NOT_PERMITTED);
    }

// 5. 分配内存空间
    LOGD("=== Prepare image buffer");
    
    // 查询当前配置下每个 framebuffer 的大小
    int32_t frameSize;
    ASSERT_OK( TYGetFrameBufferSize(hDevice, &frameSize) );
    LOGD("     - Get size of framebuffer, %d", frameSize);
    ASSERT( frameSize >= 640 * 480 * 2 );
    
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
    // ASSERT_OK( TYRegisterCallback(hDevice, frameHandler, &cb_data) );

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


}


int main(int argc, char* argv[]) {

   int init_flag = ty_RGBD_init();
   if(init_flag == -1) 
   {
	LOGD("=== camera init failed===");
        return -1;
   }
// 7. 启动采集=====
    LOGD("=== Start capture");
    ASSERT_OK( TYStartCapture(hDevice) );

    LOGD("=== While loop to fetch frame");
    exit_main = false;
    TY_FRAME_DATA frame; // 每一帧数据

    while(!exit_main) 
    {
// 8. 主动获取帧数据(主动获取模式下不调用)===========
        int err = TYFetchFrame(hDevice, &frame, -1);
        if( err != TY_STATUS_OK ) {
            LOGD("... Drop one frame");
        } 
        else 
        {
            frameHandler(&frame, &cb_data);
        }
    }
// 9. 停止采集=================
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
    return 0;
}
