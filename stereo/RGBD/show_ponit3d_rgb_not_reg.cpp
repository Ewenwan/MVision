#include <limits>
#include <cassert>
#include <cmath>
#include "../common/common.hpp"

static char buffer[1024*1024];
static int  n;
static volatile bool exit_main;

bool hasColor = false; // 无彩色图， 点云需要和彩色图配准
//static TY_CAMERA_INTRINSIC m_colorIntrinsic;// 彩色相机内参数
DepthViewer depthViewer;// 深度 信息 可视化器

// 用户数据 结构体
struct CallbackData {
    int             index;    // id
    TY_DEV_HANDLE   hDevice;  // 设备
    DepthRender*    render;   // 深度值 渲染器 例如得到彩色深度图
    PointCloudViewer* pcviewer;// 点云可视化器 pcl提供

    bool saveOneFramePoint3d;  // 保存点云
    int  fileIndex;            // 文件id

    TY_CAMERA_DISTORTION color_dist;// 畸变数据
    TY_CAMERA_INTRINSIC color_intri;// 内参数

    cv::Mat intrCvMat;              // 相机内参数矩阵
    cv::Mat distCvMat;              // 畸变参数矩阵
};
CallbackData cb_data;         // 用户数据

cv::Mat depth_img, color_img, p3d;
cv::Mat undistort_result;

// 深度信息 转到 点云坐标
cv::Point3f depthToWorld(float* intr, int x, int y, int z)
{
    cv::Point3f world;
    world.x = (x - intr[2]) * z / intr[0];// (x - cx)/fx
    world.y = (y - intr[5]) * z / intr[4];// (y - cy)/fy
    world.z = z;

    return world;
}

void undistort_RGB_img(cv::Mat& src_color, TY_CAMERA_INTRINSIC & color_intri, TY_CAMERA_DISTORTION & color_dist){

        // cv::Mat undistort_result(src_color.size(), CV_8UC3);
        undistort_result = cv::Mat(src_color.size(), CV_8UC3);// 矫正结果
        TY_IMAGE_DATA dst;        // 目标图像
        dst.width = src_color.cols;   // 宽度 列数
        dst.height = src_color.rows;  // 高度 行数
        dst.size = undistort_result.size().area() * 3;// 3通道 
        dst.buffer = undistort_result.data;
        dst.pixelFormat = TY_PIXEL_FORMAT_RGB; // RGB 格式
        TY_IMAGE_DATA src;        // 源图像=================
        src.width = src_color.cols;
        src.height = src_color.rows;
        src.size = src_color.size().area() * 3;
        src.pixelFormat = TY_PIXEL_FORMAT_RGB;
        src.buffer = src_color.data; 
        //undistort camera image 
        //TYUndistortImage accept TY_IMAGE_DATA from TY_FRAME_DATA , pixel format RGB888 or MONO8
        //you can also use opencv API cv::undistort to do this job.
        ASSERT_OK(TYUndistortImage(&color_intri, &color_dist, NULL, &src, &dst));
        src_color = undistort_result;// 畸变矫正后的图像==========================
        //cv::Mat resizedColor;// 彩色图缩放到 和 深度图一样大
        //cv::resize(color, resizedColor, depth.size(), 0, 0, CV_INTER_LINEAR);

}



void frameHandler(TY_FRAME_DATA* frame, void* userdata)
{
    CallbackData* pData = (CallbackData*) userdata; // 用户数据
    LOGD("=== Get frame %d", ++pData->index);

    // cv::Mat depth, color, p3d;
    parseFrame(*frame, &depth_img, 0, 0, &color_img, &p3d);
    if(pData->saveOneFramePoint3d)
    {
        char file[32];
        sprintf(file, "points-%d.xyz", pData->fileIndex++);
        writePointCloud((cv::Point3f*)p3d.data, p3d.total(), file, PC_FILE_FORMAT_XYZ);// 保存点云数据
        pData->saveOneFramePoint3d = false;
        imshow("depth", depth_img * 32);
    }

    if(!depth_img.empty()){
        depthViewer.show("depth12", depth_img);// depthViewer 显示 彩色点云数据 包含中心点的距离
        imshow("depth", depth_img*32);// 普通 深度图
    }
    if(!color_img.empty()){ 
        // 矫正彩色图 
        //cv::Mat temp = color.clone();
        //cv::undistort(temp, color, pData->intrCvMat, pData->distCvMat); // 有问题

        // undistort_RGB_img(color_img, pData->color_intri, pData->color_dist);

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


        imshow("Color", color_img); // 显示彩色图像
    }
    if(!p3d.empty()){
        pData->pcviewer->show(p3d, "Point3D"); // 显示点云数据
        // LOGD("point3d w:%d, h:%d, num: %lu", p3d.rows, p3d.cols, p3d.total()); // 宽480 高640 
        if(pData->pcviewer->isStopped("Point3D")){// 点云界面被关闭===
            exit_main = true; // 退出
            return;
        }
    }

    int key = cv::waitKey(100);
    switch(key & 0xff){
        case 0xff:
            break;
        case 'q':
            exit_main = true;
            break;
        case 's':
            pData->saveOneFramePoint3d = true;
            break;
        default:
            LOGD("Pressed key %d", key);
    }

    //LOGD("=== Callback: Re-enqueue buffer(%p, %d)", frame->userBuffer, frame->bufferSize);
    ASSERT_OK( TYEnqueueBuffer(pData->hDevice, frame->userBuffer, frame->bufferSize) );
}

int main(int argc, char* argv[])
{
// 1. 确定设备id========
    const char* IP = NULL;
    const char* ID = NULL;
    const char* file = NULL;
    TY_DEV_HANDLE hDevice;

    for(int i = 1; i < argc; i++){
        if(strcmp(argv[i], "-id") == 0){
            ID = argv[++i];
        }else if(strcmp(argv[i], "-ip") == 0){
            IP = argv[++i];
        }else if(strcmp(argv[i], "-h") == 0){
            LOGI("Usage: SimpleView_Callback [-h] [-ip <IP>] [-id <ID>]");
            return 0;
        }
    }

// 2. 初始化 api======
    LOGD("=== Init lib");
    ASSERT_OK( TYInitLib() );
    TY_VERSION_INFO* pVer = (TY_VERSION_INFO*)buffer;
    ASSERT_OK( TYLibVersion(pVer) );
    LOGD("     - lib version: %d.%d.%d", pVer->major, pVer->minor, pVer->patch);

// 3. 打开设备=======
    if(IP) {
        LOGD("=== Open device %s", IP);
        ASSERT_OK( TYOpenDeviceWithIP(IP, &hDevice) );
    } else if (ID){
        LOGD("=== Open device %s", ID);
        ASSERT_OK( TYOpenDevice(ID, &hDevice) );
    } else {
        LOGD("=== Get device info");
        ASSERT_OK( TYGetDeviceNumber(&n) );
        LOGD("     - device number %d", n);

        TY_DEVICE_BASE_INFO* pBaseInfo = (TY_DEVICE_BASE_INFO*)buffer;
        ASSERT_OK( TYGetDeviceList(pBaseInfo, 100, &n) );

        if(n == 0){
            LOGD("=== No device got");
            return -1;
        }

        LOGD("=== Open device 0");
        ASSERT_OK( TYOpenDevice(pBaseInfo[0].id, &hDevice) );
    }

// 4. 配置3d 点云组件，未使用彩色相机
    LOGD("=== Configure components, open point3d cam");
    // int32_t componentIDs = TY_COMPONENT_POINT3D_CAM;
    int32_t componentIDs = TY_COMPONENT_POINT3D_CAM | TY_COMPONENT_RGB_CAM;// 使用彩色相机
    ASSERT_OK( TYEnableComponents(hDevice, componentIDs) );

    // CallbackData cb_data;      // 用户数据

    int32_t allComps=0;
    ASSERT_OK( TYGetComponentIDs(hDevice, &allComps) );
    if(allComps & TY_COMPONENT_RGB_CAM)
    {
        TY_CAMERA_DISTORTION color_dist;// 相机 畸变参数
        TY_CAMERA_INTRINSIC color_intri;// 相机 内参数
        // 得到彩色 相机内参数=====
        TY_STATUS ret = TYGetStruct(hDevice, TY_COMPONENT_RGB_CAM, 
                                     TY_STRUCT_CAM_INTRINSIC, (void*)&color_intri, sizeof(color_intri));
                  ret |= TYGetStruct(hDevice, TY_COMPONENT_RGB_CAM, 
                                     TY_STRUCT_CAM_INTRINSIC, &color_intri, sizeof(color_intri));
    
        if(ret != TY_STATUS_OK)
        { 
            LOGE("Get camera RGB intrinsic failed: %s", TYErrorString(ret));
            // 需要设置默认 相机参数
        } 
        else 
        {
            hasColor = true;
            cb_data.color_intri = color_intri;// 相机 内参数
           // cb_data.intrCvMat = (cv::Mat_<float>(3,3) << color_intri.data[0], color_intri.data[1],
           //                                              color_intri.data[2], color_intri.data[3],
            //                                             color_intri.data[4], color_intri.data[5],
           //                                              color_intri.data[6], color_intri.data[7], color_intri.data[8]);
            cb_data.color_dist= color_dist;   // 相机 畸变参数
            //cb_data.distCvMat = (cv::Mat_<float>(5,1) << color_dist.data[0], color_dist.data[1],
            //                                             color_dist.data[2], color_dist.data[3],
            //                                            color_dist.data[4]);  
        }
    }

// 5. 配置深度图 分辨率===================
    LOGD("=== Configure feature, set resolution to 640x480.");
    LOGD("Note: DM460 resolution feature is in component TY_COMPONENT_DEVICE,");
    LOGD("      other device may lays in some other components.");
    TY_STATUS err = TYSetEnum(hDevice, TY_COMPONENT_DEPTH_CAM, TY_ENUM_IMAGE_MODE, TY_IMAGE_MODE_640x480);
    ASSERT(err == TY_STATUS_OK || err == TY_STATUS_NOT_PERMITTED);

    err = TYSetEnum(hDevice, TY_COMPONENT_RGB_CAM, TY_ENUM_IMAGE_MODE, TY_IMAGE_MODE_640x480);
    ASSERT(err == TY_STATUS_OK || err == TY_STATUS_NOT_PERMITTED);

// 6. 准备缓冲区========================
    LOGD("=== Prepare image buffer");
    int32_t frameSize;
    ASSERT_OK( TYGetFrameBufferSize(hDevice, &frameSize) );
    LOGD("     - Get size of framebuffer, %d", frameSize);
    ASSERT( frameSize >= 640*480*2 );

    LOGD("     - Allocate & enqueue buffers");
    char* frameBuffer[2];
    frameBuffer[0] = new char[frameSize];
    frameBuffer[1] = new char[frameSize];
    LOGD("     - Enqueue buffer (%p, %d)", frameBuffer[0], frameSize);
    ASSERT_OK( TYEnqueueBuffer(hDevice, frameBuffer[0], frameSize) );
    LOGD("     - Enqueue buffer (%p, %d)", frameBuffer[1], frameSize);
    ASSERT_OK( TYEnqueueBuffer(hDevice, frameBuffer[1], frameSize) );

// 7. 注册回调函数======================
    LOGD("=== Register callback");
    LOGD("Note: Callback may block internal data receiving,");
    LOGD("      so that user should not do long time work in callback.");
    LOGD("      To avoid copying data, we pop the framebuffer from buffer queue and");
    LOGD("      give it back to user, user should call TYEnqueueBuffer to re-enqueue it.");
    DepthRender render;        // 深度值渲染器
    PointCloudViewer pcviewer; // 点云可视化器
    // CallbackData cb_data;      // 用户数据
    cb_data.index = 0;
    cb_data.hDevice = hDevice;
    cb_data.render = &render;
    cb_data.pcviewer = &pcviewer;
    cb_data.saveOneFramePoint3d = false;// 不保存点云
    cb_data.fileIndex = 0;
    // ASSERT_OK( TYRegisterCallback(hDevice, frameHandler, &cb_data) );

// 8. 关闭触发模式======
    LOGD("=== Disable trigger mode");
    ASSERT_OK( TYSetBool(hDevice, TY_COMPONENT_DEVICE, TY_BOOL_TRIGGER_MODE, false) );

// 9. 开始捕获=======
    LOGD("=== Start capture");
    ASSERT_OK( TYStartCapture(hDevice) );

    LOGD("=== While loop to fetch frame");
    exit_main = false;
    TY_FRAME_DATA frame;

    while(!exit_main){
        int err = TYFetchFrame(hDevice, &frame, -1);// 获取一帧数据
        if( err != TY_STATUS_OK ){
            LOGD("... Drop one frame");
            continue;
        }

        frameHandler(&frame, &cb_data);// 解析数据 =====
    }

    ASSERT_OK( TYStopCapture(hDevice) );
    ASSERT_OK( TYCloseDevice(hDevice) );
    ASSERT_OK( TYDeinitLib() );
    // MSLEEP(10); // sleep to ensure buffer is not used any more
    delete frameBuffer[0];
    delete frameBuffer[1];

    LOGD("=== Main done!");
    return 0;
}
