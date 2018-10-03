/**
 * @file      TY_API.h
 * @brief     API for Percipio depth cameras.
 *
 * Copyright(C)2016 Percipio All Rights Reserved
 */

// README:
//------------------------------------------------------------------------------
//
// Depth camera, called "device", consists of several components. Each component
// is a hardware module or virtual module, such as RGB sensor, depth sensor.
// Each component has its own features, such as image resolution, pixel format.
//
// NOTE: The component TY_COMPONENT_DEVICE is a virtual component that contains
//       all features related to the whole device, such as trigger mode, device IP.
//
// Each frame consists of several images. Normally, all the images have identical
// timestamp, means they are captured at the same time.
//
//------------------------------------------------------------------------------

#ifndef TY_API_H_
#define TY_API_H_

#include <stddef.h>
#include <stdlib.h>

#ifdef WIN32
#  ifndef _WIN32
#    define _WIN32
#  endif
#endif

#ifdef _WIN32
# ifndef _STDINT_H
#  if defined(_MSC_VER) && _MSC_VER < 1600
    typedef __int8            int8_t;
    typedef __int16           int16_t;
    typedef __int32           int32_t;
    typedef __int64           int64_t;
    typedef unsigned __int8   uint8_t;
    typedef unsigned __int16  uint16_t;
    typedef unsigned __int32  uint32_t;
    typedef unsigned __int64  uint64_t;
#  else
#   include <stdint.h>
#  endif
# endif
#else
# include <stdint.h>
#endif

// copy stdbool.h here in case bool not defined or <stdbool.h> cant be found
#ifndef _STDBOOL_H
# define _STDBOOL_H
# define __bool_true_false_are_defined	1
# ifndef __cplusplus
#  define bool  _Bool
#  define true  1
#  define false 0
# endif
#endif

#ifdef _WIN32
#  include <Windows.h>
#ifdef TY_WIN32_BUILD_STATIC
#  define TY_DLLIMPORT
#  define TY_DLLEXPORT
#else
#  define TY_DLLIMPORT      __declspec(dllimport)
#  define TY_DLLEXPORT      __declspec(dllexport)
#endif
#  define TY_STDC           __stdcall
#  define TY_CDEC           __cdecl
#else
#  define TY_DLLIMPORT      __attribute__((visibility("default")))
#  define TY_DLLEXPORT      __attribute__((visibility("default")))
#  if defined(__i386__)
#    define TY_STDC         __attribute__((stdcall))
#    define TY_CDEC         __attribute__((cdecl))
#  else
#    define TY_STDC
#    define TY_CDEC
#  endif
#endif

#ifdef TY_BUILD_DLL
#  define TY_EXPORT     TY_DLLEXPORT
#else
#  define TY_EXPORT     TY_DLLIMPORT
#endif

#if defined(__cplusplus)
#  define TY_EXTC extern "C"
#else
#  define TY_EXTC
#endif


//------------------------------------------------------------------------------
//  Definitions
//------------------------------------------------------------------------------
#define TY_LIB_VERSION_MAJOR       2
#define TY_LIB_VERSION_MINOR       6
#define TY_LIB_VERSION_PATCH       10 


//------------------------------------------------------------------------------
//  Status
//------------------------------------------------------------------------------
typedef enum TY_STATUS_LIST
{
    TY_STATUS_OK                = 0,
    TY_STATUS_ERROR             = -1001,
    TY_STATUS_NOT_INITED        = -1002,
    TY_STATUS_NOT_IMPLEMENTED   = -1003,
    TY_STATUS_NOT_PERMITTED     = -1004,
    TY_STATUS_DEVICE_ERROR      = -1005,
    TY_STATUS_INVALID_PARAMETER = -1006,
    TY_STATUS_INVALID_HANDLE    = -1007,
    TY_STATUS_INVALID_COMPONENT = -1008,
    TY_STATUS_INVALID_FEATURE   = -1009,
    TY_STATUS_WRONG_TYPE        = -1010,
    TY_STATUS_WRONG_SIZE        = -1011,
    TY_STATUS_OUT_OF_MEMORY     = -1012,
    TY_STATUS_OUT_OF_RANGE      = -1013,
    TY_STATUS_TIMEOUT           = -1014,
    TY_STATUS_WRONG_MODE        = -1015,
    TY_STATUS_BUSY              = -1016,
    TY_STATUS_IDLE              = -1017,
    TY_STATUS_NO_DATA           = -1018,
    TY_STATUS_NO_BUFFER         = -1019,
    TY_STATUS_NULL_POINTER      = -1020,
    TY_STATUS_READONLY_FEATURE  = -1021
}TY_STATUS_LIST;
typedef int32_t TY_STATUS;

typedef enum TY_EVENT_LIST
{
    TY_EVENT_DEVICE_OFFLINE     = -2001,
    TY_EVENT_LICENSE_ERROR      = -2002
}TY_ENENT_LIST;
typedef int32_t TY_EVENT;

//------------------------------------------------------------------------------
//  Device Handle
//------------------------------------------------------------------------------
typedef void* TY_DEV_HANDLE;


//------------------------------------------------------------------------------
//  Device Component
//------------------------------------------------------------------------------
typedef enum TY_DEVICE_COMPONENT_LIST
{
    TY_COMPONENT_DEVICE         = 0x80000000, ///< Abstract component stands for whole device, always enabled
    TY_COMPONENT_DEPTH_CAM      = 0x00010000, ///< Depth camera
    TY_COMPONENT_POINT3D_CAM    = 0x00020000, ///< Point3D camera
    TY_COMPONENT_IR_CAM_LEFT    = 0x00040000, ///< Left IR camera
    TY_COMPONENT_IR_CAM_RIGHT   = 0x00080000, ///< Right IR camera
    TY_COMPONENT_RGB_CAM_LEFT   = 0x00100000, ///< Left RGB camera
    TY_COMPONENT_RGB_CAM_RIGHT  = 0x00200000, ///< Right RGB camera
    TY_COMPONENT_LASER          = 0x00400000, ///< Laser
    TY_COMPONENT_IMU            = 0x00800000, ///< Inertial Measurement Unit
    TY_COMPONENT_BRIGHT_HISTO   = 0x01000000, ///< virtual component for brightness histogram of ir

    TY_COMPONENT_RGB_CAM        = TY_COMPONENT_RGB_CAM_LEFT /// Some device has only one RGB camera, map it to left
}TY_DEVICE_COMPONENT_LIST;
typedef int32_t TY_COMPONENT_ID;


//------------------------------------------------------------------------------
//  Feature
//------------------------------------------------------------------------------
typedef enum TY_FEATURE_TYPE_LIST
{
    TY_FEATURE_INT              = 0x1000,
    TY_FEATURE_FLOAT            = 0X2000,
    TY_FEATURE_ENUM             = 0x3000,
    TY_FEATURE_BOOL             = 0x4000,
    TY_FEATURE_STRING           = 0x5000,
    TY_FEATURE_BYTEARRAY        = 0x6000,
    TY_FEATURE_STRUCT           = 0x7000,
}TY_FEATURE_TYPE_LIST;
typedef int32_t TY_FEATURE_TYPE;


typedef enum TY_FEATURE_ID_LIST
{
    TY_STRUCT_CAM_INTRINSIC         = 0x000 | TY_FEATURE_STRUCT, ///< see TY_CAMERA_INTRINSIC
    TY_STRUCT_EXTRINSIC_TO_LEFT_IR  = 0x001 | TY_FEATURE_STRUCT, ///< extrinsic from current component to left IR, see TY_CAMERA_EXTRINSIC
    TY_STRUCT_EXTRINSIC_TO_LEFT_RGB = 0x002 | TY_FEATURE_STRUCT, ///< extrinsic from current component to left RGB, see TY_CAMERA_EXTRINSIC
    TY_STRUCT_NET_INFO              = 0x005 | TY_FEATURE_STRUCT, ///< see TY_DEVICE_NET_INFO
    TY_STRUCT_CAM_DISTORTION        = 0x006 | TY_FEATURE_STRUCT, ///< see TY_CAMERA_DISTORTION

    TY_INT_WIDTH_MAX            = 0x100 | TY_FEATURE_INT,
    TY_INT_HEIGHT_MAX           = 0x101 | TY_FEATURE_INT,
    TY_INT_OFFSET_X             = 0x102 | TY_FEATURE_INT,
    TY_INT_OFFSET_Y             = 0x103 | TY_FEATURE_INT,
    TY_INT_WIDTH                = 0x104 | TY_FEATURE_INT,
    TY_INT_HEIGHT               = 0x105 | TY_FEATURE_INT,
    TY_INT_IMAGE_SIZE           = 0x106 | TY_FEATURE_INT,
    TY_ENUM_PIXEL_FORMAT        = 0x107 | TY_FEATURE_ENUM, ///< Pixel format, see TY_PIXEL_FORMAT_LIST
    TY_ENUM_IMAGE_MODE          = 0x108 | TY_FEATURE_ENUM, ///< Pixel format, see TY_IMAGE_MODE_LIST

    TY_BOOL_TRIGGER_MODE        = 0x200 | TY_FEATURE_BOOL, ///< Trigger mode switch
    TY_ENUM_TRIGGER_ACTIVATION  = 0x201 | TY_FEATURE_ENUM, ///< Trigger activation, see TY_TRIGGER_ACTIVATION_LIST
    TY_INT_FRAME_PER_TRIGGER    = 0x202 | TY_FEATURE_INT,  ///< Number of frames captured per trigger
    TY_BOOL_KEEP_ALIVE_ONOFF    = 0x203 | TY_FEATURE_BOOL, ///< Keep Alive switch
    TY_INT_KEEP_ALIVE_TIMEOUT   = 0x204 | TY_FEATURE_INT,  ///< Keep Alive timeout

    TY_BOOL_AUTO_EXPOSURE       = 0x300 | TY_FEATURE_BOOL, ///< Auto exposure switch
    TY_INT_EXPOSURE_TIME        = 0x301 | TY_FEATURE_INT,  ///< Exposure time in percentage
    TY_BOOL_AUTO_GAIN           = 0x302 | TY_FEATURE_BOOL, ///< Auto gain switch
    TY_INT_GAIN                 = 0x303 | TY_FEATURE_INT,  ///< Gain
    TY_BOOL_AUTO_AWB            = 0x304 | TY_FEATURE_BOOL, ///< Auto white balance

    TY_INT_LASER_POWER          = 0x500 | TY_FEATURE_INT,  ///< Laser power level
    TY_BOOL_LASER_AUTO_CTRL     = 0x501 | TY_FEATURE_BOOL,  ///< Laser auto ctrl

    TY_BOOL_UNDISTORTION        = 0x510 | TY_FEATURE_BOOL, ///< Output undistorted image
    TY_BOOL_BRIGHTNESS_HISTOGRAM    = 0x511 | TY_FEATURE_BOOL, ///< Output bright histogram

    TY_INT_R_GAIN               = 0x520 | TY_FEATURE_INT,  ///< Gain of R channel
    TY_INT_G_GAIN               = 0x521 | TY_FEATURE_INT,  ///< Gain of G channel
    TY_INT_B_GAIN               = 0x522 | TY_FEATURE_INT,  ///< Gain of B channel

    TY_STRUCT_WORK_MODE         = 0x523 | TY_FEATURE_STRUCT,  ///< mode of trigger

    TY_INT_ANALOG_GAIN          = 0x524 | TY_FEATURE_INT,  ///< Analog gain
    TY_INT_RGB_ANALOG_GAIN      = 0x525 | TY_FEATURE_INT,  ///< RGB Analog gain
}TY_FEATURE_ID_LIST;
typedef int32_t TY_FEATURE_ID;


typedef enum TY_IMAGE_MODE_LIST
{
    TY_IMAGE_MODE_160x120       = (160<<12)+120, ///< 655480 
    TY_IMAGE_MODE_320x240       = (320<<12)+240, ///< 1310960
    TY_IMAGE_MODE_640x480       = (640<<12)+480, ///< 2621920
    TY_IMAGE_MODE_1280x960      = (1280<<12)+960,///< 5243840
    TY_IMAGE_MODE_2592x1944     = (2592<<12)+1944,///< 10618776
}TY_IMAGE_MODE_LIST;
typedef int32_t TY_IMAGE_MODE;


typedef enum TY_TRIGGER_ACTIVATION_LIST
{
    TY_TRIGGER_ACTIVATION_FALLINGEDGE = 0,
    TY_TRIGGER_ACTIVATION_RISINGEDGE  = 1,
}TY_TRIGGER_ACTIVATION_LIST;
typedef int32_t TY_TRIGGER_ACTIVATION;


typedef enum TY_INTERFACE_LIST
{
    TY_INTERFACE_UNKNOWN        = 0,
    TY_INTERFACE_ETHERNET       = 1,
    TY_INTERFACE_USB            = 2,
}TY_INTERFACE_LIST;
typedef int32_t TY_INTERFACE;


typedef enum TY_ACCESS_MODE_LIST
{
    TY_ACCESS_READABLE          = 0x1,
    TY_ACCESS_WRITABLE          = 0x2,
}TY_ACCESS_MODE_LIST;


//------------------------------------------------------------------------------
//  Pixel
//------------------------------------------------------------------------------
typedef enum TY_PIXEL_TYPE_LIST
{
    TY_PIXEL_MONO               = 0x10000000,
    TY_PIXEL_COLOR              = 0x20000000,
    TY_PIXEL_DEPTH              = 0x30000000,
    TY_PIXEL_POINT3D            = 0x40000000,
}TY_PIXEL_TYPE_LIST;


typedef enum TY_PIXEL_BITS_LIST{
    TY_PIXEL_8BIT               = 0x00080000,
    TY_PIXEL_16BIT              = 0x00100000,
    TY_PIXEL_24BIT              = 0x00180000,
    TY_PIXEL_32BIT              = 0x00200000,
    TY_PIXEL_96BIT              = 0x00600000,
}TY_PIXEL_BITS_LIST;


typedef enum TY_PIXEL_FORMAT_LIST
{
    TY_PIXEL_FORMAT_UNDEFINED   = 0,
    TY_PIXEL_FORMAT_MONO        = (TY_PIXEL_MONO    | TY_PIXEL_8BIT  | 0x0000), //0x10080000
    TY_PIXEL_FORMAT_RGB         = (TY_PIXEL_COLOR   | TY_PIXEL_24BIT | 0x0010), //0x20180010
    TY_PIXEL_FORMAT_YUV422      = (TY_PIXEL_COLOR   | TY_PIXEL_16BIT | 0x0011), //0x20100011, YVYU
    TY_PIXEL_FORMAT_YVYU        =  TY_PIXEL_FORMAT_YUV422                     , //0x20100011, YVYU
    TY_PIXEL_FORMAT_YUYV        = (TY_PIXEL_COLOR   | TY_PIXEL_16BIT | 0x0012), //0x20100012, YUYV
    TY_PIXEL_FORMAT_JPEG        = (TY_PIXEL_COLOR   | TY_PIXEL_24BIT | 0x0013), //0x20180013, JPEG
    TY_PIXEL_FORMAT_DEPTH16     = (TY_PIXEL_DEPTH   | TY_PIXEL_16BIT | 0x0020), //0x30100020
    TY_PIXEL_FORMAT_FPOINT3D    = (TY_PIXEL_POINT3D | TY_PIXEL_96BIT | 0x0030), //0x40600030
    TY_PIXEL_FORMAT_BAYER8GB         = (TY_PIXEL_MONO    | TY_PIXEL_8BIT  | 0x0090), //0x10080090
}TY_PIXEL_FORMAT_LIST;
typedef int32_t TY_PIXEL_FORMAT;

typedef enum TY_TRIGGER_MODE_LIST
{
    TY_TRIGGER_MODE_CONTINUES        = 0, //not trigger mode, continues mode
    TY_TRIGGER_MODE_TRIG_SLAVE       = 1, //slave mode
    TY_TRIGGER_MODE_M_SIG            = 2, //master mode 1, trigger once got a trigger cmd
    TY_TRIGGER_MODE_M_PER            = 3, //master mode 2, trigger period, with const frame rate
}TY_TRIGGER_MODE_LIST;

//------------------------------------------------------------------------------
//  Struct
//------------------------------------------------------------------------------
typedef struct TY_VERSION_INFO
{
    int32_t major;
    int32_t minor;
    int32_t patch;
    int32_t reserved;
}TY_VERSION_INFO;

typedef struct TY_DEVICE_NET_INFO
{
    char    mac[32];
    char    ip[32];
    char    netmask[32];
    char    gateway[32];
    char    reserved[256];
}TY_DEVICE_NET_INFO;

typedef struct TY_DEVICE_BASE_INFO
{
    TY_INTERFACE        devInterface;       ///< interface, see TY_INTERFACE_LIST
    char                id[32];
    char                vendorName[32];
    char                modelName[32];
    TY_VERSION_INFO     hardwareVersion;
    TY_VERSION_INFO     firmwareVersion;
    TY_DEVICE_NET_INFO  netInfo;
    TY_STATUS           status;
    char                reserved[248];
}TY_DEVICE_BASE_INFO;

typedef struct TY_FEATURE_INFO
{
    bool            isValid;            ///< true if feature exists, false otherwise
    int8_t          accessMode;         ///< feature access mode, see TY_ACCESS_MODE_LIST
    bool            writableAtRun;      ///< feature can be written while capturing
    char            reserved0[1];
    TY_COMPONENT_ID componentID;
    TY_FEATURE_ID   featureID;
    char            name[32];
    int32_t         bindComponentID;    ///< component ID current feature bind to
    int32_t         bindFeatureID;      ///< feature ID current feature bind to
    char            reserved[252];
}TY_FEATURE_INFO;

typedef struct TY_INT_RANGE
{
	int32_t min;
	int32_t max;
	int32_t inc;
	int32_t reserved[1];
}TY_INT_RANGE;

typedef struct TY_FLOAT_RANGE
{
	float   min;
	float   max;
	float   inc;
	float   reserved[1];
}TY_FLOAT_RANGE;

typedef struct TY_ENUM_ENTRY
{
	char    description[64];
	int32_t value;
	int32_t reserved[3];
}TY_ENUM_ENTRY;

typedef struct TY_VECT_3F
{
    float   x;
    float   y;
    float   z;
}TY_VECT_3F;

/// [fx,  0, cx,
///   0, fy, cy,
///   0,  0,  1]
typedef struct
{
    float data[3*3];
}TY_CAMERA_INTRINSIC;

/// [r11, r12, r13, t1,
///  r21, r22, r23, t2,
///  r31, r32, r33, t3,
///    0,   0,   0,  1]
typedef struct
{
    float data[4*4];
}TY_CAMERA_EXTRINSIC;

///camera distortion parameters
typedef struct
{
    float data[12];///<k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4
}TY_CAMERA_DISTORTION;

typedef struct TY_TRIGGER_MODE
{
    int16_t  mode;
    int8_t   fps;
}TY_TRIGGER_MODE;


//------------------------------------------------------------------------------
//  Buffer & Callback
//------------------------------------------------------------------------------
typedef struct TY_IMAGE_DATA
{
    uint64_t timestamp;             ///< Timestamp in microseconds
    int32_t imageIndex;             ///< image index used in trigger mode
    int32_t status;                 ///< Status of this buffer
    int32_t componentID;            ///< Where current data come from
    int32_t size;                   ///< Buffer size
    void*   buffer;                 ///< Pointer to data buffer
    int32_t width;                  ///< Image width in pixels
    int32_t height;                 ///< Image height in pixels
    int32_t pixelFormat;            ///< Pixel format, see TY_PIXEL_FORMAT_LIST
    int32_t reserved[8];            ///< Reserved
}TY_IMAGE_DATA;


typedef struct TY_FRAME_DATA
{
    void*           userBuffer;     ///< Pointer to user enqueued buffer, user should enqueue this buffer in the end of callback
    int32_t         bufferSize;     ///< Size of userBuffer
    int32_t         validCount;     ///< Number of valid data
    int32_t         reserved[6];    ///< Reserved
    TY_IMAGE_DATA   image[10];      ///< Buffer data, max to 10 images per frame, each buffer data could be an image or something else.
}TY_FRAME_DATA;

typedef void (*TY_FRAME_CALLBACK) (TY_FRAME_DATA*, void* userdata);


typedef struct TY_EVENT_INFO
{
    TY_EVENT        eventId;
}TY_EVENT_INFO;

typedef void (*TY_EVENT_CALLBACK)(TY_EVENT_INFO*, void* userdata);


//------------------------------------------------------------------------------
// inlines
//------------------------------------------------------------------------------
static inline TY_FEATURE_TYPE TYFeatureType(TY_FEATURE_ID id)
{
    return id & 0xf000;
}

static inline int32_t TYPixelSize(TY_PIXEL_FORMAT pixelFormat)
{
    return (pixelFormat >> 16) & 0x0fff;
}

static inline int32_t TYPixelType(TY_PIXEL_FORMAT pixelFormat)
{
    return pixelFormat & 0xf0000000;
}

//------------------------------------------------------------------------------
//  C API
//------------------------------------------------------------------------------
#define TY_CAPI TY_EXTC TY_EXPORT TY_STATUS TY_STDC


/// @brief Get error information.
/// @param  [in]  errorID       Error id.
/// @return Error string.
TY_EXTC TY_EXPORT const char* TY_STDC TYErrorString (TY_STATUS errorID);


/// @brief Init this library.
///
///   We make this function to be static inline, because we do a version check here.
///   Some user may use the mismatched header file and dynamic library, and
///   that's quite difficult to locate the error.
///
/// @retval TY_STATUS_OK        Succeed.
/// @retval TY_STATUS_ERROR     Has been inited.
static inline TY_STATUS TYInitLib (void);

/// @brief Deinit this library.
/// @retval TY_STATUS_OK        Succeed.
TY_CAPI TYDeinitLib               (void);

/// @brief Get current library version.
/// @param  [out] version       Version infomation to be filled.
/// @retval TY_STATUS_OK        Succeed.
/// @retval TY_STATUS_NULL_POINTER      buffer is NULL.
TY_CAPI TYLibVersion              (TY_VERSION_INFO* version);



/// @brief Get number of current connected devices.
/// @param  [out] deviceNumber  Number of connected devices.
/// @retval TY_STATUS_OK        Succeed.
/// @retval TY_STATUS_NOT_INITED        TYInitLib not called.
/// @retval TY_STATUS_NULL_POINTER      deviceNumber is NULL.
TY_CAPI TYGetDeviceNumber         (int32_t* deviceNumber);

/// @brief Get device info list.
/// @param  [out] deviceInfos   Device info array to be filled.
/// @param  [in]  bufferCount   Array size of deviceInfos.
/// @param  [out] filledDeviceCount     Number of filled TY_DEVICE_BASE_INFO.
/// @retval TY_STATUS_OK        Succeed.
/// @retval TY_STATUS_NOT_INITED        TYInitLib not called.
/// @retval TY_STATUS_NULL_POINTER      deviceInfos or filledDeviceCount is NULL.
TY_CAPI TYGetDeviceList           (TY_DEVICE_BASE_INFO* deviceInfos, int32_t bufferCount, int32_t* filledDeviceCount);



/// @brief Open device by device ID.
/// @param  [in]  deviceID      Device ID string, can be get from TY_DEVICE_BASE_INFO.
/// @param  [out] deviceHandle  Handle of opened device.
/// @retval TY_STATUS_OK        Succeed.
/// @retval TY_STATUS_NOT_INITED        TYInitLib not called.
/// @retval TY_STATUS_NULL_POINTER      deviceID or deviceHandle is NULL.
/// @retval TY_STATUS_INVALID_PARAMETER Device not found.
/// @retval TY_STATUS_BUSY              Device has been opened.
/// @retval TY_STATUS_DEVICE_ERROR      Open device failed.
TY_CAPI TYOpenDevice              (const char* deviceID, TY_DEV_HANDLE* deviceHandle);

/// @brief Open device by device IP, useful when device not listed.
/// @param  [in]  IP            Device IP.
/// @param  [out] deviceHandle  Handle of opened device.
/// @retval TY_STATUS_OK        Succeed.
/// @retval TY_STATUS_NOT_INITED        TYInitLib not called.
/// @retval TY_STATUS_NULL_POINTER      IP or deviceHandle is NULL.
/// @retval TY_STATUS_INVALID_PARAMETER Device not found.
/// @retval TY_STATUS_BUSY              Device has been opened, may occupied somewhere else.
/// @retval TY_STATUS_DEVICE_ERROR      Open device failed.
TY_CAPI TYOpenDeviceWithIP        (const char* IP, TY_DEV_HANDLE* deviceHandle);

/// @brief Close device by device handle.
/// @param  [in]  hDevice       Device handle.
/// @retval TY_STATUS_OK        Succeed.
/// @retval TY_STATUS_INVALID_HANDLE    Invalid device handle.
/// @retval TY_STATUS_IDLE              Device has been closed.
TY_CAPI TYCloseDevice             (TY_DEV_HANDLE hDevice);

/// @brief Get base info of the open device.
/// @param  [in]  hDevice       Device handle.
/// @param  [out] info          Base info out.
/// @retval TY_STATUS_OK        Succeed.
/// @retval TY_STATUS_INVALID_HANDLE    Invalid device handle.
/// @retval TY_STATUS_NULL_POINTER      componentIDs is NULL.
TY_CAPI TYGetDeviceInfo           (TY_DEV_HANDLE hDevice, TY_DEVICE_BASE_INFO* info);

/// @brief Get all components IDs.
/// @param  [in]  hDevice       Device handle.
/// @param  [out] componentIDs  All component IDs this device has.
/// @retval TY_STATUS_OK        Succeed.
/// @retval TY_STATUS_INVALID_HANDLE    Invalid device handle.
/// @retval TY_STATUS_NULL_POINTER      componentIDs is NULL.
TY_CAPI TYGetComponentIDs         (TY_DEV_HANDLE hDevice, int32_t* componentIDs);

/// @brief Get all enabled components IDs.
/// @param  [in]  hDevice       Device handle.
/// @param  [out] componentIDs  Enabled component IDs.
/// @retval TY_STATUS_OK        Succeed.
/// @retval TY_STATUS_INVALID_HANDLE    Invalid device handle.
/// @retval TY_STATUS_NULL_POINTER      componentIDs is NULL.
TY_CAPI TYGetEnabledComponentIDs  (TY_DEV_HANDLE hDevice, int32_t* componentIDs);

/// @brief Enable components.
/// @param  [in]  hDevice       Device handle.
/// @param  [in]  componentIDs  Components to be enabled.
/// @retval TY_STATUS_OK        Succeed.
/// @retval TY_STATUS_INVALID_HANDLE    Invalid device handle.
/// @retval TY_STATUS_INVALID_COMPONENT Some components specified by componentIDs are invalid.
/// @retval TY_STATUS_BUSY      Device is capturing.
TY_CAPI TYEnableComponents        (TY_DEV_HANDLE hDevice, int32_t componentIDs);

/// @brief Disable components.
/// @param  [in]  hDevice       Device handle.
/// @param  [in]  componentIDs  Components to be disabled.
/// @retval TY_STATUS_OK        Succeed.
/// @retval TY_STATUS_INVALID_HANDLE    Invalid device handle.
/// @retval TY_STATUS_INVALID_COMPONENT Some components specified by componentIDs are invalid.
/// @retval TY_STATUS_BUSY      Device is capturing.
TY_CAPI TYDisableComponents       (TY_DEV_HANDLE hDevice, int32_t componentIDs);



/// @brief Get total buffer size of one frame in current configuration.
/// @param  [in]  hDevice       Device handle.
/// @param  [out] bufferSize    Buffer size per frame.
/// @retval TY_STATUS_OK        Succeed.
/// @retval TY_STATUS_INVALID_HANDLE    Invalid device handle.
/// @retval TY_STATUS_NULL_POINTER      bufferSize is NULL.
TY_CAPI TYGetFrameBufferSize      (TY_DEV_HANDLE hDevice, int32_t* bufferSize);

/// @brief Enqueue a user allocated buffer.
/// @param  [in]  hDevice       Device handle.
/// @param  [in]  buffer        Buffer to be enqueued.
/// @param  [in]  bufferSize    Size of the input buffer.
/// @retval TY_STATUS_OK        Succeed.
/// @retval TY_STATUS_INVALID_HANDLE    Invalid device handle.
/// @retval TY_STATUS_NULL_POINTER      buffer is NULL.
/// @retval TY_STATUS_WRONG_SIZE        The input buffer is not large enough.
TY_CAPI TYEnqueueBuffer           (TY_DEV_HANDLE hDevice, void* buffer, int32_t bufferSize);

/// @brief Clear the internal buffer queue, so that user can release all the buffer.
/// @param  [in]  hDevice       Device handle.
/// @retval TY_STATUS_OK        Succeed.
/// @retval TY_STATUS_INVALID_HANDLE    Invalid device handle.
/// @retval TY_STATUS_BUSY      Device is capturing.
TY_CAPI TYClearBufferQueue        (TY_DEV_HANDLE hDevice);



/// @brief Start capture.
/// @param  [in]  hDevice       Device handle.
/// @retval TY_STATUS_OK        Succeed.
/// @retval TY_STATUS_INVALID_HANDLE    Invalid device handle.
/// @retval TY_STATUS_INVALID_COMPONENT No components enabled.
/// @retval TY_STATUS_BUSY              Device has been started.
/// @retval TY_STATUS_DEVICE_ERROR      Start capture failed.
TY_CAPI TYStartCapture            (TY_DEV_HANDLE hDevice);

/// @brief Stop capture.
/// @param  [in]  hDevice       Device handle.
/// @retval TY_STATUS_OK        Succeed.
/// @retval TY_STATUS_INVALID_HANDLE    Invalid device handle.
/// @retval TY_STATUS_IDLE              Device is not capturing.
/// @retval TY_STATUS_DEVICE_ERROR      Stop capture failed.
TY_CAPI TYStopCapture             (TY_DEV_HANDLE hDevice);

/// @brief Get if the device is capturing.
/// @param  [in]  hDevice       Device handle.
/// @param  [out] isCapturing   Return capturing status.
/// @retval TY_STATUS_OK        Succeed.
/// @retval TY_STATUS_INVALID_HANDLE    Invalid device handle.
/// @retval TY_STATUS_NULL_POINTER      isCapturing is NULL.
TY_CAPI TYIsCapturing             (TY_DEV_HANDLE hDevice, bool* isCapturing);

/// @brief Send a software trigger when device works in trigger mode.
/// @param  [in]  hDevice       Device handle.
/// @retval TY_STATUS_OK        Succeed.
/// @retval TY_STATUS_INVALID_HANDLE    Invalid device handle.
/// @retval TY_STATUS_INVALID_FEATURE   Not support soft trigger.
/// @retval TY_STATUS_IDLE              Device has not started capture.
/// @retval TY_STATUS_WRONG_MODE        Not in trigger mode.
TY_CAPI TYSendSoftTrigger         (TY_DEV_HANDLE hDevice);



/// @brief Register callback of frame. Register NULL to clean callback.
/// @param  [in]  hDevice       Device handle.
/// @param  [in]  callback      Callback function.
/// @param  [in]  userdata      User private data.
/// @retval TY_STATUS_OK        Succeed.
/// @retval TY_STATUS_INVALID_HANDLE    Invalid device handle.
/// @retval TY_STATUS_BUSY      Device is capturing.
TY_CAPI TYRegisterCallback        (TY_DEV_HANDLE hDevice, TY_FRAME_CALLBACK callback, void* userdata);

/// @brief Register device status callback. Register NULL to clean callback.
/// @param  [in]  hDevice       Device handle.
/// @param  [in]  callback      Callback function.
/// @param  [in]  userdata      User private data.
/// @retval TY_STATUS_OK        Succeed.
/// @retval TY_STATUS_INVALID_HANDLE    Invalid device handle.
/// @retval TY_STATUS_BUSY      Device is capturing.
TY_CAPI TYRegisterEventCallback   (TY_DEV_HANDLE hDevice, TY_EVENT_CALLBACK callback, void* userdata);

/// @brief Fetch one frame.
/// @param  [in]  hDevice       Device handle.
/// @param  [out] frame         Frame data to be filled.
/// @param  [in]  timeout       Timeout in milliseconds. <0 for infinite.
/// @retval TY_STATUS_OK        Succeed.
/// @retval TY_STATUS_INVALID_HANDLE    Invalid device handle.
/// @retval TY_STATUS_NULL_POINTER      frame is NULL.
/// @retval TY_STATUS_IDLE              Device capturing is not started.
/// @retval TY_STATUS_WRONG_MODE        Callback has been registered, this function is disabled.
/// @retval TY_STATUS_TIMEOUT   Timeout.
TY_CAPI TYFetchFrame              (TY_DEV_HANDLE hDevice, TY_FRAME_DATA* frame, int32_t timeout);



/// @brief Get feature info.
/// @param  [in]  hDevice       Device handle.
/// @param  [in]  componentID   Component ID.
/// @param  [in]  featureID     Feature ID.
/// @param  [out] featureInfo   Feature info.
/// @retval TY_STATUS_OK        Succeed.
/// @retval TY_STATUS_INVALID_HANDLE    Invalid device handle.
/// @retval TY_STATUS_INVALID_COMPONENT Invalid component ID.
/// @retval TY_STATUS_NULL_POINTER      featureInfo is NULL.
TY_CAPI TYGetFeatureInfo          (TY_DEV_HANDLE hDevice, TY_COMPONENT_ID componentID, TY_FEATURE_ID featureID, TY_FEATURE_INFO* featureInfo);



/// @brief Get value range of integer feature.
/// @param  [in]  hDevice       Device handle.
/// @param  [in]  componentID   Component ID.
/// @param  [in]  featureID     Feature ID.
/// @param  [out] intRange      Integer range to be filled.
/// @retval TY_STATUS_OK        Succeed.
/// @retval TY_STATUS_INVALID_HANDLE    Invalid device handle.
/// @retval TY_STATUS_INVALID_COMPONENT Invalid component ID.
/// @retval TY_STATUS_INVALID_FEATURE   Invalid feature ID.
/// @retval TY_STATUS_WRONG_TYPE        The feature's type is not TY_FEATURE_INT.
/// @retval TY_STATUS_NULL_POINTER      intRange is NULL.
TY_CAPI TYGetIntRange             (TY_DEV_HANDLE hDevice, TY_COMPONENT_ID componentID, TY_FEATURE_ID featureID, TY_INT_RANGE* intRange);

/// @brief Get value of integer feature.
/// @param  [in]  hDevice       Device handle.
/// @param  [in]  componentID   Component ID.
/// @param  [in]  featureID     Feature ID.
/// @param  [out] value         Integer value.
/// @retval TY_STATUS_OK        Succeed.
/// @retval TY_STATUS_INVALID_HANDLE    Invalid device handle.
/// @retval TY_STATUS_INVALID_COMPONENT Invalid component ID.
/// @retval TY_STATUS_INVALID_FEATURE   Invalid feature ID.
/// @retval TY_STATUS_WRONG_TYPE        The feature's type is not TY_FEATURE_INT.
/// @retval TY_STATUS_NULL_POINTER      value is NULL.
TY_CAPI TYGetInt                  (TY_DEV_HANDLE hDevice, TY_COMPONENT_ID componentID, TY_FEATURE_ID featureID, int32_t* value);

/// @brief Set value of integer feature.
/// @param  [in]  hDevice       Device handle.
/// @param  [in]  componentID   Component ID.
/// @param  [in]  featureID     Feature ID.
/// @param  [in]  value         Integer value.
/// @retval TY_STATUS_OK        Succeed.
/// @retval TY_STATUS_INVALID_HANDLE    Invalid device handle.
/// @retval TY_STATUS_INVALID_COMPONENT Invalid component ID.
/// @retval TY_STATUS_INVALID_FEATURE   Invalid feature ID.
/// @retval TY_STATUS_NOT_PERMITTED     The feature is not writable.
/// @retval TY_STATUS_WRONG_TYPE        The feature's type is not TY_FEATURE_INT.
/// @retval TY_STATUS_OUT_OF_RANGE      value is out of range.
/// @retval TY_STATUS_BUSY              Device is capturing, the feature is locked.
TY_CAPI TYSetInt                  (TY_DEV_HANDLE hDevice, TY_COMPONENT_ID componentID, TY_FEATURE_ID featureID, int32_t value);

/// @brief Get value range of float feature.
/// @param  [in]  hDevice       Device handle.
/// @param  [in]  componentID   Component ID.
/// @param  [in]  featureID     Feature ID.
/// @param  [out] floatRange    Float range to be filled.
/// @retval TY_STATUS_OK        Succeed.
/// @retval TY_STATUS_INVALID_HANDLE    Invalid device handle.
/// @retval TY_STATUS_INVALID_COMPONENT Invalid component ID.
/// @retval TY_STATUS_INVALID_FEATURE   Invalid feature ID.
/// @retval TY_STATUS_WRONG_TYPE        The feature's type is not TY_FEATURE_FLOAT.
/// @retval TY_STATUS_NULL_POINTER      floatRange is NULL.
TY_CAPI TYGetFloatRange           (TY_DEV_HANDLE hDevice, TY_COMPONENT_ID componentID, TY_FEATURE_ID featureID, TY_FLOAT_RANGE* floatRange);

/// @brief Get value of float feature.
/// @param  [in]  hDevice       Device handle.
/// @param  [in]  componentID   Component ID.
/// @param  [in]  featureID     Feature ID.
/// @param  [out] value         Float value.
/// @retval TY_STATUS_OK        Succeed.
/// @retval TY_STATUS_INVALID_HANDLE    Invalid device handle.
/// @retval TY_STATUS_INVALID_COMPONENT Invalid component ID.
/// @retval TY_STATUS_INVALID_FEATURE   Invalid feature ID.
/// @retval TY_STATUS_WRONG_TYPE        The feature's type is not TY_FEATURE_FLOAT.
/// @retval TY_STATUS_NULL_POINTER      value is NULL.
TY_CAPI TYGetFloat                (TY_DEV_HANDLE hDevice, TY_COMPONENT_ID componentID, TY_FEATURE_ID featureID, float* value);

/// @brief Set value of float feature.
/// @param  [in]  hDevice       Device handle.
/// @param  [in]  componentID   Component ID.
/// @param  [in]  featureID     Feature ID.
/// @param  [in]  value         Float value.
/// @retval TY_STATUS_OK        Succeed.
/// @retval TY_STATUS_INVALID_HANDLE    Invalid device handle.
/// @retval TY_STATUS_INVALID_COMPONENT Invalid component ID.
/// @retval TY_STATUS_INVALID_FEATURE   Invalid feature ID.
/// @retval TY_STATUS_NOT_PERMITTED     The feature is not writable.
/// @retval TY_STATUS_WRONG_TYPE        The feature's type is not TY_FEATURE_FLOAT.
/// @retval TY_STATUS_OUT_OF_RANGE      value is out of range.
/// @retval TY_STATUS_BUSY              Device is capturing, the feature is locked.
TY_CAPI TYSetFloat                (TY_DEV_HANDLE hDevice, TY_COMPONENT_ID componentID, TY_FEATURE_ID featureID, float value);

/// @brief Get number of enum entries.
/// @param  [in]  hDevice       Device handle.
/// @param  [in]  componentID   Component ID.
/// @param  [in]  featureID     Feature ID.
/// @param  [out] entryCount    Entry count.
/// @retval TY_STATUS_OK        Succeed.
/// @retval TY_STATUS_INVALID_HANDLE    Invalid device handle.
/// @retval TY_STATUS_INVALID_COMPONENT Invalid component ID.
/// @retval TY_STATUS_INVALID_FEATURE   Invalid feature ID.
/// @retval TY_STATUS_WRONG_TYPE        The feature's type is not TY_FEATURE_ENUM.
/// @retval TY_STATUS_NULL_POINTER      entryCount is NULL.
TY_CAPI TYGetEnumEntryCount       (TY_DEV_HANDLE hDevice, TY_COMPONENT_ID componentID, TY_FEATURE_ID featureID, int32_t* entryCount);

/// @brief Get list of enum entries.
/// @param  [in]  hDevice       Device handle.
/// @param  [in]  componentID   Component ID.
/// @param  [in]  featureID     Feature ID.
/// @param  [out] entries       Output entries.
/// @param  [in]  entryCount    Array size of input parameter "entries".
/// @param  [out] filledEntryCount      Number of filled entries.
/// @retval TY_STATUS_OK        Succeed.
/// @retval TY_STATUS_INVALID_HANDLE    Invalid device handle.
/// @retval TY_STATUS_INVALID_COMPONENT Invalid component ID.
/// @retval TY_STATUS_INVALID_FEATURE   Invalid feature ID.
/// @retval TY_STATUS_WRONG_TYPE        The feature's type is not TY_FEATURE_ENUM.
/// @retval TY_STATUS_NULL_POINTER      entries or filledEntryCount is NULL.
TY_CAPI TYGetEnumEntryInfo        (TY_DEV_HANDLE hDevice, TY_COMPONENT_ID componentID, TY_FEATURE_ID featureID, TY_ENUM_ENTRY* entries, int32_t entryCount, int32_t* filledEntryCount);

/// @brief Get current value of enum feature.
/// @param  [in]  hDevice       Device handle.
/// @param  [in]  componentID   Component ID.
/// @param  [in]  featureID     Feature ID.
/// @param  [out] value         Enum value.
/// @retval TY_STATUS_OK        Succeed.
/// @retval TY_STATUS_INVALID_HANDLE    Invalid device handle.
/// @retval TY_STATUS_INVALID_COMPONENT Invalid component ID.
/// @retval TY_STATUS_INVALID_FEATURE   Invalid feature ID.
/// @retval TY_STATUS_WRONG_TYPE        The feature's type is not TY_FEATURE_ENUM.
/// @retval TY_STATUS_NULL_POINTER      value is NULL.
TY_CAPI TYGetEnum                 (TY_DEV_HANDLE hDevice, TY_COMPONENT_ID componentID, TY_FEATURE_ID featureID, int32_t* value);

/// @brief Set value of enum feature.
/// @param  [in]  hDevice       Device handle.
/// @param  [in]  componentID   Component ID.
/// @param  [in]  featureID     Feature ID.
/// @param  [in]  value         Enum value.
/// @retval TY_STATUS_OK        Succeed.
/// @retval TY_STATUS_INVALID_HANDLE    Invalid device handle.
/// @retval TY_STATUS_INVALID_COMPONENT Invalid component ID.
/// @retval TY_STATUS_INVALID_FEATURE   Invalid feature ID.
/// @retval TY_STATUS_NOT_PERMITTED     The feature is not writable.
/// @retval TY_STATUS_WRONG_TYPE        The feature's type is not TY_FEATURE_ENUM.
/// @retval TY_STATUS_INVALID_PARAMETER value is invalid.
/// @retval TY_STATUS_BUSY              Device is capturing, the feature is locked.
TY_CAPI TYSetEnum                 (TY_DEV_HANDLE hDevice, TY_COMPONENT_ID componentID, TY_FEATURE_ID featureID, int32_t value);

/// @brief Get value of bool feature.
/// @param  [in]  hDevice       Device handle.
/// @param  [in]  componentID   Component ID.
/// @param  [in]  featureID     Feature ID.
/// @param  [out] value         Bool value.
/// @retval TY_STATUS_OK        Succeed.
/// @retval TY_STATUS_INVALID_HANDLE    Invalid device handle.
/// @retval TY_STATUS_INVALID_COMPONENT Invalid component ID.
/// @retval TY_STATUS_INVALID_FEATURE   Invalid feature ID.
/// @retval TY_STATUS_WRONG_TYPE        The feature's type is not TY_FEATURE_BOOL.
/// @retval TY_STATUS_NULL_POINTER      value is NULL.
TY_CAPI TYGetBool                 (TY_DEV_HANDLE hDevice, TY_COMPONENT_ID componentID, TY_FEATURE_ID featureID, bool* value);

/// @brief Set value of bool feature.
/// @param  [in]  hDevice       Device handle.
/// @param  [in]  componentID   Component ID.
/// @param  [in]  featureID     Feature ID.
/// @param  [in]  value         Bool value.
/// @retval TY_STATUS_OK        Succeed.
/// @retval TY_STATUS_INVALID_HANDLE    Invalid device handle.
/// @retval TY_STATUS_INVALID_COMPONENT Invalid component ID.
/// @retval TY_STATUS_INVALID_FEATURE   Invalid feature ID.
/// @retval TY_STATUS_NOT_PERMITTED     The feature is not writable.
/// @retval TY_STATUS_WRONG_TYPE        The feature's type is not TY_FEATURE_BOOL.
/// @retval TY_STATUS_BUSY              Device is capturing, the feature is locked.
TY_CAPI TYSetBool                 (TY_DEV_HANDLE hDevice, TY_COMPONENT_ID componentID, TY_FEATURE_ID featureID, bool value);

/// @brief Get internal buffer size of string feature.
/// @param  [in]  hDevice       Device handle.
/// @param  [in]  componentID   Component ID.
/// @param  [in]  featureID     Feature ID.
/// @param  [out] size          String buffer size.
/// @retval TY_STATUS_OK        Succeed.
/// @retval TY_STATUS_INVALID_HANDLE    Invalid device handle.
/// @retval TY_STATUS_INVALID_COMPONENT Invalid component ID.
/// @retval TY_STATUS_INVALID_FEATURE   Invalid feature ID.
/// @retval TY_STATUS_WRONG_TYPE        The feature's type is not TY_FEATURE_STRING.
/// @retval TY_STATUS_NULL_POINTER      size is NULL.
TY_CAPI TYGetStringBufferSize     (TY_DEV_HANDLE hDevice, TY_COMPONENT_ID componentID, TY_FEATURE_ID featureID, int32_t* size);

/// @brief Get value of string feature.
/// @param  [in]  hDevice       Device handle.
/// @param  [in]  componentID   Component ID.
/// @param  [in]  featureID     Feature ID.
/// @param  [out] buffer        String buffer.
/// @param  [in]  bufferSize    Size of buffer.
/// @retval TY_STATUS_OK        Succeed.
/// @retval TY_STATUS_INVALID_HANDLE    Invalid device handle.
/// @retval TY_STATUS_INVALID_COMPONENT Invalid component ID.
/// @retval TY_STATUS_INVALID_FEATURE   Invalid feature ID.
/// @retval TY_STATUS_WRONG_TYPE        The feature's type is not TY_FEATURE_STRING.
/// @retval TY_STATUS_NULL_POINTER      buffer is NULL.
TY_CAPI TYGetString               (TY_DEV_HANDLE hDevice, TY_COMPONENT_ID componentID, TY_FEATURE_ID featureID, char* buffer, int32_t bufferSize);

/// @brief Set value of string feature.
/// @param  [in]  hDevice       Device handle.
/// @param  [in]  componentID   Component ID.
/// @param  [in]  featureID     Feature ID.
/// @param  [in]  buffer        String buffer.
/// @retval TY_STATUS_OK        Succeed.
/// @retval TY_STATUS_INVALID_HANDLE    Invalid device handle.
/// @retval TY_STATUS_INVALID_COMPONENT Invalid component ID.
/// @retval TY_STATUS_INVALID_FEATURE   Invalid feature ID.
/// @retval TY_STATUS_NOT_PERMITTED     The feature is not writable.
/// @retval TY_STATUS_WRONG_TYPE        The feature's type is not TY_FEATURE_STRING.
/// @retval TY_STATUS_NULL_POINTER      buffer is NULL.
/// @retval TY_STATUS_OUT_OF_RANGE      Input string is too long.
/// @retval TY_STATUS_BUSY              Device is capturing, the feature is locked.
TY_CAPI TYSetString               (TY_DEV_HANDLE hDevice, TY_COMPONENT_ID componentID, TY_FEATURE_ID featureID, const char* buffer);

/// @brief Get value of struct.
/// @param  [in]  hDevice       Device handle.
/// @param  [in]  componentID   Component ID.
/// @param  [in]  featureID     Feature ID.
/// @param  [out] pStruct       Pointer of struct.
/// @param  [in]  structSize    Size of input buffer pStruct..
/// @retval TY_STATUS_OK        Succeed.
/// @retval TY_STATUS_INVALID_HANDLE    Invalid device handle.
/// @retval TY_STATUS_INVALID_COMPONENT Invalid component ID.
/// @retval TY_STATUS_INVALID_FEATURE   Invalid feature ID.
/// @retval TY_STATUS_WRONG_TYPE        The feature's type is not TY_FEATURE_STRUCT.
/// @retval TY_STATUS_NULL_POINTER      pStruct is NULL.
/// @retval TY_STATUS_WRONG_SIZE        structSize incorrect.
TY_CAPI TYGetStruct               (TY_DEV_HANDLE hDevice, TY_COMPONENT_ID componentID, TY_FEATURE_ID featureID, void* pStruct, int32_t structSize);

/// @brief Set value of struct.
/// @param  [in]  hDevice       Device handle.
/// @param  [in]  componentID   Component ID.
/// @param  [in]  featureID     Feature ID.
/// @param  [in]  pStruct       Pointer of struct.
/// @param  [in]  structSize    Size of struct.
/// @retval TY_STATUS_OK        Succeed.
/// @retval TY_STATUS_INVALID_HANDLE    Invalid device handle.
/// @retval TY_STATUS_INVALID_COMPONENT Invalid component ID.
/// @retval TY_STATUS_INVALID_FEATURE   Invalid feature ID.
/// @retval TY_STATUS_NOT_PERMITTED     The feature is not writable.
/// @retval TY_STATUS_WRONG_TYPE        The feature's type is not TY_FEATURE_STRUCT.
/// @retval TY_STATUS_NULL_POINTER      pStruct is NULL.
/// @retval TY_STATUS_WRONG_SIZE        structSize incorrect.
/// @retval TY_STATUS_BUSY              Device is capturing, the feature is locked.
TY_CAPI TYSetStruct               (TY_DEV_HANDLE hDevice, TY_COMPONENT_ID componentID, TY_FEATURE_ID featureID, void* pStruct, int32_t structSize);


/// @brief Convert points on depth image to world coordinate.
/// format of depth data should be:
///   +------+------+------+------+------+------+----
///   | p0.x | p0.y | p0.z | p1.x | p1.y | p1.z | ...
///   +------+------+------+------+------+------+----
/// and world coordinate should be:
///   +------+------+------+---------------+------+------+------+---------------+----
///   | p0.x | p0.y | p0.z | padding bytes | p1.x | p1.y | p1.z | padding bytes | ...
///   +------+------+------+---------------+------+------+------+---------------+----
///   * padding bytes could be 0
///   * for PCL, world coordinate padding size should be calculated based on the point type
/// @param  [in]  hDevice       Device handle.
/// @param  [in]  depth         Depth values.
/// @param  [out] world         World coordinate.
/// @param  [in]  worldPaddingBytes Number of world padding bytes.
/// @param  [in]  pointCount    Number of points to be calculated.
/// @retval TY_STATUS_OK        Succeed.
/// @retval TY_STATUS_INVALID_HANDLE    Invalid device handle.
/// @retval TY_STATUS_NULL_POINTER      pDepth or pWorld is NULL.
/// @retval TY_STATUS_INVALID_PARAMETER worldPaddingBytes is not 4x.
TY_CAPI TYDepthToWorld            (TY_DEV_HANDLE hDevice, const TY_VECT_3F* depth, TY_VECT_3F* world, int32_t worldPaddingBytes, int32_t pointCount);

/// @brief Convert world coordinate to depth coordinate.
/// format of depth data should be:
///   +------+------+------+------+------+------+----
///   | p0.x | p0.y | p0.z | p1.x | p1.y | p1.z | ...
///   +------+------+------+------+------+------+----
/// and world coordinate should be:
///   +------+------+------+---------------+------+------+------+---------------+----
///   | p0.x | p0.y | p0.z | padding bytes | p1.x | p1.y | p1.z | padding bytes | ...
///   +------+------+------+---------------+------+------+------+---------------+----
///   * padding bytes could be 0
///   * for PCL, world coordinate padding size should be calculated based on the point type
/// @param  [in]  hDevice       Device handle.
/// @param  [in]  world         World coordinate.
/// @param  [out] depth         Depth values.
/// @param  [in]  worldPaddingBytes Number of depth padding bytes.
/// @param  [in]  pointCount    Number of points to be calculated.
/// @retval TY_STATUS_OK        Succeed.
/// @retval TY_STATUS_INVALID_HANDLE    Invalid device handle.
/// @retval TY_STATUS_NULL_POINTER      pDepth or pWorld is NULL.
/// @retval TY_STATUS_INVALID_PARAMETER worldPaddingBytes is not 4x.
TY_CAPI TYWorldToDepth            (TY_DEV_HANDLE hDevice, const TY_VECT_3F* world, TY_VECT_3F* depth, int32_t worldPaddingBytes, int32_t pointCount);

/// @brief Correct image for lens distortion
/// Format of source image data should be  TY_PIXEL_FORMAT_MONO or  TY_PIXEL_FORMAT_RGB
/// Output buffer is allocated by caller.
/// For IR image undistortion, enable TY_BOOL_UNDISTORTION  to get better performance.
///
/// @param  [in]  cameraIntrinsic       input image camera intrinsic parameters
/// @param  [in]  cameraDistortion      input image camera distortion parameters
/// @param  [in]  cameraNewIntrinsic    output image camera intrinsic , cameraIntrinsic will be used if is NULL
/// @param  [in]  srcImage              input image buffer
/// @param  [out] dstImage              Output image buffer
///
/// @retval TY_STATUS_OK                Succeed.
/// @retval TY_STATUS_NULL_POINTER      buffer is NULL.
/// @retval TY_STATUS_INVALID_PARAMETER parameter is invalid.
/// @retval TY_STATUS_ERROR             internal error
TY_CAPI             TYUndistortImage          (const TY_CAMERA_INTRINSIC *cameraIntrinsic, const TY_CAMERA_DISTORTION* cameraDistortion,const TY_CAMERA_INTRINSIC *cameraNewIntrinsic,const TY_IMAGE_DATA *srcImage, TY_IMAGE_DATA *dstImage);

//------------------------------------------------------------------------------
//  Version check
//------------------------------------------------------------------------------
TY_CAPI _TYInitLib(void);
static inline TY_STATUS TYInitLib(void)
{
    TY_VERSION_INFO soVersion;
    TYLibVersion(&soVersion);
    if(!(soVersion.major == TY_LIB_VERSION_MAJOR && soVersion.minor >= TY_LIB_VERSION_MINOR)){
        abort();   // generate fault directly
    }
    return _TYInitLib();
}

//------------------------------------------------------------------------------
//  Summary
//------------------------------------------------------------------------------
TY_EXTC TY_EXPORT const char* TY_STDC TYErrorString (TY_STATUS errorID);

inline TY_STATUS    TYInitLib                 (void);
TY_CAPI             TYDeinitLib               (void);
TY_CAPI             TYLibVersion              (TY_VERSION_INFO* version);
TY_EXTC TY_EXPORT   const char*         TYGetFirmwareVer          (const char* deviceID);
TY_CAPI             TYGetDeviceNumber         (int32_t* deviceNumber);
TY_CAPI             TYGetDeviceList           (TY_DEVICE_BASE_INFO* deviceInfos, int32_t bufferCount, int32_t* filledDeviceCount);

TY_CAPI             TYOpenDevice              (const char* deviceID, TY_DEV_HANDLE* outDeviceHandle);
TY_CAPI             TYOpenDeviceWithIP        (const char* IP, TY_DEV_HANDLE* outDeviceHandle);
TY_CAPI             TYCloseDevice             (TY_DEV_HANDLE hDevice);

TY_CAPI             TYGetDeviceInfo           (TY_DEV_HANDLE hDevice, TY_DEVICE_BASE_INFO* info);
TY_CAPI             TYGetComponentIDs         (TY_DEV_HANDLE hDevice, int32_t* componentIDs);
TY_CAPI             TYGetEnabledComponents    (TY_DEV_HANDLE hDevice, int32_t* componentIDs);
TY_CAPI             TYEnableComponents        (TY_DEV_HANDLE hDevice, int32_t componentIDs);
TY_CAPI             TYDisableComponents       (TY_DEV_HANDLE hDevice, int32_t componentIDs);

TY_CAPI             TYGetFrameBufferSize      (TY_DEV_HANDLE hDevice, int32_t* outSize);
TY_CAPI             TYEnqueueBuffer           (TY_DEV_HANDLE hDevice, void* buffer, int32_t bufferSize);
TY_CAPI             TYClearBufferQueue        (TY_DEV_HANDLE hDevice);

TY_CAPI             TYStartCapture            (TY_DEV_HANDLE hDevice);
TY_CAPI             TYStopCapture             (TY_DEV_HANDLE hDevice);
TY_CAPI             TYIsCapturing             (TY_DEV_HANDLE hDevice, bool* isCapturing);
TY_CAPI             TYSendSoftTrigger         (TY_DEV_HANDLE hDevice);
TY_CAPI             TYRegisterCallback        (TY_DEV_HANDLE hDevice, TY_FRAME_CALLBACK callback, void* userdata);
TY_CAPI             TYRegisterEventCallback   (TY_DEV_HANDLE hDevice, TY_EVENT_CALLBACK callback, void* userdata);
TY_CAPI             TYFetchFrame              (TY_DEV_HANDLE hDevice, TY_FRAME_DATA* frame, int32_t timeout);

TY_CAPI             TYGetFeatureInfo          (TY_DEV_HANDLE hDevice, TY_COMPONENT_ID componentID, TY_FEATURE_ID featureID, TY_FEATURE_INFO* featureInfo);
TY_CAPI             TYGetIntRange             (TY_DEV_HANDLE hDevice, TY_COMPONENT_ID componentID, TY_FEATURE_ID featureID, TY_INT_RANGE* intRange);
TY_CAPI             TYGetInt                  (TY_DEV_HANDLE hDevice, TY_COMPONENT_ID componentID, TY_FEATURE_ID featureID, int32_t* value);
TY_CAPI             TYSetInt                  (TY_DEV_HANDLE hDevice, TY_COMPONENT_ID componentID, TY_FEATURE_ID featureID, int32_t value);
TY_CAPI             TYGetFloatRange           (TY_DEV_HANDLE hDevice, TY_COMPONENT_ID componentID, TY_FEATURE_ID featureID, TY_FLOAT_RANGE* floatRange);
TY_CAPI             TYGetFloat                (TY_DEV_HANDLE hDevice, TY_COMPONENT_ID componentID, TY_FEATURE_ID featureID, float* value);
TY_CAPI             TYSetFloat                (TY_DEV_HANDLE hDevice, TY_COMPONENT_ID componentID, TY_FEATURE_ID featureID, float value);
TY_CAPI             TYGetEnumEntryCount       (TY_DEV_HANDLE hDevice, TY_COMPONENT_ID componentID, TY_FEATURE_ID featureID, int32_t* entryCount);
TY_CAPI             TYGetEnumEntryInfo        (TY_DEV_HANDLE hDevice, TY_COMPONENT_ID componentID, TY_FEATURE_ID featureID, TY_ENUM_ENTRY* entries, int32_t entryCount, int32_t* filledEntryCount);
TY_CAPI             TYGetEnum                 (TY_DEV_HANDLE hDevice, TY_COMPONENT_ID componentID, TY_FEATURE_ID featureID, int32_t* value);
TY_CAPI             TYSetEnum                 (TY_DEV_HANDLE hDevice, TY_COMPONENT_ID componentID, TY_FEATURE_ID featureID, int32_t value);
TY_CAPI             TYGetBool                 (TY_DEV_HANDLE hDevice, TY_COMPONENT_ID componentID, TY_FEATURE_ID featureID, bool* value);
TY_CAPI             TYSetBool                 (TY_DEV_HANDLE hDevice, TY_COMPONENT_ID componentID, TY_FEATURE_ID featureID, bool value);
TY_CAPI             TYGetStringLength         (TY_DEV_HANDLE hDevice, TY_COMPONENT_ID componentID, TY_FEATURE_ID featureID, int32_t* length);
TY_CAPI             TYGetString               (TY_DEV_HANDLE hDevice, TY_COMPONENT_ID componentID, TY_FEATURE_ID featureID, char* buffer, int32_t bufferSize);
TY_CAPI             TYSetString               (TY_DEV_HANDLE hDevice, TY_COMPONENT_ID componentID, TY_FEATURE_ID featureID, const char* buffer);
TY_CAPI             TYGetStruct               (TY_DEV_HANDLE hDevice, TY_COMPONENT_ID componentID, TY_FEATURE_ID featureID, void* pStruct, int32_t structSize);
TY_CAPI             TYSetStruct               (TY_DEV_HANDLE hDevice, TY_COMPONENT_ID componentID, TY_FEATURE_ID featureID, void* pStruct, int32_t structSize);

// utils api
TY_CAPI             TYDepthToWorld            (TY_DEV_HANDLE hDevice, const TY_VECT_3F* depth, TY_VECT_3F* world, int32_t worldPaddingBytes, int32_t pointCount);
TY_CAPI             TYWorldToDepth            (TY_DEV_HANDLE hDevice, const TY_VECT_3F* world, TY_VECT_3F* depth, int32_t worldPaddingBytes, int32_t pointCount);

TY_CAPI             TYRegisterWorldToColor    (TY_DEV_HANDLE hDevice, const TY_VECT_3F* world, int32_t worldPaddingBytes, int32_t pointCount, uint16_t* outDepthBuffer, int32_t bufferSize);
TY_CAPI             TYRegisterWorldToColor2   (TY_DEV_HANDLE hDevice, const TY_VECT_3F* world, int32_t worldPaddingBytes, int32_t pointCount, int32_t color_width, int32_t color_height, uint16_t* outDepthBuffer, int32_t bufferSize);
TY_CAPI             TYUndistortImage          (const TY_CAMERA_INTRINSIC *cameraIntrinsic, const TY_CAMERA_DISTORTION* cameraDistortion,const TY_CAMERA_INTRINSIC *cameraNewIntrinsic,const TY_IMAGE_DATA *srcImage, TY_IMAGE_DATA *dstImage);


#endif // TY_API_H_
