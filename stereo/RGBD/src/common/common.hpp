#ifndef SAMPLE_COMMON_COMMON_HPP_
#define SAMPLE_COMMON_COMMON_HPP_

#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include "TY_API.h"

#ifndef ASSERT
#define ASSERT(x)   do{ \
                if(!(x)) { \
                    LOGE("Assert failed at %s:%d", __FILE__, __LINE__); \
                    LOGE("    : " #x ); \
                    abort(); \
                } \
            }while(0)
#endif

#ifndef ASSERT_OK
#define ASSERT_OK(x)    do{ \
                int err = (x); \
                if(err != TY_STATUS_OK) { \
                    LOGE("Assert failed: error %d(%s) at %s:%d", err, TYErrorString(err), __FILE__, __LINE__); \
                    LOGE("    : " #x ); \
                    abort(); \
                } \
            }while(0)
#endif


#ifdef _WIN32
# include <windows.h>
# include <time.h>
  static inline int32_t getSystemTime()
  {
      SYSTEMTIME wtm;
      struct tm tm;
      GetLocalTime(&wtm);
      tm.tm_year     = wtm.wYear - 1900;
      tm.tm_mon     = wtm.wMonth - 1;
      tm.tm_mday     = wtm.wDay;
      tm.tm_hour     = wtm.wHour;
      tm.tm_min     = wtm.wMinute;
      tm.tm_sec     = wtm.wSecond;
      tm. tm_isdst    = -1;
      return mktime(&tm) * 1000 + wtm.wMilliseconds;
  }
  static inline void MSleep(uint32_t ms)
  {
      Sleep(ms);
  }
#else
# include <sys/time.h>
# include <unistd.h>
  inline int32_t getSystemTime()
  {
      struct timeval tv;
      gettimeofday(&tv, NULL);
      return tv.tv_sec*1000 + tv.tv_usec/1000;
  }
  static inline void MSleep(uint32_t ms)
  {
      usleep(ms * 1000);
  }
#endif


#define LOGD(fmt,...)  printf("%d " fmt "\n", getSystemTime(), ##__VA_ARGS__)
#define LOGI(fmt,...)  printf("%d " fmt "\n", getSystemTime(), ##__VA_ARGS__)
#define LOGW(fmt,...)  printf("%d " fmt "\n", getSystemTime(), ##__VA_ARGS__)
#define LOGE(fmt,...)  printf("%d Error: " fmt "\n", getSystemTime(), ##__VA_ARGS__)
#define xLOGD(fmt,...)
#define xLOGI(fmt,...)
#define xLOGW(fmt,...)
#define xLOGE(fmt,...)


#ifdef _WIN32
#  include <windows.h>
#  define MSLEEP(x)     Sleep(x)
   // windows defined macro max/min
#  ifdef max
#    undef max
#  endif
#  ifdef min
#    undef min
#  endif
#else
#  include <unistd.h>
#  include <sys/time.h>
#  define MSLEEP(x)     usleep((x)*1000)
#endif

#include "Utils.hpp"
#include "DepthRender.hpp"
#include "MatViewer.hpp"
#include "PointCloudViewer.hpp"    // 显示XYZ点
#include "RGBPointCloudViewer.hpp" // 显示XYZRGB点
#endif
