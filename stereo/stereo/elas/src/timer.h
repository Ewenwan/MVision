/*

*/

#ifndef __TIMER_H__
#define __TIMER_H__

#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <sys/time.h>

// Define fixed-width datatypes for Visual Studio projects
#ifndef _MSC_VER
  #include <stdint.h>
#else
  typedef __int8            int8_t;
  typedef __int16           int16_t;
  typedef __int32           int32_t;
  typedef __int64           int64_t;
  typedef unsigned __int8   uint8_t;
  typedef unsigned __int16  uint16_t;
  typedef unsigned __int32  uint32_t;
  typedef unsigned __int64  uint64_t;
#endif

class Timer {
  
public:
  
  Timer() {}
  
  ~Timer() {}
  
  void start (std::string title) {
    desc.push_back(title);
    push_back_time();
  }
  
  void stop () {
    if (time.size()<=desc.size())
      push_back_time();
  }
  
  void plot () {
    stop();
    float total_time = 0;
    for (int32_t i=0; i<desc.size(); i++) {
      float curr_time = getTimeDifferenceMilliseconds(time[i],time[i+1]);
      total_time += curr_time;
      std::cout.width(30);
      std::cout << desc[i] << " ";
      std::cout << std::fixed << std::setprecision(1) << std::setw(6);
      std::cout << curr_time;
      std::cout << " ms" << std::endl;
    }
    std::cout << "========================================" << std::endl;
    std::cout << "                    Total time ";
    std::cout << std::fixed << std::setprecision(1) << std::setw(6);
    std::cout << total_time;
    std::cout << " ms" << std::endl << std::endl;
  }
  
  void reset () {
    desc.clear();
    time.clear();
  }
  
private:
  
  std::vector<std::string>  desc;
  std::vector<timeval>      time;
  
  void push_back_time () {
    timeval curr_time;
    gettimeofday(&curr_time,0);
    time.push_back(curr_time);
  }
  
  float getTimeDifferenceMilliseconds(timeval a,timeval b) {
    return ((float)(b.tv_sec -a.tv_sec ))*1e+3 +
           ((float)(b.tv_usec-a.tv_usec))*1e-3;
  }
};

#endif
