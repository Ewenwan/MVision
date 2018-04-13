#ifndef COMMON_H
#define COMMON_H

#ifdef _WIN32
    typedef unsigned int uint;
    typedef unsigned char uchar;
#endif

#define COST_TYPE_NOT_FLOAT 1

#if COST_TYPE_NOT_FLOAT
    const int COST_FACTOR = 65000;
    typedef unsigned short int costType;
    #define COST_MAP_TYPE CV_16U
#else
    const float COST_FACTOR = 1.0;
    typedef float costType;
    #define COST_MAP_TYPE CV_32F
#endif

#endif // COMMON_H
