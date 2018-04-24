/*

*/

#ifndef __FILTER_H__
#define __FILTER_H__

#include <emmintrin.h>
#include <pmmintrin.h>

// define fixed-width datatypes for Visual Studio projects
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

// fast filters: implements 3x3 and 5x5 sobel filters and 
//               5x5 blob and corner filters based on SSE2/3 instructions
namespace filter {
  
  // private namespace, public user functions at the bottom of this file
  namespace detail {
    void integral_image( const uint8_t* in, int32_t* out, int w, int h );
    void unpack_8bit_to_16bit( const __m128i a, __m128i& b0, __m128i& b1 );
    void pack_16bit_to_8bit_saturate( const __m128i a0, const __m128i a1, __m128i& b );
    
    // convolve image with a (1,4,6,4,1) row vector. Result is accumulated into output.
    // output is scaled by 1/128, then clamped to [-128,128], and finally shifted to [0,255].
    void convolve_14641_row_5x5_16bit( const int16_t* in, uint8_t* out, int w, int h );
    
    // convolve image with a (1,2,0,-2,-1) row vector. Result is accumulated into output.
    // This one works on 16bit input and 8bit output.
    // output is scaled by 1/128, then clamped to [-128,128], and finally shifted to [0,255].
    void convolve_12021_row_5x5_16bit( const int16_t* in, uint8_t* out, int w, int h );

    // convolve image with a (1,2,1) row vector. Result is accumulated into output.
    // This one works on 16bit input and 8bit output.
    // output is scaled by 1/4, then clamped to [-128,128], and finally shifted to [0,255].
    void convolve_121_row_3x3_16bit( const int16_t* in, uint8_t* out, int w, int h );
    
    // convolve image with a (1,0,-1) row vector. Result is accumulated into output.
    // This one works on 16bit input and 8bit output.
    // output is scaled by 1/4, then clamped to [-128,128], and finally shifted to [0,255].
    void convolve_101_row_3x3_16bit( const int16_t* in, uint8_t* out, int w, int h );
    
    void convolve_cols_5x5( const unsigned char* in, int16_t* out_v, int16_t* out_h, int w, int h );
    
    void convolve_col_p1p1p0m1m1_5x5( const unsigned char* in, int16_t* out, int w, int h );
    
    void convolve_row_p1p1p0m1m1_5x5( const int16_t* in, int16_t* out, int w, int h );
    
    void convolve_cols_3x3( const unsigned char* in, int16_t* out_v, int16_t* out_h, int w, int h );
  }
  
  void sobel3x3( const uint8_t* in, uint8_t* out_v, uint8_t* out_h, int w, int h );
  
  void sobel5x5( const uint8_t* in, uint8_t* out_v, uint8_t* out_h, int w, int h );
  
  // -1 -1  0  1  1
  // -1 -1  0  1  1
  //  0  0  0  0  0
  //  1  1  0 -1 -1
  //  1  1  0 -1 -1
  void checkerboard5x5( const uint8_t* in, int16_t* out, int w, int h );
  
  // -1 -1 -1 -1 -1
  // -1  1  1  1 -1
  // -1  1  8  1 -1
  // -1  1  1  1 -1
  // -1 -1 -1 -1 -1
  void blob5x5( const uint8_t* in, int16_t* out, int w, int h );
};

#endif
