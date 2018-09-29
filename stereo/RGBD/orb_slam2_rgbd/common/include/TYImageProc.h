#ifndef TY_IMAGE_PROC_H_
#define TY_IMAGE_PROC_H_


#include "TY_API.h"

TY_CAPI TYUndistortImage (const TY_CAMERA_INTRINSIC *cameraIntrinsic
        , const TY_CAMERA_DISTORTION* cameraDistortion
        , const TY_CAMERA_INTRINSIC *cameraNewIntrinsic
        , const TY_IMAGE_DATA *srcImage
        , TY_IMAGE_DATA *dstImage
        );


// -----------------------------------------------------------
struct DepthSpeckleFilterParameters {
    int max_speckle_size; // blob size smaller than this will be removed
    int max_speckle_diff; // Maximum difference between neighbor disparity pixels
};
#define DepthSpeckleFilterParameters_Initializer {150, 64}

TY_CAPI TYDepthSpeckleFilter (TY_IMAGE_DATA* depth_image
        , const DepthSpeckleFilterParameters* param
        );


// -----------------------------------------------------------
struct DepthEnhenceParameters{
    float sigma_s;          // filter on space
    float sigma_r;          // filter on range
    int   outlier_win_sz;   //outlier filter windows ize
    float outlier_rate;
};
#define DepthEnhenceParameters_Initializer {10, 20, 10, 0.1f}

TY_CAPI TYDepthEnhenceFilter (const TY_IMAGE_DATA* depth_images
        , int image_num
        , TY_IMAGE_DATA *guide
        , TY_IMAGE_DATA *output
        , const DepthEnhenceParameters* param
        );


#endif
