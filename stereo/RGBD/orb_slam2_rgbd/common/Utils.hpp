#ifndef SAMPLE_COMMON_UTILS_HPP_
#define SAMPLE_COMMON_UTILS_HPP_

#include <opencv2/opencv.hpp>
#include "./include/TY_API.h"

static inline const char* colorFormatName(TY_PIXEL_FORMAT fmt)
{
#define FORMAT_CASE(a) case (a): return #a
    switch(fmt){
        FORMAT_CASE(TY_PIXEL_FORMAT_UNDEFINED);
        FORMAT_CASE(TY_PIXEL_FORMAT_MONO);
        FORMAT_CASE(TY_PIXEL_FORMAT_RGB);
        FORMAT_CASE(TY_PIXEL_FORMAT_YVYU);
        FORMAT_CASE(TY_PIXEL_FORMAT_YUYV);
        FORMAT_CASE(TY_PIXEL_FORMAT_DEPTH16);
        FORMAT_CASE(TY_PIXEL_FORMAT_FPOINT3D);
        FORMAT_CASE(TY_PIXEL_FORMAT_BAYER8GB);
        default: return "UNKNOWN FORMAT";
    }
#undef FORMAT_CASE
}


static inline const TY_IMAGE_DATA* TYImageInFrame(const TY_FRAME_DATA& frame
        , const TY_COMPONENT_ID comp)
{
    for(int i = 0; i < frame.validCount; i++){
        if(frame.image[i].componentID == comp){
            return &frame.image[i];
        }
    }
    return NULL;
}


static inline int parseFrame(const TY_FRAME_DATA& frame, cv::Mat* pDepth
        , cv::Mat* pLeftIR, cv::Mat* pRightIR
        , cv::Mat* pColor, cv::Mat* pPoints)
{
    for( int i = 0; i < frame.validCount; i++ ){
        // get depth image
        if(pDepth && frame.image[i].componentID == TY_COMPONENT_DEPTH_CAM){
            *pDepth = cv::Mat(frame.image[i].height, frame.image[i].width
                    , CV_16U, frame.image[i].buffer);
        }
        // get left ir image
        if(pLeftIR && frame.image[i].componentID == TY_COMPONENT_IR_CAM_LEFT){
            *pLeftIR = cv::Mat(frame.image[i].height, frame.image[i].width
                    , CV_8U, frame.image[i].buffer);
        }
        // get right ir image
        if(pRightIR && frame.image[i].componentID == TY_COMPONENT_IR_CAM_RIGHT){
            *pRightIR = cv::Mat(frame.image[i].height, frame.image[i].width
                    , CV_8U, frame.image[i].buffer);
        }
        // get BGR
        if(pColor && frame.image[i].componentID == TY_COMPONENT_RGB_CAM){
            if (frame.image[i].pixelFormat == TY_PIXEL_FORMAT_JPEG){
                cv::Mat jpeg(frame.image[i].height, frame.image[i].width
			    	    , CV_8UC1, frame.image[i].buffer);
				*pColor = cv::imdecode(jpeg,CV_LOAD_IMAGE_COLOR);
			}
            if (frame.image[i].pixelFormat == TY_PIXEL_FORMAT_YVYU){
                cv::Mat yuv(frame.image[i].height, frame.image[i].width
                            , CV_8UC2, frame.image[i].buffer);
                cv::cvtColor(yuv, *pColor, cv::COLOR_YUV2BGR_YVYU);
            }
            else if (frame.image[i].pixelFormat == TY_PIXEL_FORMAT_YUYV){
                cv::Mat yuv(frame.image[i].height, frame.image[i].width
                            , CV_8UC2, frame.image[i].buffer);
                cv::cvtColor(yuv, *pColor, cv::COLOR_YUV2BGR_YUYV);
            } else if(frame.image[i].pixelFormat == TY_PIXEL_FORMAT_RGB){
                cv::Mat rgb(frame.image[i].height, frame.image[i].width
                        , CV_8UC3, frame.image[i].buffer);
                cv::cvtColor(rgb, *pColor, cv::COLOR_RGB2BGR);
            } else if(frame.image[i].pixelFormat == TY_PIXEL_FORMAT_MONO){
                cv::Mat gray(frame.image[i].height, frame.image[i].width
                        , CV_8U, frame.image[i].buffer);
                cv::cvtColor(gray, *pColor, cv::COLOR_GRAY2BGR);
             } else if(frame.image[i].pixelFormat == TY_PIXEL_FORMAT_BAYER8GB){
                cv::Mat raw(frame.image[i].height, frame.image[i].width
                        , CV_8U, frame.image[i].buffer);
                cv::cvtColor(raw, *pColor, cv::COLOR_BayerGB2BGR);
            }
        }
        // get point3D
        if(pPoints && frame.image[i].componentID == TY_COMPONENT_POINT3D_CAM){
            *pPoints = cv::Mat(frame.image[i].height, frame.image[i].width
                    , CV_32FC3, frame.image[i].buffer);
        }
    }

    return 0;
}


#endif
