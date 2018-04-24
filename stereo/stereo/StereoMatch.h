/********************************************************************
	功能:	双目匹配与三维重建类
*********************************************************************/
#ifndef _STEREO_MATCH_H_
#define _STEREO_MATCH_H_

#pragma once//编译一次

#include <vector>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/types_c.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
//#include "opencv2/contrib/contrib.hpp"
#include "opencv2/core/utility.hpp"
#include <iostream>

// elas 
#include "elas/elas.h"

//pcl
//点云数据处理
#include <pcl/point_cloud.h>
#include <pcl/point_types.h> 

typedef pcl::PointXYZRGB PointT; //点云中的点对象  位置和像素值
typedef pcl::PointCloud<PointT> PointCloud;//整个点云对象

using namespace std;
using namespace cv;
class StereoMatch//自定义类　名字
{
public://共有　方法
	StereoMatch(int imgWidth, int imgHeight, const char* xmlFilePath);
       //默认构造函数　图像宽度　图像高度　配置文件
	virtual ~StereoMatch(void);//默认析构函数　虚析构函数　可继承　
	/*----------------------------
	 * 功能 : 初始化内部变量，载入双目定标结果数据，相机内外参数
	 *----------------------------
	 * 函数 : StereoMatch::init
	 * 访问 : public 
	 * 返回 : 1		成功
	 *	　0		读入校正参数失败
	 *	-1		定标参数的图像尺寸与当前配置的图像尺寸不一致
	 *	-2		校正方法不是 BOUGUET 方法
	 *	-99	未知错误
	 * 参数 : imgWidth	[in]	图像宽度
	 * 参数 : imgHeight	[in]	图像高度
	 * 参数 : xmlFilePath	[in]	双目定标结果数据文件
	 */
	int init(int imgWidth, int imgHeight, const char* xmlFilePath);

	/*----------------------------
	 * 功能 : 基于 BM 算法计算视差
	 *----------------------------
	 * 函数 : StereoMatch::bmMatch
	 * 访问 : public 
	 * 返回 : 0 - 失败，1 - 成功
	 *
	 * 参数 : frameLeft	[in]	左摄像机帧图
	 * 参数 : frameRight	[in]	右摄像机帧图
	 * 参数 : disparity	[out]	视差图
	 */
	int bmMatch(cv::Mat& frameLeft, cv::Mat& frameRight, cv::Mat& disparity);

	/*----------------------------
	 * 功能 : 基于 SGBM 算法计算视差
	 *----------------------------
	 * 函数 : StereoMatch::sgbmMatch
	 * 访问 : public 
	 * 返回 : 0 - 失败，1 - 成功
	 *
	 * 参数 : frameLeft	[in]	左摄像机帧图
	 * 参数 : frameRight	[in]	右摄像机帧图
	 * 参数 : disparity	[out]	视差图
	 */
	int sgbmMatch(cv::Mat& frameLeft, cv::Mat& frameRight, cv::Mat& disparity);
	int hhMatch(cv::Mat& frameLeft, cv::Mat& frameRight, cv::Mat& disparity);
	int wayMatch(cv::Mat& frameLeft, cv::Mat& frameRight, cv::Mat& disparity);
	/*----------------------------
	 * 功能 : 基于 elas 算法计算视差
	 *----------------------------
	 * 函数 : StereoMatch::elasMatch
	 * 访问 : public 
	 * 返回 : 0 - 失败，1 - 成功
	 *
	 * 参数 : frameLeft		[in]	左摄像机帧图
	 * 参数 : frameRight		[in]	右摄像机帧图
	 * 参数 : disparity		[out]	视差图
	 */
        int elasMatch(cv::Mat& frameLeft, cv::Mat& frameRight, cv::Mat& disparity);
	/*----------------------------
	 * 功能 : 基于 VAR 算法计算视差
	 *----------------------------
	 * 函数 : StereoMatch::sgbmMatch
	 * 访问 : public 
	 * 返回 : 0 - 失败，1 - 成功
	 *
	 * 参数 : frameLeft		[in]	左摄像机帧图
	 * 参数 : frameRight		[in]	右摄像机帧图
	 * 参数 : disparity		[out]	视差图
	 * 参数 : imageLeft		[out]	处理后的左视图，用于显示
	 * 参数 : imageRight		[out]	处理后的右视图，用于显示
	 */
//	int varMatch(cv::Mat& frameLeft, cv::Mat& frameRight, cv::Mat& disparity);

	/*----------------------------
	 * 功能 : 计算三维点云
	 *----------------------------
	 * 函数 : StereoMatch::getPointClouds
	 * 访问 : public 
	 * 返回 : 0 - 失败，1 - 成功
	 *
	 * 参数 : disparity		[in]	视差数据
	 * 参数 : pointClouds	[out]	三维点云
	 */
	int getPointClouds(cv::Mat& disparity, cv::Mat& pointClouds);
	int getPCL(cv::Mat& disparity, cv::Mat& img_L, PointCloud& pointCloud);
	int my_getpc(cv::Mat& disparity, cv::Mat& img_L, PointCloud& pointCloud);
	/*----------------------------
	 * 功能 : 获取伪彩色视差图
	 *----------------------------
	 * 函数 : StereoMatch::getDisparityImage
	 * 访问 : public 
	 * 返回 : 0 - 失败，1 - 成功
	 *
	 * 参数 : disparity		[in]	原始视差数据
	 * 参数 : disparityImage	[out]	视差图像
	 * 参数 : isColor		[in]	是否采用伪彩色，默认为 true，设为 false 时返回灰度视差图
	 */
	int getDisparityImage(cv::Mat& disparity, cv::Mat& disparityImage, bool isColor = true);

	/*----------------------------
	 * 功能 : 保存三维点云到本地 txt 文件
	 *----------------------------
	 * 函数 : StereoMatch::savePointClouds
	 * 访问 : public 
	 * 返回 : void
	 *
	 * 参数 : pointClouds	[in]	三维点云数据
	 * 参数 : filename		[in]	文件路径
	 */
	void savePointClouds(cv::Mat& pointClouds, const char* filename);
    
	/*----------------------------
	 * 功能 : 设置视场范围
	 *----------------------------
	 * 函数 : StereoMatch::setViewField
	 * 访问 : public 
	 * 返回 : void
     *
     * 参数 : viewWidth   [in]	视场宽度
     * 参数 : viewHeight  [in]	视场高度
     * 参数 : viewDepth   [in]	视场深度
	 */
    void setViewField(int viewWidth, int viewHeight, int viewDepth)
    {
        m_nViewWidth = viewWidth;
        m_nViewHeight = viewHeight;
        m_nViewDepth = viewDepth;
    }

	/*----------------------------
	 * 功能 : 获取环境俯视图
	 *----------------------------
	 * 函数 : StereoMatch::savePointClouds
	 * 访问 : public 
	 * 返回 : void
     *
     * 参数 : pointClouds	[in]	三维点云数据
     * 参数 : topDownView	[out]	环境俯视图
     * 参数 : image       	[in]	环境图像
	 */
    void getTopDownView(cv::Mat& pointClouds, cv::Mat& topDownView, cv::Mat& image);
    
	/*----------------------------
	 * 功能 : 获取环境俯视图
	 *----------------------------
	 * 函数 : StereoMatch::savePointClouds
	 * 访问 : public 
	 * 返回 : void
     *
     * 参数 : pointClouds	[in]	三维点云数据
     * 参数 : sideView    	[out]	环境侧视图
     * 参数 : image       	[in]	环境图像
	 */
    void getSideView(cv::Mat& pointClouds, cv::Mat& sideView, cv::Mat& image);

	/***
	 *	公开变量
	 */
        cv::Ptr<cv::StereoBM>	m_ptr_BM;   // 立体匹配 BM 方法
	cv::Ptr<cv::StereoSGBM>     m_ptr_SGBM; // 立体匹配 SGBM 方法
	//cv::StereoBM		m_BM;	// 立体匹配 BM 方法
	//cv::StereoSGBM	m_SGBM;	// 立体匹配 SGBM 方法
	//cv::StereoVar		m_VAR;	// 立体匹配 VAR 方法
	Elas elas;//elas类
	//cv::Ptr<Elas> elas_ptr;//
	double			m_FL;	// 左摄像机校正后的焦距值

private:

	/*----------------------------
	 * 功能 : 载入双目定标结果数据
	 *----------------------------
	 * 函数 : StereoMatch::loadCalibData
	 * 访问 : public 
	 * 返回 : 1	成功
	 *	　　0	读入校正参数失败
	 *	 -1	定标参数的图像尺寸与当前配置的图像尺寸不一致
	 *	 -2	校正方法不是 BOUGUET 方法
	 *	-99	未知错误
	 * 
	 * 参数 : xmlFilePath	[in]	双目定标结果数据文件
	 */
	int loadCalibData(const char* xmlFilePath);
	/***
	 *	私有变量
	 */
	bool m_Calib_Data_Loaded;		// 是否成功载入定标参数
	cv::Mat m_Calib_Mat_Q;			// 视差到坐标投影映射矩阵
	cv::Mat m_Calib_Mat_L_M;		// 左相机　内参数 K
	cv::Mat m_Calib_Mat_L_D;		// 左相机　畸变矫正 D
	cv::Mat m_Calib_Mat_L_P;		// 左相机　投影映射矩阵 P
	cv::Mat m_Calib_Mat_L_R;		// 左相机　旋转矩阵 R
	cv::Mat m_Calib_Mat_R_M;		// 右相机　内参数 K
	cv::Mat m_Calib_Mat_R_D;		// 右相机　畸变矫正 D
	cv::Mat m_Calib_Mat_R_P;		// 右相机　投影映射矩阵 P
	cv::Mat m_Calib_Mat_R_R;		// 右相机　旋转矩阵 R
        // 矫正映射矩阵
	cv::Mat m_Calib_Mat_Remap_L_X;		// 左视图畸变校正像素坐标映射矩阵 X
	cv::Mat m_Calib_Mat_Remap_L_Y;		// 左视图畸变校正像素坐标映射矩阵 Y
	cv::Mat m_Calib_Mat_Remap_R_X;		// 右视图畸变校正像素坐标映射矩阵 X
	cv::Mat m_Calib_Mat_Remap_R_Y;		// 右视图畸变校正像素坐标映射矩阵 Y

	cv::Mat m_Calib_Mat_Mask_Roi;		// 左视图校正后的有效区域
//BM 算法需要
	cv::Rect m_Calib_Roi_L;			// 左视图校正后的有效区域矩形
	cv::Rect m_Calib_Roi_R;			// 右视图校正后的有效区域矩形

        int m_maxDisparies_bm;                  // 视差变化范围
        int m_maxDisparies_sgbm;                // 视差变化范围
	int m_maxDisparies_elas;                // 视差变化范围

	int m_frameWidth;                       // 帧宽
        int m_frameHeight;                      // 帧高
        int m_numberOfDisparies;                // 视差变化范围
        int m_nViewWidth;                       // 视场宽度
        int m_nViewHeight;                      // 视场高度
        int m_nViewDepth;                       // 视场深度
};

#endif
