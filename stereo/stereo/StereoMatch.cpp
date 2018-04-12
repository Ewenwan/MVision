/*
opencv 双目匹配类包装
BM SGBM 

*/
#include "StereoMatch.h"

//默认构造函数　图像宽度　图像高度　配置文件
StereoMatch::StereoMatch(int imgWidth, int imgHeight, const char* xmlFilePath)
	: m_frameWidth(0), m_frameHeight(0), m_numberOfDisparies(0)
{
	//init(imgWidth, imgHeight, xmlFilePath);
}
//默认析构函数
StereoMatch::~StereoMatch(void)
{}

/*----------------------------
 * 功能 : 初始化内部变量，载入双目定标结果数据
 *----------------------------
 * 函数 : StereoMatch::init
 * 访问 : public 
 * 返回 : 0 - 载入定标数据失败，1 - 载入定标数据成功
 *
 * 参数 : imgWidth	[in]	图像宽度
 * 参数 : imgHeight	[in]	图像高度
 * 参数 : xmlFilePath	[in]	双目定标结果数据文件
 */
int StereoMatch::init(int imgWidth, int imgHeight, const char* xmlFilePath)
{
        std::cout << "Init  StereoMatch obj" << std::endl;
	m_frameWidth = imgWidth;
	m_frameHeight = imgHeight;
	m_numberOfDisparies = 0;
        m_ptr_BM   = cv::StereoBM::create(16,9);//局部的BM;
        m_ptr_SGBM = cv::StereoSGBM::create(0,16,3);//全局的SGBM;
        int ret =  loadCalibData(xmlFilePath);
	return ret;
}

/*----------------------------
 * 功能 : 载入双目标定结果数据 和算法参数
 *----------------------------
 * 函数 : StereoMatch::loadCalibData
 * 访问 : public 
 * 返回 : 1	成功
 *	 　0	读入校正参数失败
 *	 -1	定标参数的图像尺寸与当前配置的图像尺寸不一致
 *	　-99	未知错误　参数读入错误
 * 
 * 参数 : xmlFilePath	[in]	双目定标结果数据 和　算法参数　文件
 */
int StereoMatch::loadCalibData(const char* xmlFilePath)
{
   std::cout << " Loading param.." << std::endl;
   try
    {
        cv::FileStorage fs(xmlFilePath, cv::FileStorage::READ);//读取参数文件
	if ( !fs.isOpened() )
	{
          std::cerr << "ERROR: Config File Read Error!" << std::endl;  
	  return (0);
	}
	cv::Size imageSize;//图像大小
	//cv::FileNodeIterator it = fs["imageSize"].begin(); 
	//it >> imageSize.width >> imageSize.height;
        imageSize.width  = fs["Camera.width"];
        imageSize.height = fs["Camera.height"];
	if (imageSize.width != m_frameWidth || imageSize.height != m_frameHeight)
	{
                std::cerr << "ERROR: Camera Not Fit!" << std::endl;  
		return (-1);
	}
	// 内参数
	fs["M_L"] >> m_Calib_Mat_L_M;// 左相机　内参数 K1  
	fs["D_L"] >> m_Calib_Mat_L_D;// 左相机　畸变矫正
	fs["M_R"] >> m_Calib_Mat_R_M;// 右相机　内参数 K1
	fs["D_R"] >> m_Calib_Mat_R_D;// 右相机　畸变矫正
	// 外参数  
	fs["P_L"] >> m_Calib_Mat_L_P;// 世界坐标　W 　-->　左相机投影矩阵 P1 --> 左相机像素点　(u1,v1,1)
	fs["R_L"] >> m_Calib_Mat_L_R;// M1 * [R1 t1] ---> P1 左相机 旋转矩阵 R
	fs["P_R"] >> m_Calib_Mat_R_P;// 世界坐标　W  --> 右相机投影矩阵 P2 --> 右相机像素点　(u2,v2,1)
	fs["R_R"] >> m_Calib_Mat_R_R;// M2 * [R2 t2] ---> P2 右相机 旋转矩阵 R
	fs["Q"]   >> m_Calib_Mat_Q;  // 射影矩阵  (u,v,d,1)转置 *　Q = W = (X,Y,Z,1) 
        cv::Mat R, T;
        //　两相机变换矩阵
	fs["R"] >> R;// 
	fs["T"] >> T;// 
	if(m_Calib_Mat_L_M.empty() || m_Calib_Mat_L_D.empty() || m_Calib_Mat_L_P.empty() || 
	   m_Calib_Mat_L_R.empty() || m_Calib_Mat_R_M.empty() || m_Calib_Mat_R_D.empty() || 
	   m_Calib_Mat_R_P.empty() || m_Calib_Mat_R_R.empty() || m_Calib_Mat_Q.empty())
	{
	  std::cerr << "ERROR: Calibration parameters to rectify stereo are missing!" << std::endl;
	  return -1;
	} 
// 获取矫正映射矩阵
	cv::initUndistortRectifyMap(m_Calib_Mat_L_M, m_Calib_Mat_L_D, 
				    m_Calib_Mat_L_R, m_Calib_Mat_L_P, 
				    imageSize, CV_32F, 
	 			    m_Calib_Mat_Remap_L_X, m_Calib_Mat_Remap_L_Y);
	cv::initUndistortRectifyMap(m_Calib_Mat_R_M, m_Calib_Mat_R_D, 
				    m_Calib_Mat_R_R, m_Calib_Mat_R_P, 
				    imageSize, CV_32F, 
				    m_Calib_Mat_Remap_R_X, m_Calib_Mat_Remap_R_Y);
// 计算左视图校正后的有效区域矩形
 	cv::stereoRectify( m_Calib_Mat_L_M, m_Calib_Mat_L_D, m_Calib_Mat_R_M, m_Calib_Mat_R_D,
			   imageSize, R, T, m_Calib_Mat_L_R, m_Calib_Mat_R_R, m_Calib_Mat_L_P, 
		           m_Calib_Mat_R_P, m_Calib_Mat_Q, CALIB_ZERO_DISPARITY, -1, imageSize, 			   &m_Calib_Roi_L, &m_Calib_Roi_R );

	int bmPreFilterSize, bmPreFilterCap, bmBlockSize, 
	    bmNumDisparities, bmTextureThreshold, bmUniquenessRatio;
	fs["BM.preFilterSize"]    >> bmPreFilterSize;
	fs["BM.preFilterCap"]     >> bmPreFilterCap;
	fs["BM.blockSize"]        >> bmBlockSize;
	fs["BM.numDisparities"]   >> bmNumDisparities;
	fs["BM.textureThreshold"] >> bmTextureThreshold;
	fs["BM.uniquenessRatio"]  >> bmUniquenessRatio;
	m_maxDisparies_bm = bmNumDisparities;

        if ( m_maxDisparies_bm < 1 || m_maxDisparies_bm % 16 != 0 ){
            printf("命令行参数错误: 最大视差 数 为 16的倍数 正数 \n\n");// 16的倍数 正数
            return -1;
         }
	// bm算法
	m_ptr_BM->setROI1(m_Calib_Roi_L);//左右视图的有效像素区域 在有效视图之外的视差值被消零
	m_ptr_BM->setROI2(m_Calib_Roi_R);
	m_ptr_BM->setPreFilterType(CV_STEREO_BM_XSOBEL);
	m_ptr_BM->setPreFilterSize(bmPreFilterSize > 5 ? bmPreFilterSize : 9);//滤波器尺寸 [5,255]奇数
	m_ptr_BM->setPreFilterCap(bmPreFilterCap > 10 ? bmPreFilterCap : 31);//预处理滤波器的截断值 [1-31] 
	m_ptr_BM->setBlockSize(bmBlockSize > 0 ? bmBlockSize : 15);//sad窗口大小
	m_ptr_BM->setMinDisparity(0);//最小视差值，代表了匹配搜索从哪里开始
	m_ptr_BM->setNumDisparities(bmNumDisparities > 0 ? bmNumDisparities : 60);//表示最大搜索视差数
	m_ptr_BM->setTextureThreshold(bmTextureThreshold > 0 ? bmTextureThreshold : 10);
	//低纹理区域的判断阈值 x方向导数绝对值之和小于阈值 100 1000
	m_ptr_BM->setUniquenessRatio(bmUniquenessRatio > 0 ? bmUniquenessRatio : 15);
	//视差唯一性百分比  使用匹配功能模式  
	m_ptr_BM->setSpeckleWindowSize(100);//检查视差连通域 变化度的窗口大小
	m_ptr_BM->setSpeckleRange(32);//视差变化阈值  当窗口内视差变化大于阈值时，该窗口内的视差清零
	m_ptr_BM->setDisp12MaxDiff(-1);// -1
	//左视图差（直接计算）和右视图差（cvValidateDisparity计算得出）之间的最大允许差异 默认为-1  
	// 对视差生成效果影响较大的主要参数是setSADWindowSize、
	//setNumberDisparities和 setUniquenessRatio，这三个参数要重点关注和调整。其他参数的影响不算很大。


	int sgbmPreFilterCap, sgbmBlockSize, sgbmNumDisparities, 
	    sgbmTextureThreshold, sgbmUniquenessRatio;
	fs["SGBM.preFilterCap"]     >> sgbmPreFilterCap;
	fs["SGBM.blockSize"]        >> sgbmBlockSize;
	fs["SGBM.numDisparities"]   >> sgbmNumDisparities;
	fs["SGBM.uniquenessRatio"]  >> sgbmUniquenessRatio;
	m_maxDisparies_sgbm = sgbmNumDisparities;

        if ( m_maxDisparies_sgbm < 1 || m_maxDisparies_sgbm % 16 != 0 ){
            printf("命令行参数错误: 最大视差 数 为 16的倍数 正数 \n\n");// 16的倍数 正数
            return -1;
         }
	// sgbm算法
	m_ptr_SGBM->setPreFilterCap(sgbmPreFilterCap > 10 ? sgbmPreFilterCap : 63);//预处理滤波器的截断值 [1-63] 
	int sgbmWinSize = sgbmBlockSize > 0 ? sgbmBlockSize : 3;
	m_ptr_SGBM->setBlockSize(sgbmWinSize);
	int cn = 3;//彩色图
	m_ptr_SGBM->setP1(8*cn*sgbmWinSize*sgbmWinSize);
    // 控制视差变化平滑性的参数。P1、P2的值越大，视差越平滑。
    //P1是相邻像素点视差增/减 1 时的惩罚系数；P2是相邻像素点视差变化值大于1时的惩罚系数。P2必须大于P1
	m_ptr_SGBM->setP2(32*cn*sgbmWinSize*sgbmWinSize);
	m_ptr_SGBM->setMinDisparity(0);//最小视差值，代表了匹配搜索从哪里开始
	m_ptr_SGBM->setNumDisparities(sgbmNumDisparities > 0 ? sgbmNumDisparities : 60);//表示最大搜索视差数
	m_ptr_SGBM->setUniquenessRatio(sgbmUniquenessRatio > 0 ? sgbmUniquenessRatio : 10);//表示匹配功能函数
	m_ptr_SGBM->setSpeckleWindowSize(100);//检查视差连通域 变化度的窗口大小
	m_ptr_SGBM->setSpeckleRange(32);//视差变化阈值  当窗口内视差变化大于阈值时，该窗口内的视差清零
	m_ptr_SGBM->setDisp12MaxDiff(1);// -1
	m_ptr_SGBM->setMode(StereoSGBM::MODE_SGBM);// StereoSGBM::MODE_HH  StereoSGBM::MODE_SGBM_3WAY


	m_Calib_Data_Loaded = true;

    }
    catch (std::exception& e)
    {
       std::cerr << "ERROR: Config File Read Error!" << std::endl;
       m_Calib_Data_Loaded = false;
       return (-99);	
    }
    std::cout << "Load param successful " << std::endl;
	
    return 1;
}


/*----------------------------
 * 功能 : 基于 BM 算法计算视差
 *----------------------------
 * 函数 : StereoMatch::bmMatch
 * 访问 : public 
 * 返回 : 0 - 失败，1 - 成功
 *
 * 参数 : frameLeft		[in]	左摄像机帧图
 * 参数 : frameRight		[in]	右摄像机帧图
 * 参数 : disparity		[out]	视差图
 */
int StereoMatch::bmMatch(cv::Mat& frameLeft, cv::Mat& frameRight, cv::Mat& disparity)
{
	// 输入检查
	if (frameLeft.empty() || frameRight.empty())
	{
		disparity = cv::Scalar(0);
		return 0;
	}

	// BM只能处理单通道图像　转换为灰度图
	cv::Mat imgLproc, imgRproc;
	cv::cvtColor(frameLeft,  imgLproc, COLOR_BGR2GRAY);
	cv::cvtColor(frameRight, imgRproc, COLOR_BGR2GRAY);

	// 校正图像，使左右视图行对齐	
	cv::Mat imgLremap, imgRremap;
	if (m_Calib_Data_Loaded)
	{
	  cv::remap(imgLproc, imgLremap, m_Calib_Mat_Remap_L_X, 
	        m_Calib_Mat_Remap_L_Y, cv::INTER_LINEAR);// 对用于视差计算的画面进行校正
	  cv::remap(imgRproc, imgRremap, m_Calib_Mat_Remap_R_X, 
	        m_Calib_Mat_Remap_R_Y, cv::INTER_LINEAR);
	} 
	else
	{
	  imgLremap = imgLproc;
	  imgRremap = imgRproc;
	}

	// 对左右视图的左边进行边界延拓，以获取与原始视图相同大小的有效视差区域
	cv::Mat imgLborder, imgRborder;
	//if (m_numberOfDisparies != m_ptr_BM->params.numDisparities)
	//	m_numberOfDisparies = m_ptr_BM->params.numDisparities;
	copyMakeBorder( imgLremap, imgLborder, 0, 0, 
			m_maxDisparies_bm, 0, IPL_BORDER_REPLICATE);
	copyMakeBorder( imgRremap, imgRborder, 0, 0, 
			m_maxDisparies_bm, 0, IPL_BORDER_REPLICATE);
        m_numberOfDisparies = m_maxDisparies_bm;
	// 计算视差 带有扩展的边框
	cv::Mat dispBorder;
	m_ptr_BM->compute(imgLborder, imgRborder, dispBorder);

	// 截取与原始画面对应的视差区域（舍去加宽的部分）
	cv::Mat disp;
	disp = dispBorder.colRange(m_numberOfDisparies, imgLborder.cols);	
	//disp.copyTo(disparity, m_Calib_Mat_Mask_Roi);
        disp.copyTo(disparity);
	// 输出处理后的图像
	//if (m_Calib_Data_Loaded)
	//	remap(frameLeft, imageLeft, m_Calib_Mat_Remap_X_L, m_Calib_Mat_Remap_Y_L, cv::INTER_LINEAR);
	//else
	//	frameLeft.copyTo(imageLeft);
	//rectangle(imageLeft, m_Calib_Roi_L, CV_RGB(0,0,255), 3);
//
	//if (m_Calib_Data_Loaded)
	//	remap(frameRight, imageRight, m_Calib_Mat_Remap_X_R, m_Calib_Mat_Remap_Y_R, cv::INTER_LINEAR);
	//else
	//	frameRight.copyTo(imageRight);
	//rectangle(imageRight, m_Calib_Roi_R, CV_RGB(0,0,255), 3);

	return 1;
}


/*----------------------------
 * 功能 : 基于 SGBM 算法计算视差
 *----------------------------
 * 函数 : StereoMatch::sgbmMatch
 * 访问 : public 
 * 返回 : 0 - 失败，1 - 成功
 *
 * 参数 : frameLeft		[in]	左摄像机帧图
 * 参数 : frameRight		[in]	右摄像机帧图
 * 参数 : disparity		[out]	视差图
 */
int StereoMatch::sgbmMatch(cv::Mat& frameLeft, cv::Mat& frameRight, cv::Mat& disparity)
{
	// 输入检查
	if (frameLeft.empty() || frameRight.empty())
	{
		disparity = cv::Scalar(0);
		return 0;
	}

	// 复制图像
	cv::Mat imgLproc, imgRproc;
	frameLeft.copyTo(imgLproc);
	frameRight.copyTo(imgRproc);

	// 校正图像，使左右视图行对齐	
	cv::Mat imgLremap, imgRremap;
	if (m_Calib_Data_Loaded)
	{
	  cv::remap(imgLproc, imgLremap, m_Calib_Mat_Remap_L_X, 
	        m_Calib_Mat_Remap_L_Y, cv::INTER_LINEAR);// 对用于视差计算的画面进行校正
	  cv::remap(imgRproc, imgRremap, m_Calib_Mat_Remap_R_X, 
	        m_Calib_Mat_Remap_R_Y, cv::INTER_LINEAR);
	} 
	else
	{
	  imgLremap = imgLproc;
	  imgRremap = imgRproc;
	}

	// 对左右视图的左边进行边界延拓，以获取与原始视图相同大小的有效视差区域
	cv::Mat imgLborder, imgRborder;
	//if (m_numberOfDisparies != m_ptr_SGBM->params.numDisparities)
	//	m_numberOfDisparies = m_ptr_SGBM->params.numDisparities;
	copyMakeBorder( imgLremap, imgLborder, 0, 0, 
			m_maxDisparies_sgbm, 0, IPL_BORDER_REPLICATE);
	copyMakeBorder( imgRremap, imgRborder, 0, 0, 
			m_maxDisparies_sgbm, 0, IPL_BORDER_REPLICATE);
        m_numberOfDisparies = m_maxDisparies_sgbm;
	// 计算视差 带有扩展的边框
	cv::Mat dispBorder;
	m_ptr_SGBM->compute(imgLborder, imgRborder, dispBorder);

	// 截取与原始画面对应的视差区域（舍去加宽的部分）
	cv::Mat disp;
	disp = dispBorder.colRange(m_numberOfDisparies, imgLborder.cols);	
	//disp.copyTo(disparity, m_Calib_Mat_Mask_Roi);
	disp.copyTo(disparity);

	// 输出处理后的图像
	//imageLeft = img1remap.clone();
	//imageRight = img2remap.clone();
	//rectangle(imageLeft, m_Calib_Roi_L, CV_RGB(0,255,0), 3);
	//rectangle(imageRight, m_Calib_Roi_R, CV_RGB(0,255,0), 3);

	return 1;
}


/*----------------------------
 * 功能 : 基于 VAR 算法计算视差
 *----------------------------
 * 函数 : StereoMatch::varMatch
 * 访问 : public 
 * 返回 : 0 - 失败，1 - 成功
 *
 * 参数 : frameLeft		[in]	左摄像机帧图
 * 参数 : frameRight		[in]	右摄像机帧图
 * 参数 : disparity		[out]	视差图
 * 参数 : imageLeft		[out]	处理后的左视图，用于显示
 * 参数 : imageRight		[out]	处理后的右视图，用于显示
 */
/*
int StereoMatch::varMatch(cv::Mat& frameLeft, cv::Mat& frameRight, cv::Mat& disparity, cv::Mat& imageLeft, cv::Mat& imageRight)
{
	// 输入检查
	if (frameLeft.empty() || frameRight.empty())
	{
		disparity = cv::Scalar(0);
		return 0;
	}

	// 复制图像
	cv::Mat img1proc, img2proc;
	frameLeft.copyTo(img1proc);
	frameRight.copyTo(img2proc);

	// 校正图像，使左右视图行对齐	
	cv::Mat img1remap, img2remap;
	if (m_Calib_Data_Loaded)
	{
		remap(img1proc, img1remap, m_Calib_Mat_Remap_X_L, m_Calib_Mat_Remap_Y_L, cv::INTER_LINEAR);		// 对用于视差计算的画面进行校正
		remap(img2proc, img2remap, m_Calib_Mat_Remap_X_R, m_Calib_Mat_Remap_Y_R, cv::INTER_LINEAR);
	} 
	else
	{
		img1remap = img1proc;
		img2remap = img2proc;
	}

	// 对左右视图的左边进行边界延拓，以获取与原始视图相同大小的有效视差区域
	cv::Mat img1border, img2border;
	if (m_numberOfDisparies != m_VAR.maxDisp)
		m_numberOfDisparies = m_VAR.maxDisp;
	copyMakeBorder(img1remap, img1border, 0, 0, m_VAR.maxDisp, 0, IPL_BORDER_REPLICATE);
	copyMakeBorder(img2remap, img2border, 0, 0, m_VAR.maxDisp, 0, IPL_BORDER_REPLICATE);

	// 计算视差
	cv::Mat dispBorder;
	m_VAR(img1border, img2border, dispBorder);

	// 截取与原始画面对应的视差区域（舍去加宽的部分）
	cv::Mat disp;
	disp = dispBorder.colRange(m_VAR.maxDisp, img1border.cols);	
	disp.copyTo(disparity, m_Calib_Mat_Mask_Roi);

	// 输出处理后的图像
	imageLeft = img1remap.clone();
	imageRight = img2remap.clone();
	rectangle(imageLeft, m_Calib_Roi_L, CV_RGB(0,255,0), 3);
	rectangle(imageRight, m_Calib_Roi_R, CV_RGB(0,255,0), 3);

	return 1;
}
*/

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
int StereoMatch::getPointClouds(cv::Mat& disparity, cv::Mat& pointClouds)
{
	if (disparity.empty())
	{
		return 0;
	}

	//计算生成三维点云
	cv::reprojectImageTo3D(disparity, pointClouds, m_Calib_Mat_Q, true);
        pointClouds *= 1.6;
	
	// 校正 Y 方向数据，正负反转
	// 原理参见：http://blog.csdn.net/chenyusiyuan/article/details/5970799 
	for (int y = 0; y < pointClouds.rows; ++y)
	{
		for (int x = 0; x < pointClouds.cols; ++x)
		{
		 cv::Point3f point = pointClouds.at<cv::Point3f>(y,x);
                 point.y = -point.y;// Y 坐标数据是正负颠倒的
		 pointClouds.at<cv::Point3f>(y,x) = point;
		}
	}

	return 1;
}
int StereoMatch::getPCL(cv::Mat& disparity, cv::Mat& img_L, PointCloud& pointCloud)
{
	if (disparity.empty())
	{
		return 0;
	}
    cv::Mat pointClouds;
    //计算生成三维点云
    cv::reprojectImageTo3D(disparity, pointClouds, m_Calib_Mat_Q, true);
    //pointClouds *= 1.6;//最好不要乘
    for (int y = 0; y < pointClouds.rows; ++y)
    {
        for (int x = 0; x < pointClouds.cols; ++x)
        {
            PointT p ; //点云 XYZRGB
            cv::Point3f point = pointClouds.at<cv::Point3f>(y, x);//坐标
            cv::Vec3b colorValue = img_L.at<Vec3b>(y, x);//颜色值
            p.x = point.x;//现实世界中的位置坐标
            p.y = -point.y;//Y 坐标数据是正负颠倒的
            p.z = point.z;
            p.b = static_cast<int>(colorValue[0]);
            p.g = static_cast<int>(colorValue[1]);
            p.r = static_cast<int>(colorValue[2]);
            pointCloud.points.push_back(p);
        }
    }
    //pointCloud.is_dense = false; 
    pointCloud.width = pointCloud.points.size();
    pointCloud.height = 1;
    cout << "cloud size " <<  pointCloud.size() << endl;
    return 1;
}
int StereoMatch::my_getpc(cv::Mat& disparity, cv::Mat& img_L, PointCloud& pointCloud){

    if (disparity.empty())
    {
       return 0;
    }
    // Read out Q Values for faster access
    double Q03 = m_Calib_Mat_Q.at<double>(0, 3);// -c_x
    double Q13 = m_Calib_Mat_Q.at<double>(1, 3);// -c_y
    double Q23 = m_Calib_Mat_Q.at<double>(2, 3);// f
    double Q32 = m_Calib_Mat_Q.at<double>(3, 2);// -1/B
    double Q33 = m_Calib_Mat_Q.at<double>(3, 3);// (c_x-c_x')/B

    for (int y = 0; y < disparity.rows; ++y)// y 每一行
    {
        for (int x = 0; x < disparity.cols; ++x)// x　每一列
        {
            PointT point ; //点云 XYZRGB
            // 读取　视差　disparity
            float d = disparity.at<float>(y, x);
// 这里好像有问题
            //if ( d <= 0 ) d = disparity.at<float>(y-1, x);
            //if ( d <= 0 ) d = disparity.at<float>(y, x-1);
            //if ( d <= 0 ) d = disparity.at<float>(y, x+1);
            //if ( d <= 0 ) d = disparity.at<float>(y-1, x+1);

            if ( d <= 0 ) continue; //Discard bad pixels
            // 读取　颜色 color
            Vec3b colorValue = img_L.at<Vec3b>(y, x);
            point.r = static_cast<int>(colorValue[2]);
            point.g = static_cast<int>(colorValue[1]);
            point.b = static_cast<int>(colorValue[0]);
            // Transform 2D -> 3D and normalise to point
            double xx = Q03 + x;// x - c_x
            double yy = Q13 + y;// y - c_y
            double zz = Q23;    // f
            double w = (Q32 * d) + Q33;//-1/B * d + (c_x-c_x')/B
            point.x = -xx / w;//注意上面得出的为负值
            point.y = -yy / w;
            point.z = zz / w;
            pointCloud.points.push_back(point);
        }
    }
    // pointCloud.is_dense = false; 
    // Resize PCL and save to file
    pointCloud.width = pointCloud.points.size();
    pointCloud.height = 1;
    cout << "cloud size " <<  pointCloud.size() << endl;
    return 1;
}

/*----------------------------
 * 功能 : 获取伪彩色视差图
 *----------------------------
 * 函数 : StereoMatch::getDisparityImage
 * 访问 : public 
 * 返回 : 0 - 失败，1 - 成功
 *
 * 参数 : disparity	 [in]	原始视差数据
 * 参数 : disparityImage [out]	伪彩色视差图
 * 参数 : isColor	 [in]	是否采用伪彩色，默认为 true，设为 false 时返回灰度视差图
 */
int StereoMatch::getDisparityImage(cv::Mat& disparity, cv::Mat& disparityImage, bool isColor)
{
	// 将原始视差数据的位深转换为 8 位
	cv::Mat disp8u;
	if (disparity.depth() != CV_8U)
	{
		if (disparity.depth() == CV_8S)//
		{
			disparity.convertTo(disp8u, CV_8U);
		} 
		else
		{
			disparity.convertTo(disp8u, CV_8U, 255/(m_numberOfDisparies*16.));
		}
	} 
	else
	{
		disp8u = disparity;
	}

	// 转换为伪彩色图像 或 灰度图像
	if (isColor)
	{
		if (disparityImage.empty() || disparityImage.type() != CV_8UC3 || 
		    disparityImage.size() != disparity.size())
		{
			disparityImage = cv::Mat::zeros(disparity.rows, disparity.cols, CV_8UC3);
		}

		for (int y=0;y<disparity.rows;y++)
		{
			for (int x=0;x<disparity.cols;x++)
			{
				uchar val = disp8u.at<uchar>(y,x);
				uchar r,g,b;

				if (val==0)//视差为0定为黑色　无穷远　这里未匹配到 
					r = g = b = 0;
				else
				{
					b = 255-val;
					g = val < 128 ? val*2 : (uchar)((255 - val)*2);
					r = val;//视差值越大depth = b*f/disparity　越近 红色越足
				}

				disparityImage.at<cv::Vec3b>(y,x) = cv::Vec3b(b,g,r);
			}
		}
	} 
	else
	{
		disp8u.copyTo(disparityImage);
	}
	return 1;
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
void StereoMatch::getTopDownView(cv::Mat& pointClouds, cv::Mat& topDownView, cv::Mat& image)
{
    int VIEW_WIDTH = m_nViewWidth, VIEW_DEPTH = m_nViewDepth;
    cv::Size mapSize = cv::Size(VIEW_DEPTH, VIEW_WIDTH);

    if (topDownView.empty() || topDownView.size() != mapSize || topDownView.type() != CV_8UC3)
        topDownView = cv::Mat(mapSize, CV_8UC3);

    topDownView = cv::Scalar::all(50);

    if (pointClouds.empty())
        return;

    if (image.empty() || image.size() != pointClouds.size())
        image = 255 * cv::Mat::ones(pointClouds.size(), CV_8UC3);
    
    for(int y = 0; y < pointClouds.rows; y++)
    {
        for(int x = 0; x < pointClouds.cols; x++)
        {
            cv::Point3f point = pointClouds.at<cv::Point3f>(y, x);
            int pos_Z = point.z;

            if ((0 <= pos_Z) && (pos_Z < VIEW_DEPTH))
            {
                int pos_X = point.x + VIEW_WIDTH/2;
                if ((0 <= pos_X) && (pos_X < VIEW_WIDTH))
                {
                    topDownView.at<cv::Vec3b>(pos_X,pos_Z) = image.at<cv::Vec3b>(y,x);
                }
            }
        }
    }
}
    
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
void StereoMatch::getSideView(cv::Mat& pointClouds, cv::Mat& sideView, cv::Mat& image)
{
    int VIEW_HEIGTH = m_nViewHeight, VIEW_DEPTH = m_nViewDepth;
    cv::Size mapSize = cv::Size(VIEW_DEPTH, VIEW_HEIGTH);

    if (sideView.empty() || sideView.size() != mapSize || sideView.type() != CV_8UC3)
        sideView = cv::Mat(mapSize, CV_8UC3);
    
    sideView = cv::Scalar::all(50);

    if (pointClouds.empty())
        return;

    if (image.empty() || image.size() != pointClouds.size())
        image = 255 * cv::Mat::ones(pointClouds.size(), CV_8UC3);

    for(int y = 0; y < pointClouds.rows; y++)
    {
        for(int x = 0; x < pointClouds.cols; x++)
        {
            cv::Point3f point = pointClouds.at<cv::Point3f>(y, x);
            int pos_Y = -point.y + VIEW_HEIGTH/2;
            int pos_Z = point.z;

            if ((0 <= pos_Z) && (pos_Z < VIEW_DEPTH))
            {
                if ((0 <= pos_Y) && (pos_Y < VIEW_HEIGTH))
                {
                    sideView.at<cv::Vec3b>(pos_Y,pos_Z) = image.at<cv::Vec3b>(y,x);
                }
            }
        }
    }
}


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
void StereoMatch::savePointClouds(cv::Mat& pointClouds, const char* filename)
{
	const double max_z = 1.0e4;
	try
	{
		FILE* fp = fopen(filename, "wt");
		for(int y = 0; y < pointClouds.rows; y++)
		{
			for(int x = 0; x < pointClouds.cols; x++)
			{
				cv::Vec3f point = pointClouds.at<cv::Vec3f>(y, x);
				if(fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z)
					fprintf(fp, "%d %d %d\n", 0, 0, 0);
				else
					fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
			}
		}
		fclose(fp);
	}
	catch (std::exception* e)
	{
		printf("Failed to save point clouds. Error: %s \n\n", e->what());
	}
}
