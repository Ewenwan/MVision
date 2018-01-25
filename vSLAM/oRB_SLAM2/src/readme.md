原文件
* 系统入口:
* 1】输入图像    得到 相机位置
*       单目 GrabImageMonocular(im);
*       双目 GrabImageStereo(imRectLeft, imRectRight);
*       深度 GrabImageMonocular(imRectLeft, imRectRight);
* 
* 2】转换为灰度图
*       单目 mImGray
*       双目 mImGray, imGrayRight
*       深度 mImGray, imDepth
* 
* 3】构造 帧Frame
*       单目 未初始化  Frame(mImGray, mpIniORBextractor)
*       单目 已初始化  Frame(mImGray, mpORBextractorLeft)
*       双目      Frame(mImGray, imGrayRight, mpORBextractorLeft, mpORBextractorRight)
*       深度      Frame(mImGray, imDepth,        mpORBextractorLeft)
* 
* 4】跟踪 Track
*   数据流进入 Tracking线程   Tracking.cc
