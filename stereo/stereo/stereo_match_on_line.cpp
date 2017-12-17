/*
 *  stereo_match_on_line.cpp
 *  calibration
 * 运行 ./stereo_match_on_line
 */

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/utility.hpp"

#include <stdio.h>
#include <iostream> 

#include <boost/format.hpp>  // 格式化字符串 for formating strings 处理图像文件格式

using namespace cv;

static void print_help()
{
    printf("\nDemo stereo matching converting L and R images into disparity and point clouds\n");//生成视差和点云图
    printf("\nUsage: stereo_match  [--algorithm=bm|sgbm|hh|sgbm3way] [--blocksize=<block_size>]\n"
           "[--max-disparity=<max_disparity>] [--scale=scale_factor>] [-i=<intrinsic_filename>] [-e=<extrinsic_filename>]\n"
           "[--no-display]  \n");
}
int main(int argc, char** argv)
{
    //std::string img1_filename = "";//图像文件名
    //std::string img2_filename = "";
    std::string intrinsic_filename = "";//内参数文件名 给定
    std::string extrinsic_filename = "";//外参数文件名 给定
    std::string disparity_filename = "disparity.jpg";//视差文件名 生成
    
   //枚举符号  算法列表
    enum { STEREO_BM=0, STEREO_SGBM=1, STEREO_HH=2, STEREO_VAR=3, STEREO_3WAY=4 };
    int alg = STEREO_SGBM;
    //算法参数
    int SADWindowSize, numberOfDisparities;
    bool no_display;
    float scale;

    Ptr<StereoBM> bm = StereoBM::create(16,9);
    Ptr<StereoSGBM> sgbm = StereoSGBM::create(0,16,3);
    //参数解析
    cv::CommandLineParser parser(argc, argv,
        "{help h||}{algorithm|sgbm3way|}{max-disparity|32|}{blocksize|3|}{no-display||}{scale|1|}{i|intrinsics.yml|}{e|extrinsics.yml|}");
    if(parser.has("help"))
    {
        print_help();
        return 0;
    }
    
    if (parser.has("algorithm"))
    {
        std::string _alg = parser.get<std::string>("algorithm");
        alg = _alg == "bm" ? STEREO_BM :
            _alg == "sgbm" ? STEREO_SGBM :
            _alg == "hh" ? STEREO_HH :
            _alg == "var" ? STEREO_VAR :
            _alg == "sgbm3way" ? STEREO_3WAY : -1;
    }
    numberOfDisparities = parser.get<int>("max-disparity");
    SADWindowSize = parser.get<int>("blocksize");
    scale = parser.get<float>("scale");
    no_display = parser.has("no-display");
    if( parser.has("i") )
        intrinsic_filename = parser.get<std::string>("i");//内参
    if( parser.has("e") )
        extrinsic_filename = parser.get<std::string>("e");//外参
    if (!parser.check()) {
        parser.printErrors();
        return 1;
    }
    if( alg < 0 ) {
        printf("命令行参数错误: 未知的双目算法\n\n");
        print_help();
        return -1;
    }
    if ( numberOfDisparities < 1 || numberOfDisparities % 16 != 0 ){
        printf("命令行参数错误: 最大视差 数 为 16的倍数 正数 \n\n");// 16的倍数 正数
        print_help();
        return -1;
    }
    if (scale < 0) {
        printf("命令行参数错误: 尺度因子为正数\n\n");
        return -1;
    }
    if (SADWindowSize < 1 || SADWindowSize % 2 != 1) {
        printf("命令行参数错误: T块大小为正奇数\n\n");//正奇数
        return -1;
    }    
    if( (intrinsic_filename.empty()) || (extrinsic_filename.empty()) ) {
        printf("无相机内外参数文件\n\n");
        return -1;
    }
      
      
   int color_mode = alg == STEREO_BM ? 0 : -1;
	
  cv::Mat img1,img0;  
  cv::Mat img2;  
  cv:: Mat img1r, img2r;//双目矫正图
  cv::Mat disp, disp8;     //视差图
  //img1= src_img(cv::Range(0, 480), cv::Range(0, 640));   //imread(file,0) 灰度图  imread(file,1) 彩色图  默认彩色图
   img0 = imread("left01.jpg", 1);//图像数据
   Size img_size = img0.size();//图像尺寸
  
   Rect roi1, roi2;
   Mat Q;//射影矩阵  
   Mat map11, map12, map21, map22;
  if( !intrinsic_filename.empty() )//内参数文件
            {
		  // reading intrinsic parameters
		  FileStorage fs(intrinsic_filename, FileStorage::READ);//读取
		  if(!fs.isOpened()){
		      printf("Failed to open file %s\n", intrinsic_filename.c_str());
		      return -1;
		  }
		  //内参数
		  Mat M1, D1, M2, D2;
		  fs["M1"] >> M1;// K1  
		  fs["D1"] >> D1; // 畸变矫正
		  fs["M2"] >> M2;
		  fs["D2"] >> D2;

		  M1 *= scale;
		  M2 *= scale;

		  fs.open(extrinsic_filename, FileStorage::READ);//外参数文件
		  if(!fs.isOpened())
		  {
		      printf("Failed to open file %s\n", extrinsic_filename.c_str());
		      return -1;
		  }
		//外参数
		  Mat R, T, R1, P1, R2, P2;
		  fs["R"] >> R;
		  fs["T"] >> T;	  
		 // fs["P1"] >> P1;
		 // fs["P2"] >> P2;
		 // fs["R1"] >> R1;
		 //  fs["R2"] >> R2;		  
	         //图像矫正摆正 映射计算  
		  stereoRectify( M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2 );
		  initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
		  initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);
    } 
    
    numberOfDisparities = numberOfDisparities > 0 ? numberOfDisparities : ((img_size.width/8) + 15) & -16;
    // bm算法
    bm->setROI1(roi1);
    bm->setROI2(roi2);
    bm->setPreFilterCap(31);
    bm->setBlockSize(SADWindowSize > 0 ? SADWindowSize : 9);
    bm->setMinDisparity(0);//是控制匹配搜索的第一个参数，代表了匹配搜苏从哪里开始
    bm->setNumDisparities(numberOfDisparities);//表示最大搜索视差数
    bm->setTextureThreshold(10);
    bm->setUniquenessRatio(15);//表示匹配功能函数
    bm->setSpeckleWindowSize(100);
    bm->setSpeckleRange(32);
    bm->setDisp12MaxDiff(1);
  // sgbm算法
    sgbm->setPreFilterCap(63);
    int sgbmWinSize = SADWindowSize > 0 ? SADWindowSize : 3;
    sgbm->setBlockSize(sgbmWinSize);
    int cn = img1.channels();
    sgbm->setP1(8*cn*sgbmWinSize*sgbmWinSize);
    sgbm->setP2(32*cn*sgbmWinSize*sgbmWinSize);
    sgbm->setMinDisparity(0);//是控制匹配搜索的第一个参数，代表了匹配搜苏从哪里开始
    sgbm->setNumDisparities(numberOfDisparities);//表示最大搜索视差数
    sgbm->setUniquenessRatio(10);//表示匹配功能函数
    sgbm->setSpeckleWindowSize(100);
    sgbm->setSpeckleRange(32);
    sgbm->setDisp12MaxDiff(1);
    if(alg==STEREO_HH)               sgbm->setMode(StereoSGBM::MODE_HH);
    else if(alg==STEREO_SGBM)  sgbm->setMode(StereoSGBM::MODE_SGBM);
    else if(alg==STEREO_3WAY)   sgbm->setMode(StereoSGBM::MODE_SGBM_3WAY);
    
    cv::Mat src_img;   
    cv::VideoCapture CapAll(1); //打开相机设备 
    if( !CapAll.isOpened() ) printf("打开摄像头失败\r\n");
    //设置分辨率   1280*480  分成两张 640*480  × 2 左右相机
    CapAll.set(CV_CAP_PROP_FRAME_WIDTH,1280);  
    CapAll.set(CV_CAP_PROP_FRAME_HEIGHT, 480);  

    while(CapAll.read(src_img)) 
	{  
	     img1= src_img(cv::Range(0, 480), cv::Range(0, 640));   //imread(file,0) 灰度图  imread(file,1) 彩色图  默认彩色图
	     img2= src_img(cv::Range(0, 480), cv::Range(640, 1280));  
	     if(alg == STEREO_BM){
		    cvtColor(img1, img1, COLOR_BGR2GRAY);//转换到 灰度图
		    cvtColor(img2, img2, COLOR_BGR2GRAY);//转换到 灰度图	   
	     }
	     if (img1.empty()||img2.empty()){
		  printf("获取不到图像\n\n");
		  return -1;
	     }
	     
	     if (scale != 1.f){
		  Mat temp1, temp2;
		  int method = scale < 1 ? INTER_AREA : INTER_CUBIC;
		  resize(img1, temp1, Size(), scale, scale, method);
		  img1 = temp1;
		  resize(img2, temp2, Size(), scale, scale, method);
		  img2 = temp2;
	     }      
	    //图像矫正
	    remap(img1, img1r, map11, map12, INTER_LINEAR);
	    remap(img2, img2r, map21, map22, INTER_LINEAR);
	    img1 = img1r;
	    img2 = img2r;
	      
	      //Mat img1p, img2p, dispp;
	      //copyMakeBorder(img1, img1p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);
	      //copyMakeBorder(img2, img2p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);
	    //  int64 t = getTickCount();
	      if( alg == STEREO_BM )
		  bm->compute(img1, img2, disp);
	      else if( alg == STEREO_SGBM || alg == STEREO_HH || alg == STEREO_3WAY )
		  sgbm->compute(img1, img2, disp);
	     // t = getTickCount() - t;
	     // printf("Time elapsed: %fms\n", t*1000/getTickFrequency());
	      //disp = dispp.colRange(numberOfDisparities, img1p.cols);
	      if( alg != STEREO_VAR )
		  disp.convertTo(disp8, CV_8U, 255/(numberOfDisparities*16.));
	      else
		  disp.convertTo(disp8, CV_8U);
	//      if( !no_display )
	 //     {
		  namedWindow("左相机", 1);
		  imshow("左相机", img1);
		  namedWindow("右相机", 1);
		  imshow("右相机", img2);
		  namedWindow("视差", 0);
		  imshow("视差", disp8);
		  //printf("press any key to continue...");
		  fflush(stdout);
		  char c = waitKey();
		  printf("\n");
	//      } 
	      if( c == 27 || c == 'q' || c == 'Q' )//按ESC/q/Q退出  
		break; 
	}

    return 0;
}
