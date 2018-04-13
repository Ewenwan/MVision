/*
 */
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream> 
#include <iomanip>
#include <string>
#include <boost/format.hpp>  // 格式化字符串 for formating strings 处理图像文件格式
#include <boost/thread/thread.hpp>
// 参数文件解析
#include <libconfig.h++>
// 图像预处理
#include "src/imageprocessor.h"
#include <cstdlib>
//双目图像处理
#include "src/stereoprocessor.h"

//点云数据处理
#include <pcl/point_types.h> 
#include <pcl/io/pcd_io.h> //io读写
#include <pcl/visualization/pcl_visualizer.h>//可视化
#include <pcl/visualization/cloud_viewer.h>//点云可视化

// 定义点云使用的格式：这里用的是XYZRGB　即　空间位置和RGB色彩像素对
typedef pcl::PointXYZRGB PointT; //点云中的点对象  位置和像素值
typedef pcl::PointCloud<PointT> PointCloud;//整个点云对象
    
using namespace cv;
using namespace std;

//枚举符号  算法列表
enum { STEREO_BM=0, STEREO_SGBM=1, STEREO_HH=2, STEREO_VAR=3, STEREO_3WAY=4 };

// 计算点云并拼接
// 相机内参 
const double cx = 319.8;//图像像素　原点平移
const double cy = 242.6;
const double fx = 484.8;//焦距和缩放  等效
const double fy = 478.5;
const double depthScale = 10000.0;// 深度 单位归一化到m 原数据单位 为 0.1mm
const double base_line = 1627.50788;//  0.1mm 为单位
const double bf = base_line * fx / depthScale;
//double bf = base_line * fx;
const double invfx = 1.0f/fx;//
const double invfy = 1.0f/fy;
const int img_rows = 480;//图像高度
const int img_cols = 640;//图像宽度
//视差图的读取的matlab分析
//存储视差数据
// http://www.xuebuyuan.com/1271407.html
void saveDisp(const char* filename, const Mat& mat)		
{
	FILE* fp = fopen(filename, "wt");
	fprintf(fp, "%02d\n", mat.rows);
	fprintf(fp, "%02d\n", mat.cols);
	for(int y = 0; y < mat.rows; y++)
	{
		for(int x = 0; x < mat.cols; x++)
		{
			float  disp = mat.at<float>(y, x); // 这里视差矩阵是CV_16S 格式的，故用 short 类型读取
			fprintf(fp, "%f\n", disp); // 若视差矩阵是 CV_32F 格式，则用 float 类型读取
		}
	}
	fclose(fp);
}

// 灰度图转伪彩色图的代码，主要功能是使灰度图中 亮度越高的像素点 距离越近，在伪彩色图中对应的点越趋向于 红色；
// 亮度越低，则对应的伪彩色越趋向于 蓝色；总体上按照灰度值高低，由红渐变至蓝，中间色为绿色
// red   0~255 to big
// blue  0~255 big t0 0
// green 0~127to big
// green 128~255 big t0 0
int getDisparityImage(cv::Mat& disparity8, cv::Mat& disparityImage, bool isColor)
{
    // 将原始视差数据的位深转换为 8 位
    // 转换为伪彩色图像 或 灰度图像
    if (isColor)
    {
        if (disparityImage.empty() || disparityImage.type() != CV_8UC3 || disparityImage.size() != disparity8.size())
        {
            disparityImage = cv::Mat::zeros(disparity8.rows, disparity8.cols, CV_8UC3);
        }
        for (int y = 0; y<disparity8.rows; y++)//行
        {
            for (int x = 0; x<disparity8.cols; x++)//列
            {
                uchar val = disparity8.at<uchar>(y, x);//8位的视差值
                uchar r, g, b;

                if (val == 0)
                    r = g = b = 0;
                else
                {
                    b = 255 - val;//
                    g = val < 128 ? val * 2 : (uchar)((255 - val) * 2);
                    r = val;//视差值越大depth = b*f/disparity　越近 红色越足
                }
                disparityImage.at<cv::Vec3b>(y, x) = cv::Vec3b(b, g, r);
            }
        }
    }
    else
    {
        disparity8.copyTo(disparityImage);
    }

    return 1;
}

// mergeImg 原图变成灰度图为 红色通道 视差图为绿色通道
// red    img1
// blue   disp8
// green  0
Mat F_mergeImg(Mat img1, const Mat& disp8)
{
        cvtColor(img1, img1, COLOR_BGR2GRAY);//转换到 灰度图
	Mat color_mat = Mat::zeros(img1.size(), CV_8UC3);
        //int rows = color_mat.rows, cols = color_mat.cols;
	Mat red   = img1.clone();
	Mat green = disp8.clone();
	Mat blue  = Mat::zeros(img1.size(), CV_8UC1);

        vector<Mat> vM1(3);
        vM1.push_back(red);
        vM1.push_back(green);
        vM1.push_back(blue);
        //vM1.push_back(NULL);
	// 合成伪彩色图
	cv::merge(vM1, color_mat);
	return color_mat;
}
//计算生成三维点云
int getPointClouds(cv::Mat& disparity, const cv::Mat& m_Calib_Mat_Q, cv::Mat& img, PointCloud::Ptr pclpointCloud)
{
    if (disparity.empty())
    {
        return 0;
    }
    cv::Mat pointClouds;
    //计算生成三维点云
//  cv::reprojectImageTo3D(disparity, pointClouds, m_Calib_Mat_Q, true);
    reprojectImageTo3D(disparity, pointClouds, m_Calib_Mat_Q, true);

    pointClouds *= 1.6;
//为什么利用修正了的 Q 矩阵所计算得到的三维数据中， Y 坐标数据是正负颠倒的, 坐标系的问题 
    PointT p ; //点云 XYZRGB
    for (int y = 0; y < pointClouds.rows; ++y)
    {
        for (int x = 0; x < pointClouds.cols; ++x)
        {
            cv::Point3f point = pointClouds.at<cv::Point3f>(y, x);
            p.x = point.x;//现实世界中的位置坐标
            p.y = -point.y;
            p.z = point.z;
            p.b = img.data[ y*img.step + x*img.channels() ];//注意opencv彩色图像通道的顺序为 bgr
            p.g = img.data[ y*img.step + x*img.channels()+1 ];
            p.r = img.data[ y*img.step + x*img.channels()+2 ];
            pclpointCloud->points.push_back( p );
        }
    }
    return 1;
}

int getpc(cv::Mat& disparity, const cv::Mat& m_Calib_Mat_Q, cv::Mat& img, PointCloud& pointCloud){

    if (disparity.empty())
    {
       return 0;
    }
    cv::Mat pointClouds;
    //计算生成三维点云
//  cv::reprojectImageTo3D(disparity, pointClouds, m_Calib_Mat_Q, true);
    reprojectImageTo3D(disparity, pointClouds, m_Calib_Mat_Q, true);

    //pointClouds *= 1.6;
   // cout << "cloud size " <<  pointClouds.rows * pointClouds.cols << endl;
//为什么利用修正了的 Q 矩阵所计算得到的三维数据中， Y 坐标数据是正负颠倒的, 坐标系的问题 
    //PointCloud::Ptr pointCloud_PCL( new PointCloud ); 
    for (int y = 0; y < pointClouds.rows; ++y)
    {
        for (int x = 0; x < pointClouds.cols; ++x)
        {
            PointT p ; //点云 XYZRGB
            cv::Point3f point = pointClouds.at<cv::Point3f>(y, x);
            p.x = -point.x;//现实世界中的位置坐标
            p.y = -point.y;
            p.z = point.z;
            p.b = img.data[ y*img.step + x*img.channels() ];//注意opencv彩色图像通道的顺序为 bgr
            p.g = img.data[ y*img.step + x*img.channels()+1 ];
            p.r = img.data[ y*img.step + x*img.channels()+2 ];
            pointCloud.points.push_back(p);
        }
    }
    //pointCloud.is_dense = false; 
    pointCloud.width = pointCloud.points.size();
    pointCloud.height = 1;
    //cout << "cloud size " <<  pointCloud.size() << endl;
    return 1;
}
/*

 *                  				                 | Xw|
 * 			| u|     |fx  0   cx 0|     |R      T|   | Yw|
 *       		| v| =   |0   fy  cy 0|  *  |        | * | Zw|=  P*W
 * 			|1 |     |0   0   1  0|    |0 0  0 1|    | 1 |
 * http://wiki.opencv.org.cn/index.php/Cv相机标定和三维重建
 *   像素坐标齐次表示(3*1)  =  内参数矩阵 齐次表示(3*4)  ×  外参数矩阵齐次表示(4*4) ×  物体世界坐标 齐次表示(4*1)
 *   内参数齐次 × 外参数齐次 整合 得到 投影矩阵  3*4    左右两个相机 投影矩阵 P1   P2
 *  
 *    世界坐标　W 　---->　左相机投影矩阵 P1 ------> 左相机像素点　(u1,v1,1)
　*                ----> 右相机投影矩阵 P2 ------> 右相机像素点　(u2,v2,1)
　*
 *
 *  Q为 视差转深度矩阵 disparity-to-depth mapping matrix 

　Z = f*B/d       =   f    /(d/B)
 X = Z*(x-c_x)/fx = (x-c_x)/(d/B)
 Y = Z*(y-c_y)/fy = (y-c_x)/(d/B)

 *  http://blog.csdn.net/angle_cal/article/details/50800775
 *     Q= | 1   0    0         -c_x     |    Q03
 *  	  | 0   1    0         -c_y     |    Q13
 *   	  | 0   0    0          f       |    Q23
 *   	  | 0   0   -1/B   (c_x-c_x')/B |    c_x和c_x'　为左右相机　平面坐标中心的差值（内参数）
 *                   Q32        Q33
 *  以左相机光心为世界坐标系原点   左手坐标系Z  垂直向后指向 相机平面  
 *           |x|      | x-c_x           |     |X|
 *           |y|      | y-c_y           |     |Y|
 *   Q    *  |d| =    |   f             |  =  |Z|======>  Z'  =   Z/W =     f/((-d+c_x-c_x')/B)
 *           |1|      |(-d+c_x-c_x')/B  |     |W|         X'  =   X/W = ( x-c_x)/((-d+c_x-c_x')/B)
                                                          Y'  =   Y/W = ( y-c_y)/((-d+c_x-c_x')/B)
                                                         与实际值相差一个负号
 * Z = f * B/ D    f 焦距 量纲为像素点   B为双目基线距离   
 * 左右相机基线长度 量纲和标定时 所给标定板尺寸 相同  D视差 量纲也为 像素点 分子分母约去，Z的量纲同 T

[ 1., 0., 0., -3.3265590286254883e+02, 
  0., 1., 0., -2.3086411857604980e+02, 
  0., 0., 0., 3.9018919929094244e+02, 
  0., 0., 6.1428092115522364e-04, 0. ]

*/
int my_getpc(cv::Mat& disparity, const cv::Mat& m_Calib_Mat_Q, cv::Mat& leftImage, PointCloud& pointCloud){

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
            Vec3b colorValue = leftImage.at<Vec3b>(y, x);
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
// 
int ff_getpc(cv::Mat& disparity, cv::Mat& leftImage, PointCloud& pointCloud){

    if (disparity.empty())
    {
       return 0;
    }
    for (int y = 0; y < disparity.rows; ++y)// y 每一行
    {
        for (int x = 0; x < disparity.cols; ++x)// x　每一列
        {
            PointT point ; //点云 XYZRGB
            // 读取　视差　disparity
            float d = disparity.at<float>(y, x);
            //if ( d <= 0 ) continue; //Discard bad pixels
            //if ( d <= 0 ) d = disparity.at<float>(y-1, x);
            //if ( d <= 0 ) d = disparity.at<float>(y, x-1);
            //if ( d <= 0 ) d = disparity.at<float>(y, x+1);
            //if ( d <= 0 ) d = disparity.at<float>(y-1, x+1);
	    if ( d <= 0 ) continue;
            // 读取　颜色 color
            Vec3b colorValue = leftImage.at<Vec3b>(y, x);
            point.r = static_cast<int>(colorValue[2]);
            point.g = static_cast<int>(colorValue[1]);
            point.b = static_cast<int>(colorValue[0]);
            point.z = bf / (double)d;// Z = bf/d
            point.x = (double)(x - cx) * point.z * invfx;// x = Z*(x-c_x)/fx
            point.y = (double)(y - cy) * point.z * invfy;// y = Z*(y-c_y)/fy
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


// 得到深度图
void detectDepth(cv::Mat& disparity32, cv::Mat& depth)
{
    if (disparity32.empty())
    {
        return;
    }
    disparity32.copyTo(depth);
    for (int y = 0; y < disparity32.rows; ++y)
    {
        for (int x = 0; x < disparity32.cols; ++x)
        {
          float val = disparity32.at<float>(y, x);
          if(val>0)   depth.at<float>(y, x) = fx * base_line / val;
	  else  depth.at<float>(y, x) = 0;
        }
    }
}

////// 主函数//////

int main(int argc, char** argv)
{
    //Ptr<StereoBM> bm = StereoBM::create(16,9);//局部的BM;
    //Ptr<StereoSGBM> sgbm = StereoSGBM::create(0,16,3);//全局的SGBM;
    //int alg=0;
    //float scale;
    //相机内参数
    Mat M1, D1, M2, D2;
    // 相机外参数
    Mat R1, P1, R2, P2;
    Mat Q;//射影矩阵 
    Mat map11, map12, map21, map22;// 矫正映射矩阵
    //int numberOfDisparities;
    //initialize_sys(bm, sgbm, argc, argv, alg, scale, map11, map12, map21, map22, numberOfDisparities, Q);

    cv::Mat img_l, img_r, src_img;   
    cv::Mat img_l_rect, img_r_rect;//双目矫正图
    cv::Mat disp8, disp32;  // 视差图
    cv::Mat disparityImage; // 视差图有伪彩色显示

    bool readSuccessfully = false;
    libconfig::Config cfg;
    bool success = false;

    ImageProcessor iP(0.1);// 图像预处理类  像素数量阈值比率
    string cameraExtrinsic;//相机内外参数文件
//  ADCensus 算法参数
    // string savePath;
    uint dMin; uint dMax; Size censusWin;// 最小最大视差　Minimum and maximum disparity　窗口大小
    float defaultBorderCost;// 默认代价　ADCensus = AD＋Ｃensus
    float lambdaAD; float lambdaCensus;  uint aggregatingIterations;//代价聚合次数
    uint colorThreshold1; uint colorThreshold2; uint maxLength1; uint maxLength2; uint colorDifference;
    float pi1; float pi2; uint dispTolerance; uint votingThreshold; float votingRatioThreshold;
    uint maxSearchDepth; uint blurKernelSize; uint cannyThreshold1; uint cannyThreshold2; uint cannyKernelSize;
    string  config_file_name("../config/config.cfg");
//cfg参数配置文件参数读取
    if(argc >= 2) config_file_name = argv[1];
        try
        {
            cfg.readFile(config_file_name.c_str());//读取配置文件
            readSuccessfully = true;
        }
        catch(const libconfig::FileIOException &fioex)//文件读写错误
        {
            cerr << "[ADCensusCV] I/O error while reading file." << endl;
        }
        catch(const libconfig::ParseException &pex)//文件解析错误
        {
            cerr << "[ADCensusCV] Parsing error" << endl;
        }

        if(readSuccessfully)
        {
            try
            {
                dMin = (uint) cfg.lookup("dMin");//最小视差
                dMax = (uint) cfg.lookup("dMax");//最大视差
                //xmlImages = (const char *) cfg.lookup("xmlImages");
                cameraExtrinsic = (const char *) cfg.lookup("cameraExtrinsic");//内参数文件
                censusWin.height = (uint) cfg.lookup("censusWinH");//census窗口  在指定窗口内比较周围亮度值与中心点的大小
                censusWin.width = (uint) cfg.lookup("censusWinW");
                defaultBorderCost = (float) cfg.lookup("defaultBorderCost");//权重
                lambdaAD = (float) cfg.lookup("lambdaAD"); // TODO Namen anpassen 信息结合权重
                lambdaCensus = (float) cfg.lookup("lambdaCensus");// 　cost = r(Cad , lamdAD) + r(Cces, lamdCensus)
                //savePath = (const char *) cfg.lookup("savePath");
                aggregatingIterations = (uint) cfg.lookup("aggregatingIterations");// 自适应窗口代价聚合 次数
                colorThreshold1 = (uint) cfg.lookup("colorThreshold1");//搜索　自适应框时的颜色阈值
                colorThreshold2 = (uint) cfg.lookup("colorThreshold2");
                maxLength1 = (uint) cfg.lookup("maxLength1");//搜索范围阈值
                maxLength2 = (uint) cfg.lookup("maxLength2");
                colorDifference = (uint) cfg.lookup("colorDifference");
                pi1 = (float) cfg.lookup("pi1");
                pi2 = (float) cfg.lookup("pi2");
                dispTolerance = (uint) cfg.lookup("dispTolerance");
                votingThreshold = (uint) cfg.lookup("votingThreshold");
                votingRatioThreshold = (float) cfg.lookup("votingRatioThreshold");
                maxSearchDepth = (uint) cfg.lookup("maxSearchDepth");
                blurKernelSize = (uint) cfg.lookup("blurKernelSize");
                cannyThreshold1 = (uint) cfg.lookup("cannyThreshold1");
                cannyThreshold2 = (uint) cfg.lookup("cannyThreshold2");
                cannyKernelSize = (uint) cfg.lookup("cannyKernelSize");
            }
            catch(const libconfig::SettingException &ex)//未读取到部分参数
            {
                cerr << "[ADCensusCV] " << ex.what() << endl
                     << "config file format:\n"
                        "dMin(uint)\n"
                        "xmlImages(string)\n"
                        "ymlExtrinsic(string)\n"
                        "censusWinH(uint)\n"
                        "censusWinW(uint)\n"
                        "defaultBorderCost(float)\n"
                        "lambdaAD(float)\n"
                        "lambdaCensus(float)\n"
                        "savePath(string)\n"
                        "aggregatingIterations(uint)\n"
                        "colorThreshold1(uint)\n"
                        "colorThreshold2(uint)\n"
                        "maxLength1(uint)\n"
                        "maxLength2(uint)\n"
                        "colorDifference(uint)\n"
                        "pi1(float)\n"
                        "pi2(float)\n"
                        "dispTolerance(uint)\n"
                        "votingThreshold(uint)\n"
                        "votingRatioThreshold(float)\n"
                        "maxSearchDepth(uint)\n"
                        "blurKernelSize(uint)\n"
                        "cannyThreshold1(uint)\n"
                        "cannyThreshold2(uint)\n"
                        "cannyKernelSize(uint)\n";
                readSuccessfully = false;
            }
   }
 //  }
  if(readSuccessfully)
   {
   if( !cameraExtrinsic.empty() )//内参数文件
       {
	  // reading intrinsic parameters
	  FileStorage fs(cameraExtrinsic, FileStorage::READ);//读取
	  if(!fs.isOpened()){
	      printf("Failed to open file %s\n", cameraExtrinsic.c_str());
	      return -1;
	  }
	  // 内参数
	  fs["M1"] >> M1;// 左相机　内参数 K1  
	  fs["D1"] >> D1;// 左相机　畸变矫正
	  fs["M2"] >> M2;// 右相机　内参数 K1
	  fs["D2"] >> D2;// 右相机　畸变矫正
	  // 外参数  
	  fs["P1"] >> P1;// 世界坐标　W 　-->　左相机投影矩阵 P1 --> 左相机像素点　(u1,v1,1)
	  fs["P2"] >> P2;// 世界坐标　W  --> 右相机投影矩阵 P2 --> 右相机像素点　(u2,v2,1)
	  fs["R1"] >> R1;// M1 * [R1 t1] ---> P1
	  fs["R2"] >> R2;// M2 * [R2 t2] ---> P2
	  fs["Q"]  >> Q; // 射影矩阵  (u,v,d,1)转置 *　Q = W = (X,Y,Z,1) 
          if(M1.empty() || D1.empty() || P1.empty() || R1.empty() || 
	     M2.empty() || D2.empty() || P2.empty() || R2.empty() || Q.empty())
           {
             cerr << "ERROR: Calibration parameters to rectify stereo are missing!" << endl;
             return -1;
           } 
         //图像矫正摆正 映射计算  
//stereoRectify( M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, img_size, &m_Calib_Roi_L, &m_Calib_Roi_R );
// 获取矫正映射矩阵
//     cv::initUndistortRectifyMap(K_l,D_l,R_l,P_l.rowRange(0,3).colRange(0,3),cv::Size(cols_l,rows_l),CV_32F,M1l,M2l);
	  initUndistortRectifyMap(M1, D1, R1, P1, cv::Size(img_cols,img_rows), CV_32F, map11, map12);
	  initUndistortRectifyMap(M2, D2, R2, P2, cv::Size(img_cols,img_rows), CV_32F, map21, map22);
      } 
  }
if(!readSuccessfully) { cerr << "param coun't load" << endl; return -1;}

    //char depth_name[] ="depth.txt";
    cv::VideoCapture CapAll(1); //打开相机设备 
    if( !CapAll.isOpened() ) 
    { 
       printf("打开摄像头失败\r\n");
       printf("再试一次\r\n");
       //sleep(1);//延时1秒 
       cv::VideoCapture CapAll(1); //打开相机设备 
       if( !CapAll.isOpened() ) { 
         printf("打开摄像头失败\r\n");
	 printf("再试一次..\r\n");
          //sleep(1);//延时1秒 
          cv::VideoCapture CapAll(1); //打开相机设备 
          if( !CapAll.isOpened() ) { 
	  printf("打开摄像头失败\r\n");
	   return -1;
         }
       }
     }
    //设置分辨率   1280*480  分成两张 640*480  × 2 左右相机
    CapAll.set(CV_CAP_PROP_FRAME_WIDTH,1280);  
    CapAll.set(CV_CAP_PROP_FRAME_HEIGHT, 480);  
//pcl::visualization::CloudViewer viewer("pcd　viewer");// 显示窗口的名字
    int iter = 0;char c ;
 //   while(CapAll.read(src_img)) 
 //    {  
//             if(iter>255) iter =256;
//	     else iter++;
             // c = waitKey();
             //fflush(stdout);
             //cout << iter << endl;
             //if(iter<10) continue;
//             cout << iter << endl;
//      if(iter>10){
	  //   img_l = src_img(cv::Range(0, 480), cv::Range(0, 640));   //imread(file,0) 灰度图  imread(file,1) 彩色图  默认彩色图
	img_l = imread("left01.jpg");
	img_r = imread("right01.jpg");
if(img_l.empty()||img_r.empty()) {cerr << "coun't load img" << endl; return -1;}
// BM和GC算法只能对8位灰度图像计算视差，SGBM算法则可以处理24位（8bits*3）彩色图像。
// 所以在读入图像时，应该根据采用的算法来处理图像：
	  //   img_r = src_img(cv::Range(0, 480), cv::Range(640, 1280)); 
             //src_img1 = img1.clone();  
              stringstream file;
              file << 1;
	     //图像矫正 矫正原始图像
	     remap(img_l, img_l_rect, map11, map12, INTER_LINEAR);
	     remap(img_r, img_r_rect, map21, map22, INTER_LINEAR);
	     //img_l = img_l_rect;
	     //img_r = img_r_rect;
// 3*3 ksize - 核大小  1.9 * images - gauss(images )　　高斯平滑　叠加
             //img_l_rect = iP.unsharpMasking(img_l_rect, "gauss", 3, 1.9, -1);
             //img_r_rect = iP.unsharpMasking(img_r_rect, "gauss", 3, 1.9, -1);
	     StereoProcessor sP(dMin, dMax, img_l_rect, img_r_rect, 
				censusWin, defaultBorderCost, lambdaAD, 	
				lambdaCensus, file.str(), aggregatingIterations, 
				colorThreshold1, colorThreshold2, maxLength1, maxLength2,
				colorDifference, pi1, pi2, dispTolerance, votingThreshold, 
				votingRatioThreshold,maxSearchDepth, blurKernelSize, 
				cannyThreshold1, cannyThreshold2, cannyKernelSize);
             string errorMsg;
             //error = !sP.init(errorMsg);//参数检测
	     if(sP.init(errorMsg) && sP.compute())//计算视差
              {    
		 disp32 = sP.getDisparity();//得到视差 float 类型32位
              }
              disp32.convertTo(disp8, CV_8U);
              getDisparityImage(disp8, disparityImage ,true);
	     //Mat img1p, img2p, dispp, disparityImage; //防止黑边  加宽
// 对左右视图的左边进行边界延拓，以获取与原始视图相同大小的有效视差区域
	     //copyMakeBorder(img1, img1p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);
	     //copyMakeBorder(img2, img2p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);
	    //  int64 t = getTickCount();
	     // if( alg == STEREO_BM )
		//  bm->compute(img1p, img2p, dispp);
	     // else if( alg == STEREO_SGBM || alg == STEREO_HH || alg == STEREO_3WAY )
		//  sgbm->compute(img1p, img2p, dispp);
             // 截取与原始画面对应的视差区域（舍去加宽的部分）
             //disp = dispp.colRange(numberOfDisparities, img1p.cols);
	     // t = getTickCount() - t;
	     // printf("Time elapsed: %fms\n", t*1000/getTickFrequency());
	      //disp = dispp.colRange(numberOfDisparities, img1p.cols);
	      // int16  ---> uint8 
	      //if( alg != STEREO_VAR )
	      //  disp.convertTo(disp8, CV_8U, 255/(numberOfDisparities*16.));
	      //else
	      //disp.convertTo(disp8, CV_8U);
	      // 视差图有伪彩色显示
              // getDisparityImage(disp8, disparityImage ,true);

              // 计算出的视差都是CV_16S格式的，使用32位float格式可以得到真实的视差值，所以我们需要除以16
              //disp.convertTo(disp32, CV_32F, 1.0/16); 

             // Mat vdispRGB = disp8;
              //F_Gray2Color(disp8, vdispRGB);
              //Mat merge = F_mergeImg(img1, disp8);
	//      if( !no_display )
	 //     {
		  namedWindow("左相机", 1);
		  imshow("左相机", img_l_rect);
		  //namedWindow("右相机", 1);
		  //imshow("右相机", img2);
		  //namedWindow("视差8", 0);// CV_WINDOW_NORMAL
		  //imshow("视差8", disp8);
		 // namedWindow("视差32", 0);// CV_WINDOW_NORMAL
		  //imshow("视差32", disp32);
                  //PointCloud::Ptr pointCloud_PCL2( new PointCloud ); 
		 // PointCloud& pointCloud1 = *pointCloud_PCL2;
                  //getpc(disp32, Q, src_img1, pointCloud1);
                 //my_getpc(disp32, Q, src_img1, pointCloud1);
                  //ff_getpc(disp32, src_img1, pointCloud1);
                 // cout << "cloud size " <<  pointCloud1.width * pointCloud1.height<<endl;
  		 //viewer.showCloud(pointCloud_PCL2);
//pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(pointCloud_PCL2);
//viewer.addPointCloud<pcl::PointXYZRGB> (pointCloud_PCL2, rgb, "sample cloud");
		 // cv::Mat  depth;
		  //detectDepth(disp32, depth);
		  //namedWindow("深度图32", 0);// CV_WINDOW_NORMAL
		  //imshow("深度图32", depth);
		  //saveDisp(depth_name, depth);
		  namedWindow("视差伪彩色", 0);// CV_WINDOW_NORMAL
		  imshow("视差伪彩色", disparityImage);
		  //saveDisp(depth_name, disparityImage);
                  //PointCloud::Ptr pointCloud_PCL = &pointCloud; 
		  //getPointClouds(disp32, Q, img1, pointCloud_PCL);
                  //pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_color_handler (&pointCloud, 255, 0, 0);//红色
                 // viewer_ptr->addPointCloud<PointT>(pointCloud_PCL, cloud_color_handler, "original point cloud");//点云标签
		 // cout << "深度图 rows cols " << disp8.rows << " " <<  disp8.cols << endl; 
	          //namedWindow("视差彩色图", 0);
		  //imshow("视差彩色图", vdispRGB);
		//namedWindow("视差merge图", 0);
		//imshow("视差merge图", merge);
                //*/
                 //printf("press any key to continue...");
		   //while (!viewer.wasStopped())
		   //{
			//viewer.spinOnce ();
			//pcl_sleep(0.08);
		    //}
		  //fflush(stdout);
		  //char c = waitKey(0);
		  //printf("\n");
	//      } 
	      //if( c == 27 || c == 'q' || c == 'Q' )//按ESC/q/Q退出  
		//break; 
//     }
  //}
  return 0;
}
