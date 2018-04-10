/*
 * 双目匹配 + 生成点云
 * 运行 ./stereo_match_on_line
参考代码　https://github.com/yuhuazou/StereoVision/blob/master/StereoVision/StereoMatch.cpp
https://blog.csdn.net/hujingshuang/article/details/47759579

http://blog.sina.com.cn/s/blog_c3db2f830101fp2l.html

视差快匹配代价函数：
	【1】对应像素差的绝对值和（SAD, Sum of Absolute Differences）   
	【2】对应像素差的平方和（SSD, Sum of Squared Differences）
	【3】图像的相关性（NCC, Normalized Cross Correlation） 归一化积相关算法
	【4】ADCensus 代价计算  = AD＋Census
             1.  AD即 Absolute Difference 三通道颜色差值绝对值之和求均值
	     2. Census Feature特征原理很简单
　　　　　　　         在指定窗口内比较周围亮度值与中心点的大小，匹配距离用汉明距表示。
　　　　　　　　　　　　　   Census保留周边像素空间信息，对光照变化有一定鲁棒性。
	     3. 信息结合
　　　　　　　　　　　　　　　　cost = r(Cad , lamd1) + r(Cces, lamd2)
                r(C , lamd) = 1 - exp(- c/ lamd)
           　 　　Cross-based 代价聚合:
                自适应窗口代价聚合，
			在设定的最大窗口范围内搜索，
　　　　　　　　　　　　　满足下面三个约束条件确定每个像素的十字坐标，完成自适应窗口的构建。
　　　　　　　　　　　　　　　Scanline 代价聚合优化

 种方法就是以左目图像的源匹配点为中心，

视差获取

对于区域算法来说，在完成匹配代价的叠加以后，视差的获取就很容易了，
只需在一定范围内选取叠加匹配代价最优的点
（SAD和SSD取最小值，NCC取最大值）作为对应匹配点，
如胜者为王算法WTA（Winner-take-all）。
而全局算法则直接对原始匹配代价进行处理，一般会先给出一个能量评价函数，
然后通过不同的优化算法来求得能量的最小值，同时每个点的视差值也就计算出来了。
大多数立体匹配算法计算出来的视差都是一些离散的特定整数值，可满足一般应用的精度要求。
但在一些精度要求比较高的场合，如精确的三维重构中，就需要在初始视差获取后采用一些措施对视差进行细化，
如匹配代价的曲线拟合、图像滤波、图像分割等。

八、立体匹配约束
	1）极线约束
	2）唯一性约束
	3）视差连续性约束
	4）顺序一致性约束
	5）相似性约束

九、相似性判断标准
	1）像素点灰度差的平方和，即 SSD
	2）像素点灰度差的绝对值和，即 SAD
	3）归一化交叉相关，简称 NCC
	4） 零均值交叉相关，即 ZNCC
	5）Moravec 非归一化交叉相关，即 MNCC
	6）Kolmogrov-Smrnov 距离，即 KSD
	7）Jeffrey 散度
	8）Rank 变换（是以窗口内灰度值小于中心像素灰度值的像素个数来代替中心像素的灰度值）
	9）Census 变换（是根据窗口内中心像素灰度与其余像素灰度值的大小关系得一串位码，位码长度等于窗口内像素个数减一）


SAD方法就是以左目图像的源匹配点为中心，定义一个窗口D，其大小为（2m+1） (2n+1)，
统计其窗口的灰度值的和，然后在右目图像中逐步计算其左右窗口的灰度和的差值，
最后搜索到的差值最小的区域的中心像素即为匹配点。
基本流程：
	1.构造一个小窗口，类似与卷积核。
	2.用窗口覆盖左边的图像，选择出窗口覆盖区域内的所有像素点。
	3.同样用窗口覆盖右边的图像并选择出覆盖区域的像素点。
	4.左边覆盖区域减去右边覆盖区域，并求出所有像素点差的绝对值的和。
	5.移动右边图像的窗口，重复3，4的动作。（这里有个搜索范围，超过这个范围跳出）
	6.找到这个范围内SAD值最小的窗口，即找到了左边图像的最佳匹配的像素块。

 由 以上三种算法可知，SAD算法最简单，因此当模板大小确定后，SAD算法的速度最快。NCC算法与SAD算法相比要复杂得多。
------------------------------------
SAD（Sum of Absolute Difference）=SAE（Sum of Absolute Error)即绝对误差和
SSD（Sum of Squared Difference）=SSE（Sum of Squared Error)即差值的平方和
SATD（Sum of Absolute Transformed Difference）即hadamard变换后再绝对值求和
MAD（Mean Absolute Difference）=MAE（Mean Absolute Error)即平均绝对差值
MSD（Mean Squared Difference）=MSE（Mean Squared Error）即平均平方误差


三角测量原理：
 现实世界物体坐标　—(外参数 变换矩阵Ｔ变换)—>  相机坐标系　—(同/Z)—>归一化平面坐标系——>径向和切向畸变矫正——>(内参数平移　Cx Cy 缩放焦距Fx Fy)
 ——> 图像坐标系下　像素坐标
 u=Fx *X/Z + Cx 　　像素列位置坐标　
 v=Fy *Y/Z + Cy 　　像素列位置坐标　
 
 反过来
 X=(u- Cx)*Z/Fx
 Y=(u- Cy)*Z/Fy
 Z轴归一化
 X=(u- Cx)*Z/Fx/depthScale
 Y=(u- Cy)*Z/Fy/depthScale
 Z=Z/depthScale
 
外参数　T
世界坐标　
pointWorld = T*[X Y Z]
 
OpenCV三种立体匹配求视差图算法总结:

首先我们看一下BM算法：
    Ptr<StereoBM> bm = StereoBM::create(16,9);//局部的BM;
    // bm算法
    bm->setROI1(roi1);//左右视图的有效像素区域 在有效视图之外的视差值被消零
    bm->setROI2(roi2);
    bm->setPreFilterType(CV_STEREO_BM_XSOBEL);
    bm->setPreFilterSize(9);//滤波器尺寸 [5,255]奇数
    bm->setPreFilterCap(31);//预处理滤波器的截断值 [1-31] 
    bm->setBlockSize(SADWindowSize > 0 ? SADWindowSize : 15);//sad窗口大小
    bm->setMinDisparity(0);//最小视差值，代表了匹配搜索从哪里开始
    bm->setNumDisparities(numberOfDisparities);//表示最大搜索视差数
    bm->setTextureThreshold(10);//低纹理区域的判断阈值 x方向导数绝对值之和小于阈值
    bm->setUniquenessRatio(15);//视差唯一性百分比  匹配功能函数
    bm->setSpeckleWindowSize(100);//检查视差连通域 变化度的窗口大小
    bm->setSpeckleRange(32);//视差变化阈值  当窗口内视差变化大于阈值时，该窗口内的视差清零
    bm->setDisp12MaxDiff(-1);// 1

该方法速度最快，一副320*240的灰度图匹配时间为31ms 



第二种方法是SGBM方法这是OpenCV的一种新算法：
Ptr<StereoSGBM> sgbm = StereoSGBM::create(0,16,3);//全局的SGBM;
 // sgbm算法
    sgbm->setPreFilterCap(63);//预处理滤波器的截断值 [1-63] 
    int sgbmWinSize = SADWindowSize > 0 ? SADWindowSize : 3;
    sgbm->setBlockSize(sgbmWinSize);
    int cn = img0.channels();
    sgbm->setP1(8*cn*sgbmWinSize*sgbmWinSize);// 控制视差变化平滑性的参数。P1、P2的值越大，视差越平滑。
//P1是相邻像素点视差增/减 1 时的惩罚系数；P2是相邻像素点视差变化值大于1时的惩罚系数。P2必须大于P1
    sgbm->setP2(32*cn*sgbmWinSize*sgbmWinSize);
    sgbm->setMinDisparity(0);//最小视差值，代表了匹配搜索从哪里开始
    sgbm->setNumDisparities(numberOfDisparities);//表示最大搜索视差数
    sgbm->setUniquenessRatio(10);//表示匹配功能函数
    sgbm->setSpeckleWindowSize(100);//检查视差连通域 变化度的窗口大小
    sgbm->setSpeckleRange(32);//视差变化阈值  当窗口内视差变化大于阈值时，该窗口内的视差清零
    sgbm->setDisp12MaxDiff(-1);// 1
//左视图差（直接计算）和右视图差（cvValidateDisparity计算得出）之间的最大允许差异
    if(alg==STEREO_HH)               sgbm->setMode(StereoSGBM::MODE_HH);
    else if(alg==STEREO_SGBM)  sgbm->setMode(StereoSGBM::MODE_SGBM);
    else if(alg==STEREO_3WAY)   sgbm->setMode(StereoSGBM::MODE_SGBM_3WAY);
各参数设置如BM方法，速度比较快，320*240的灰度图匹配时间为78ms，

第三种为GC方法：
该方法速度超慢，但效果超好。


 */

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/utility.hpp"

#include <stdio.h>
#include <iostream> 

#include <boost/format.hpp>  // 格式化字符串 for formating strings 处理图像文件格式
#include <boost/thread/thread.hpp>

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
double cx = 319.8;//图像像素　原点平移
double cy = 242.6;
double fx = 484.8;//焦距和缩放  等效
double fy = 478.5;
double depthScale = 10000.0;// 深度 单位归一化到m 原数据单位 为 0.1mm
double base_line = 162.7;//mm为单位


static void print_help()
{
    printf("\nDemo stereo matching converting L and R images into disparity and point clouds\n");//生成视差和点云图
    printf("\nUsage: stereo_match  [--algorithm=bm|sgbm|hh|sgbm3way] [--blocksize=<block_size>]\n"
           "[--max-disparity=<max_disparity>] [--scale=<scale_factor>] [-i=<intrinsic_filename>] [-e=<extrinsic_filename>]\n"
           "[--no-display]  \n");
}


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
/*
void saveDisparity(const string file_name, const Mat& mat)
{
	ofstream fp(file_name.c_str(), ios::out);// 文件以输出方式打开（内存数据输出到文件）
	fp << mat.rows << endl;
	fp << mat.cols << endl;
	for(int y = 0; y < mat.rows; ++y){
	  for(int x = 0; x < mat.cols; ++x){
	    double disp = mat.at<short>(y, x);// cv_16s 格式 用short读取
	    fp << disp << endl;//若视差是CV_3SF格式则用float类型读取
	   }
	 }
	fp.close();
}
*/
/*
matlab 
filename = 'disparity.txt';
data = importdata(filename);
r = data(1);% clos  opencv hang youxian
c = data(2);
disp = data(3:end);
vmin = min(disp);
vmax = max(disp);
disp = reshape(disp, [c,r]);
img = unit8 ( 255 * (disp - vmin)/(vmax - vmin));
mesh(disp);
set(gca, 'Ydir', reverse);
axis tight;

*/

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


int initialize_sys(Ptr<StereoBM> bm, 
		   Ptr<StereoSGBM> sgbm, 
		   int argc, 
		   char** argv, 
		   int& alg,
 		   float& scale,
                   Mat& map11, 
		   Mat& map12, 
		   Mat& map21, 
	  	   Mat& map22,
		   int& numberOfDisparities,
		   Mat& Q
                  )
{

   //std::string img1_filename = "";//图像文件名
    //std::string img2_filename = "";
    std::string intrinsic_filename = "";//内参数文件名 给定
    std::string extrinsic_filename = "";//外参数文件名 给定
    std::string disparity_filename = "disparity.jpg";//视差文件名 生成
    
    //int alg = STEREO_SGBM;
    alg = STEREO_SGBM;
    //算法参数
    int SADWindowSize;
    bool no_display;
    //float scale;
    // bm = StereoBM::create(16,9);//局部的BM
    // sgbm = StereoSGBM::create(0,16,3);//全局的SGBM
    //Ptr<StereoBM> bm = StereoBM::create(16,9);//局部的BM
    //Ptr<StereoSGBM> sgbm = StereoSGBM::create(0,16,3);//全局的SGBM
    //参数解析
    cv::CommandLineParser parser(argc, argv,
        "{help h||}{algorithm|bm|}{max-disparity|64|}{blocksize|15|}{no-display||}{scale|1|}{i|intrinsics.yml|}{e|extrinsics.yml|}");
/*
max-disparity 是最大视差，可以理解为对比度，越大整个视差的range也就越大，这个要求是16的倍数
blocksize 一般设置为5-29之间的奇数，应该是局部算法的窗口大小
代价函数：
对应像素差的绝对值（SAD, Sum of Absolute Differences）   SAD(u,v) = Sum{|Left(u,v) - Right(u,v)|}  选择最小值
对应像素差的平方和（SSD, Sum of Squared Differences）
图像的相关性（NCC, Normalized Cross Correlation）

*/
//=======打印帮助信息
    if(parser.has("help"))
    {
        print_help();
        return 0;
    }
//=======确定算法================    
    if (parser.has("algorithm"))
    {
        std::string _alg = parser.get<std::string>("algorithm");
        alg = _alg == "bm" ? STEREO_BM :
            _alg == "sgbm" ? STEREO_SGBM :
            _alg == "hh" ? STEREO_HH :
            _alg == "var" ? STEREO_VAR :
            _alg == "sgbm3way" ? STEREO_3WAY : -1;
    }
// 
    numberOfDisparities = parser.get<int>("max-disparity");// 最大视差
    SADWindowSize = parser.get<int>("blocksize");// 局部算法的窗口大小
    scale = parser.get<float>("scale");
    no_display = parser.has("no-display");
    if( parser.has("i") )
        intrinsic_filename = parser.get<std::string>("i");//内参  内参数和畸变稀疏
    if( parser.has("e") )
        extrinsic_filename = parser.get<std::string>("e");//外参   R T
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
	
   cv::Mat img0;  
   //cv:: Mat img1r, img2r;//双目矫正图
   //cv::Mat disp, disp8;  //视差图
  //img1= src_img(cv::Range(0, 480), cv::Range(0, 640));   //imread(file,0) 灰度图  imread(file,1) 彩色图  默认彩色图
   img0 = imread("left01.jpg", 1);//图像数据
   Size img_size = img0.size();   //图像尺寸
  
   Rect roi1, roi2;//左右视图的有效像素区域 在有效视图之外的视差值被消零
  // Mat Q;//射影矩阵  
   //Mat map11, map12, map21, map22;// 矫正映射矩阵
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
		  fs["M1"] >> M1;// 内参数 K1  
		  fs["D1"] >> D1;// 畸变矫正
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
// 获取矫正映射矩阵
		  initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_32F, map11, map12);
		  initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_32F, map21, map22);


    } 
    
    numberOfDisparities = numberOfDisparities > 0 ? numberOfDisparities : ((img_size.width/8) + 15) & -16;
    // bm算法
    bm->setROI1(roi1);//左右视图的有效像素区域 在有效视图之外的视差值被消零
    bm->setROI2(roi2);
    bm->setPreFilterType(CV_STEREO_BM_XSOBEL);
    bm->setPreFilterSize(9);//滤波器尺寸 [5,255]奇数
    bm->setPreFilterCap(31);//预处理滤波器的截断值 [1-31] 
    bm->setBlockSize(SADWindowSize > 0 ? SADWindowSize : 15);//sad窗口大小
    bm->setMinDisparity(0);//最小视差值，代表了匹配搜索从哪里开始
    bm->setNumDisparities(numberOfDisparities);//表示最大搜索视差数
    bm->setTextureThreshold(10);//低纹理区域的判断阈值 x方向导数绝对值之和小于阈值
    bm->setUniquenessRatio(15);//视差唯一性百分比
    bm->setSpeckleWindowSize(100);//检查视差连通域 变化度的窗口大小
    bm->setSpeckleRange(32);//视差变化阈值  当窗口内视差变化大于阈值时，该窗口内的视差清零
    bm->setDisp12MaxDiff(-1);
//左视图差（直接计算）和右视图差（cvValidateDisparity计算得出）之间的最大允许差异

  // sgbm算法
    sgbm->setPreFilterCap(63);//预处理滤波器的截断值 [1-63] 
    int sgbmWinSize = SADWindowSize > 0 ? SADWindowSize : 3;
    sgbm->setBlockSize(sgbmWinSize);
    int cn = img0.channels();
    sgbm->setP1(8*cn*sgbmWinSize*sgbmWinSize);// 控制视差变化平滑性的参数。P1、P2的值越大，视差越平滑。
//P1是相邻像素点视差增/减 1 时的惩罚系数；P2是相邻像素点视差变化值大于1时的惩罚系数。P2必须大于P1
    sgbm->setP2(32*cn*sgbmWinSize*sgbmWinSize);
    sgbm->setMinDisparity(0);//最小视差值，代表了匹配搜索从哪里开始
    sgbm->setNumDisparities(numberOfDisparities);//表示最大搜索视差数
    sgbm->setUniquenessRatio(10);//表示匹配功能函数
    sgbm->setSpeckleWindowSize(100);//检查视差连通域 变化度的窗口大小
    sgbm->setSpeckleRange(32);//视差变化阈值  当窗口内视差变化大于阈值时，该窗口内的视差清零
    sgbm->setDisp12MaxDiff(-1);
//左视图差（直接计算）和右视图差（cvValidateDisparity计算得出）之间的最大允许差异
    if(alg==STEREO_HH)               sgbm->setMode(StereoSGBM::MODE_HH);
    else if(alg==STEREO_SGBM)  sgbm->setMode(StereoSGBM::MODE_SGBM);
    else if(alg==STEREO_3WAY)   sgbm->setMode(StereoSGBM::MODE_SGBM_3WAY);

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

    pointClouds *= 1.6;
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
 * 			| u|     |fx  0   ux 0|     |R      T|   | Yw|
 *       		| v| =   |0   fy  uy 0|  *  |        | * | Zw|=  P*W
 * 			|1 |     |0   0   1  0|    |0 0  0 1|    | 1 |
 * http://wiki.opencv.org.cn/index.php/Cv相机标定和三维重建
 *   像素坐标齐次表示(3*1)  =  内参数矩阵 齐次表示(3*4)  ×  外参数矩阵齐次表示(4*4) ×  物体世界坐标 齐次表示(4*1)
 *   内参数齐次 × 外参数齐次 整合 得到 投影矩阵  3*4    左右两个相机 投影矩阵 P1   P2
 *  
 *    世界坐标　W 　---->　左相机投影矩阵 P1 ------> 左相机像素点　(u1,v1,1)
　*                ----> 右相机投影矩阵 P２ ------> 右相机像素点　(u2,v2,1)
　*
 *
 *  Q为 视差转深度矩阵 disparity-to-depth mapping matrix 

　Z = f*B/d       =   f    /(d/B)
 X = Z*(x-c_x)/f = (x-c_x)/(d/B)
 X = Z*(y-c_y)/f = (y-y_x)/(d/B)

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
 * 
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
    //cout << "cloud size " <<  pointCloud.size() << endl;
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

//*/
/*
void detectDistance(cv::Mat& pointCloud)
{
    if (pointCloud.empty())
    {
        return;
    }

    // 提取深度图像
    vector<cv::Mat> xyzSet;
    split(pointCloud, xyzSet);
    cv::Mat depth;
    xyzSet[2].copyTo(depth);

    // 根据深度阈值进行二值化处理
    double maxVal = 0, minVal = 0;
    cv::Mat depthThresh = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
    cv::minMaxLoc(depth, &minVal, &maxVal);
    double thrVal = minVal * 1.5;
    threshold(depth, depthThresh, thrVal, 255, CV_THRESH_BINARY_INV);
    depthThresh.convertTo(depthThresh, CV_8UC1);
    //imageDenoising(depthThresh, 3);

   // double  distance = depth.at<float>(pic_info[0], pic_info[1]);
    cout << "distance:" << distance << endl;
}
*/

////// 主函数//////

int main(int argc, char** argv)
{


    Ptr<StereoBM> bm = StereoBM::create(16,9);//局部的BM;
    Ptr<StereoSGBM> sgbm = StereoSGBM::create(0,16,3);//全局的SGBM;
    int alg=0;
    float scale;
    Mat map11, map12, map21, map22;// 矫正映射矩阵
    Mat Q;//射影矩阵 
    int numberOfDisparities;
    initialize_sys(bm, sgbm, argc, argv, alg, scale, map11, map12, map21, map22, numberOfDisparities, Q);

    cv::Mat img1,img2,src_img1;  
    cv::Mat src_img; 
  
    cv::Mat img1r, img2r;//双目矫正图
    cv::Mat disp, disp8, disp32;  //视差图

    char depth_name[] ="depth.txt";
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

    // 新建一个点云 对象
    //PointCloud::Ptr pointCloud_PCL( new PointCloud ); 
    //boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_ptr (new pcl::visualization::PCLVisualizer ("3D Viewer"));  
    //设置一个boost共享对象，并分配内存空间
    //viewer_ptr->setBackgroundColor(0.0, 0.0, 0.0);//背景黑色
    pcl::visualization::CloudViewer viewer("pcd　viewer");// 显示窗口的名字
  //viewer.showCloud(cloud_filtered_ptr);

    while(CapAll.read(src_img)) 
	{  
	     img1= src_img(cv::Range(0, 480), cv::Range(0, 640));   //imread(file,0) 灰度图  imread(file,1) 彩色图  默认彩色图
// BM和GC算法只能对8位灰度图像计算视差，SGBM算法则可以处理24位（8bits*3）彩色图像。
// 所以在读入图像时，应该根据采用的算法来处理图像：
	     img2 = src_img(cv::Range(0, 480), cv::Range(640, 1280)); 
             src_img1 = img1.clone();
	     if(alg == STEREO_BM){
		    cvtColor(img1, img1, COLOR_BGR2GRAY);//转换到 灰度图
		    cvtColor(img2, img2, COLOR_BGR2GRAY);//转换到 灰度图	   
	     }
	     if (img1.empty()||img2.empty()){
		  printf("获取不到图像\n\n");
		  return -1;
	     }
///*	     
	     if (scale != 1.f){
		  Mat temp1, temp2;
		  int method = scale < 1 ? INTER_AREA : INTER_CUBIC;
		  resize(img1, temp1, Size(), scale, scale, method);
		  img1 = temp1;
		  resize(img2, temp2, Size(), scale, scale, method);
		  img2 = temp2;
	     }  
//*/    
	    //图像矫正 矫正原始图像
	    remap(img1, img1r, map11, map12, INTER_LINEAR);
	    remap(img2, img2r, map21, map22, INTER_LINEAR);
	    img1 = img1r;
	    img2 = img2r;
	      
	     Mat img1p, img2p, dispp, disparityImage; //防止黑边  加宽
// 对左右视图的左边进行边界延拓，以获取与原始视图相同大小的有效视差区域
	     copyMakeBorder(img1, img1p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);
	     copyMakeBorder(img2, img2p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);
	    //  int64 t = getTickCount();
	      if( alg == STEREO_BM )
		  bm->compute(img1p, img2p, dispp);
	      else if( alg == STEREO_SGBM || alg == STEREO_HH || alg == STEREO_3WAY )
		  sgbm->compute(img1p, img2p, dispp);
             // 截取与原始画面对应的视差区域（舍去加宽的部分）
             disp = dispp.colRange(numberOfDisparities, img1p.cols);
	     // t = getTickCount() - t;
	     // printf("Time elapsed: %fms\n", t*1000/getTickFrequency());
	      //disp = dispp.colRange(numberOfDisparities, img1p.cols);
	      // int16  ---> uint8 
	      if( alg != STEREO_VAR )
		  disp.convertTo(disp8, CV_8U, 255/(numberOfDisparities*16.));
	      else
		  disp.convertTo(disp8, CV_8U);
	      // 视差图有伪彩色显示
              getDisparityImage(disp8, disparityImage ,true);

              // 计算出的视差都是CV_16S格式的，使用32位float格式可以得到真实的视差值，所以我们需要除以16
              disp.convertTo(disp32, CV_32F, 1.0/16); 

             // Mat vdispRGB = disp8;
              //F_Gray2Color(disp8, vdispRGB);
              //Mat merge = F_mergeImg(img1, disp8);
	//      if( !no_display )
	 //     {
		  namedWindow("左相机", 1);
		  imshow("左相机", src_img1);
		  //namedWindow("右相机", 1);
		  //imshow("右相机", img2);
		  //namedWindow("视差8", 0);// CV_WINDOW_NORMAL
		  //imshow("视差8", disp8);
		 // namedWindow("视差32", 0);// CV_WINDOW_NORMAL
		  //imshow("视差32", disp32);
                  PointCloud::Ptr pointCloud_PCL2( new PointCloud ); 
		  PointCloud& pointCloud1 = *pointCloud_PCL2;
                  getpc(disp32, Q, src_img1, pointCloud1);
                 //my_getpc(disp32, Q, src_img1, pointCloud1);
                 // cout << "cloud size " <<  pointCloud1.width * pointCloud1.height<<endl;
  		  viewer.showCloud(pointCloud_PCL2);

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
		   // {
			//viewer.spinOnce ();
			//pcl_sleep(0.03);
		    //}
		  fflush(stdout);
		  char c = waitKey(0);
		  printf("\n");
	//      } 
	      if( c == 27 || c == 'q' || c == 'Q' )//按ESC/q/Q退出  
		break; 
     }

    return 0;
}
