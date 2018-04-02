/*
SURF快速鲁棒 关键点检测器  描述子  L2匹配  flann匹配

作为尺度不变特征变换算法（Sift算法）的加速版，
Surf算法在适中的条件下完成两幅图像中物体的匹配基本实现了实时处理，
其快速的基础实际上只有一个——积分图像haar求导。我们先来看介绍Sift算法的基本过程，然后再介绍Surf算法。

Sift算法简介
http://www.cnblogs.com/ronny/p/4028776.html
	Sift特征匹配算法可以处理两幅图像之间发生平移、旋转、仿射变换情况下的匹配问题，具有很强的匹配能力。

	总体来说，Sift算子具有以下特性：
		(1) 、Sift特征是图像的局部特征，对平移、旋转、尺度缩放、亮度变化、
			遮挡和噪声等具有良好的不变性，对视觉变化、仿射变换也保持一定程度的稳定性。
		(2) 、独特性好，信息量丰富，适用于在海量特征数据库中进行快速、准确的匹配。
		(3) 、多量性，即使少数的几个物体也可以产生大量Sift特征向量。
		(4) 、速度相对较快，经优化的Sift匹配算法甚至可以达到实时的要求。
		(5) 、可扩展性强，可以很方便的与其他形式的特征向量进行联合。 

	 Sift算法的三大工序为，
		(1)提取关键点；
		(2)对关键点附加详细的信息（局部特征）也就是所谓的描述器；
		(3)通过两方特征点（附带上特征向量的关键点）的两两比较找出相互匹配的若干对特征点，
			也就建立了景物间的对应关系。
	 提取关键点和 对关键点附加详细的信息（局部特征）也就是所谓的描述器 可以称做是Sift特征的生成，
	即从多幅图像中提取对尺度缩放、旋转、亮度变化无关的特征向量，
	Sift特征的生成一般包括以下几个步骤： 
	 (1) 、 构建高斯金字塔尺度空间，计算高斯差分金字塔，检测极值点，获得尺度不变性； 
	 (2) 、 特征点过滤并进行精确定位； 
	 (3) 、 为特征点分配方向值； 
	 (4) 、 生成特征描述子。 
		以特征点为中心取16*16的邻域作为采样窗口，将采样点与特征点的相对方向通过高斯加权后
		归入包含8个bin的方向直方图，最后获得4*4*8的128维特征描述子。

	当两幅图像的Sift特征向量生成以后，
		下一步就可以采用 关键点特征向量的 欧式距离 来作为 两幅图像中关键点的相似性判定度量。
		取图1的某个关键点，通过遍历找到图像2中的距离最近的两个关键点。
		在这两个关键点中，如果次近距离除以最近距离小于某个阙值，则判定为一对匹配点。


Surf算法原理：
http://www.cnblogs.com/ronny/p/4045979.html
 (1)、构建Hessian矩阵
	Hessian矩阵是Surf算法的核心，为了方便运算，假设函数f(z，y)，
	Hessian矩阵H是由函数，偏导数组成：

	H(f(x,y)) = [f2/x^2    f2/(x*y)  二阶偏导数
                     f2/(x*y)  f2/y^2 ]

       H矩阵判别式为： f2/x^2 * f2/y^2  - f2/(x*y) * f2/(x*y)  矩阵行列式的值
       判别式的值是H矩阵的特征值，可以利用判定结果的符号将所有点分类，
       根据判别式取值正负，来判别该点是或不是极值点。

       在SURF算法中，用图像像素l(x，y)代替函数值f(x，y)，
       选用二阶标准高斯函数作为滤波器，通过特定核间的卷积计算二阶偏导数，
       这样便能计算出H矩阵的三个矩阵元素Lxx Lxy Lyy，从而计算出H矩阵：

       det(H) = Lxx * Lyy  - (0.9*Lxy)^2 

 (2)、构建尺度空间
	图像的尺度空间是这幅图像在不同(模糊度 下采样率)（解析度）下的表示
	在计算视觉领域，尺度空间被象征性的表述为一个图像金字塔，其中，
	输入图像函数反复与高斯函数的核卷积并反复对其进行二次抽样，
	这种方法主要用于Sift算法的实现，但每层图像依赖于前一层图像，
	并且图像需要重设尺寸，因此，这种计算方法运算量较大，而SURF算法申请增加图像核的尺寸，
	这也是SIFT算法与SURF算法在使用金字塔原理方面的不同。

 (3)、精确定位特征点
	所有小于预设极值的取值都被丢弃，增加极值使检测到的特征点数量减少，
	最终只有几个特征最强点会被检测出来。检测过程中使用与该尺度层图像解析度相对应大小的滤波器进行检测，
	以3×3的滤波器为例，该尺度层图像中9个像素点之一图2检测特征点与自身尺度层中
	其余8个点和在其之上及之下的两个尺度层9个点进行比较，共26个点，
	图2中标记‘x’的像素点的特征值若大于周围像素则可确定该点为该区域的特征点。


 (4)、主方向确定
	为保证旋转不变性[8I，首先以特征点为中心，计算半径为6s(S为特征点所在的尺度值)的邻域内的点在z、y
	方向的Haar小波(Haar小波边长取4s)响应，并给这些响应值赋高斯权重系数，使得靠近特征点的响应贡献大，
	而远离特征点的响应贡献小，其次将60。范围内的响应相加以形成新的矢量，遍历整个圆形区域，
	选择最长矢量的方向为该特征点的主方向。这样，通过特征点逐个进行计算，得到每一个特征点的主方向。

 (5)特征点描述子生成
	首先将坐标轴旋转为关键点的方向，以确保旋转不变性。
	以关键点为中心取8×8的窗口。图左部分的中央黑点为当前关键点的位置，
	每个小格代表关键点邻域所在尺度空间的一个像素，利用公式求得每个像素的梯度幅值与梯度方向，
	箭头方向代表该像素的梯度方向，箭头长度代表梯度模值，然后用高斯窗口对其进行加权运算,每个像素对应一个向量，
	长度为，为该像素点的高斯权值，方向为， 图中蓝色的圈代表高斯加权的范围（越靠近关键点的像素梯度方向信息贡献越大）。
	然后在每4×4的小块上计算8个方向的梯度方向直方图，绘制每个梯度方向的累加值，即可形成一个种子点，如图右部分示



// SURF放在另外一个包的xfeatures2d里边了，在github.com/Itseez/opencv_contrib 这个仓库里。
// 按说明把这个仓库编译进3.0.0就可以用了。
opencv2中SurfFeatureDetector、SurfDescriptorExtractor、BruteForceMatcher在opencv3中发生了改变。
具体如何完成特征点匹配呢？示例如下：

//寻找关键点
int minHessian = 700;
Ptr<SURF>detector = SURF::create(minHessian);
detector->detect( srcImage1, keyPoint1 );
detector->detect( srcImage2, keyPoints2 );

//绘制特征关键点
Mat img_keypoints_1; Mat img_keypoints_2;
drawKeypoints( srcImage1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
drawKeypoints( srcImage2, keypoints_2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

//显示效果图
imshow("特征点检测效果图1", img_keypoints_1 );
imshow("特征点检测效果图2", img_keypoints_2 );

//计算特征向量
Ptr<SURF>extractor = SURF::create();
Mat descriptors1, descriptors2;
extractor->compute( srcImage1, keyPoint1, descriptors1 );
extractor->compute( srcImage2, keyPoints2, descriptors2 );

//使用BruteForce进行匹配
Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
std::vector< DMatch > matches;
matcher->match( descriptors1, descriptors2, matches );

//绘制从两个图像中匹配出的关键点
Mat imgMatches;
drawMatches( srcImage1, keyPoint1, srcImage2, keyPoints2, matches, imgMatches );//进行绘制
//显示
imshow("匹配图", imgMatches );


3.x的特征检测:

    算法：SURF,SIFT,BRIEF,FREAK 
    类：cv::xfeatures2d::SURF

    cv::xfeatures2d::SIFT
    cv::xfeatures::BriefDescriptorExtractor
    cv::xfeatures2d::FREAK
    cv::xfeatures2d::StarDetector

    需要进行以下几步

    加入opencv_contrib
    包含opencv2/xfeatures2d.hpp
    using namepsace cv::xfeatures2d
    使用create(),detect(),compute(),detectAndCompute()

*/

#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
//#include "opencv2/xfeatures2d.hpp" // opencv_contrib 内
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <string>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;
void readme();

//主函数
int main( int argc, char** argv )
{

 string imageName("../../common/data/building.jpg"); // 图片文件名路径（默认值）
    if( argc > 1)
    {
        imageName = argv[1];//如果传递了文件 就更新
    }

  Mat src = imread( imageName );
  if( src.empty() )
    {   
        cout <<  "can't load image " << endl;
	readme();
	return -1;  
    }
  // 转换成灰度图
  Mat src_gray;
  cvtColor( src, src_gray, CV_BGR2GRAY );

//=======【1】检测关键点 使用SURF检测器
  int minHessian = 400;//点数量
  //SurfFeatureDetector detector( minHessian );// SURF 检测器  老版本
  Ptr<SURF> detector = SURF::create(minHessian);// 3.x版本 需要安装 opencv_contrib 
  std::vector<KeyPoint> keypoints_1;         // 关键点

  // detector.detect( img_1, keypoints_1 );
  detector->detect( img_1, keypoints_1 );//检测关键点

// SurfDescriptorExtractor extractor;//老版本的描述子提取器
  Mat descriptors_1;//描述子 
// extractor.compute( img_1, keypoints_1, descriptors_1 );
 // detector->compute( img_1, Mat(), keypoints_1, descriptors_1 );// 检测
 detector->detectAndCompute( img_1, Mat(), keypoints_1, descriptors_1 );//检测关键点 并计算描述子

//=======【2】将关键点 画在图像上
  Mat img_keypoints_1;;
  drawKeypoints( src, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
//=====【3】显示===============
  imshow("Keypoints 1", img_keypoints_1 );

// 关键点描述子匹配    新版
  BFMatcher matcher(NORM_L2);// 匹配代价 l2距离 差平方和
   // 老版本
   //   BruteForceMatcher< L2<float> > matcher;  
  std::vector< DMatch > matches;// 1匹配到2中的 点
  matcher.match( descriptors_1, descriptors_2, matches );

//-- 显示匹配点对
  Mat img_matches;
  drawMatches( img_1, keypoints_1, img_2, keypoints_2, matches, img_matches );
  //-- Show detected matches
  imshow("Matches", img_matches );


//================快速最近邻算法求 匹配点对===================
// cv::FlannBasedMatcher
  FlannBasedMatcher matcher;
  std::vector< DMatch > matches;
  matcher.match( descriptors_1, descriptors_2, matches );
  // 找到最小的最大距离 
  double max_dist = 0; double min_dist = 100;//最大最小距离
  //-- Quick calculation of max and min distances between keypoints
  for( int i = 0; i < descriptors_1.rows; i++ )
  { double dist = matches[i].distance;//1匹配上2中点对 距离
    if( dist < min_dist ) min_dist = dist;//最小距离
    if( dist > max_dist ) max_dist = dist;//最大距离
  }
  printf("-- Max dist : %f \n", max_dist );
  printf("-- Min dist : %f \n", min_dist );

// 筛选 最优的匹配点对  距离小于 max(2*最小距离 ， 0.02) 为好的匹配点对
// 其次 可以在考虑 相互匹配 的 才为 好的匹配点
  std::vector< DMatch > good_matches;
  for( int i = 0; i < descriptors_1.rows; i++ )
  { if( matches[i].distance <= max(2*min_dist, 0.02) )
    { good_matches.push_back( matches[i]); }
  }

//=====显示匹配点
  Mat img_matches;
  drawMatches( img_1, keypoints_1, img_2, keypoints_2,
               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
  //-- Show detected matches
  imshow( "Good Matches", img_matches );


  waitKey(0);

  return 0;
  }

// 用法
  void readme()
  { std::cout << " Usage: ./SURF_detector <img1> " << std::endl; }
