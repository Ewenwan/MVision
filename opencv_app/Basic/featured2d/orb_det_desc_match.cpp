/*
使用 orb 特征检测 匹配 
使用二维特征点(orb  Features2D )和单映射(Homography)寻找已知物体

【1】 创建新的控制台(console)项目。读入两个输入图像。
【2】 检测两个图像的关键点（尺度旋转都不发生变化的关键点）
【3】 计算每个关键点的描述向量(Descriptor)
【4】 计算两幅图像中的关键点对应的描述向量距离，寻找两图像中距离最近的描述向量对应的关键点，即为两图像中匹配上的关键点:
【5】 寻找两个点集合中的单映射变换（homography transformation）:
【6】 创建内匹配点集合同时绘制出匹配上的点。用perspectiveTransform函数来通过单映射来映射点:
【7】 用 drawMatches 来绘制内匹配点.

*/
#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"

using namespace cv;
using namespace std;
void readme();

// 主函数
int main( int argc, char** argv )
{
  /// 加载源图像
    string imageName1("../../common/data/box.png"); // 图片文件名路径（默认值）
    string imageName2("../../common/data/box_in_scene.png"); // 图片文件名路径（默认值）
    if( argc > 2)
    {
        imageName1 = argv[1];//如果传递了文件 就更新
	imageName2 = argv[2];//如果传递了文件 就更新
    }
    Mat img_object = imread( imageName1, CV_LOAD_IMAGE_GRAYSCALE );
    Mat img_scene  = imread( imageName2, CV_LOAD_IMAGE_GRAYSCALE );
    if( img_object.empty() || img_scene.empty() )
    { 
        cout <<  "can't load image " << endl;
        readme();
	return -1;  
    }

//======【1】检测关键点  ==================
  std::vector<KeyPoint> keypoints_object, keypoints_scene;

  //int minHessian = 400; // SURF关键点
  //SurfFeatureDetector detector( minHessian );
  //detector.detect( img_object, keypoints_object );
  //detector.detect( img_scene, keypoints_scene );

  Ptr<FeatureDetector> orb = ORB::create();//orb 检测器
  orb->detect(img_object, keypoints_object);
  orb->detect(img_scene,  keypoints_scene);

  cout<< "keypoints_object size() " << keypoints_object.size() << endl;
  cout<< "keypoints_scene size() "  << keypoints_scene.size() << endl;

//======【2】计算描述子========
  Mat descriptors_object, descriptors_scene;

  //SurfDescriptorExtractor extractor;
  //extractor.compute( img_object, keypoints_object, descriptors_object );
  //extractor.compute( img_scene, keypoints_scene, descriptors_scene );

  orb->compute(img_object, keypoints_object, descriptors_object);
  orb->compute(img_scene,  keypoints_scene,  descriptors_scene);

///*
//=====【3】对描述子进行匹配 使用FLANN 匹配
// 鲁棒匹配器设置　描述子匹配器
  Ptr<cv::flann::IndexParams> indexParams = makePtr<flann::LshIndexParams>(6, 12, 1); //  LSH index parameters
  Ptr<cv::flann::SearchParams> searchParams = makePtr<flann::SearchParams>(50);       //   flann search parameters
  // instantiate FlannBased matcher
  Ptr<DescriptorMatcher> matcher = makePtr<FlannBasedMatcher>(indexParams, searchParams);

  //FlannBasedMatcher matcher;
  //std::vector< DMatch > matches;
  //matcher.match( descriptors_object, descriptors_scene, matches );

 std::vector<std::vector<cv::DMatch> > matches12, matches21;// 最匹配　和　次匹配
 //std::vector<cv::DMatch> matches12;
 matcher->knnMatch(descriptors_object, descriptors_scene, matches12, 1); // 1->2
 // matcher->knnMatch(descriptors_scene, descriptors_object, matches21, 1);
 cout<< "match12 size() " << matches12.size() << endl;

///*
   double max_dist = 0; double min_dist = 100;
  //  找到最小的最大距离
  for( int i = 0; i < descriptors_object.rows; i++ )
  { 
    double dist = matches12[i][0].distance;//1匹配上2中点对 距离
    if( dist < min_dist ) min_dist = dist;//最小距离
    if( dist > max_dist ) max_dist = dist;//最大距离
  }
  printf("-- Max dist : %f \n", max_dist );
  printf("-- Min dist : %f \n", min_dist );
///*
//====筛选 最优的匹配点对  距离小于 2.3 *最小距离 为好的匹配点对============
// 其次 可以在考虑 相互匹配 的 才为 好的匹配点
  std::vector< DMatch > good_matches;
  for( int i = 0; i < descriptors_object.rows; i++ )
  { if( matches12[i][0].distance < 2.3 * min_dist )
     { good_matches.push_back( matches12[i][0]); }
  }

 cout<< "good_matches size() " << good_matches.size() << endl;
///*
//=====显示匹配点======================
  Mat img_matches;
  drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

  //-- 得到2d-2d匹配点坐标
  std::vector<Point2f> obj;
  std::vector<Point2f> scene;
  for( int i = 0; i < good_matches.size(); i++ )
  {
    //-- Get the keypoints from the good matches
    obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
    scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
  }

  // 求解单映射矩阵 H  p1 = H * p2   平面变换
  Mat H = findHomography( obj, scene, CV_RANSAC );

  //-- 得到需要检测的物体的四个顶点  
  std::vector<Point2f> obj_corners(4);
  obj_corners[0] = cvPoint(0,0); 
  obj_corners[1] = cvPoint( img_object.cols, 0 );
  obj_corners[2] = cvPoint( img_object.cols, img_object.rows ); 
  obj_corners[3] = cvPoint( 0, img_object.rows );

  std::vector<Point2f> scene_corners(4);// 场景中的 该物体的顶点

  // 投影过去
  perspectiveTransform( obj_corners, scene_corners, H);

  //--在场景中显示检测到的物体 (the mapped object in the scene - image_2 )
  line( img_matches, scene_corners[0] + Point2f( img_object.cols, 0), scene_corners[1] + Point2f( img_object.cols, 0), Scalar(0, 255, 0), 4 );
  line( img_matches, scene_corners[1] + Point2f( img_object.cols, 0), scene_corners[2] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
  line( img_matches, scene_corners[2] + Point2f( img_object.cols, 0), scene_corners[3] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
  line( img_matches, scene_corners[3] + Point2f( img_object.cols, 0), scene_corners[0] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );

  //-- 显示检测结果
  imshow( "Good Matches & Object detection", img_matches );

  waitKey(0);
  return 0;
}

// 用法
  void readme()
  { std::cout << " Usage: ./SURF_descriptor <img1> <img2>" << std::endl; }
