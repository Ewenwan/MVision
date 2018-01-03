/*
 *三角测量法 求解 两组单目相机  图像点深度
 * s1 * x1 = s2  * R * x2 + t
 * x1 x2 为两帧图像上 两点对 在归一化坐标平面上的坐标
 * s1  和 s2为两个特征点的深度 ，由于误差存在， s1 * x1 = s2  * R * x2 + t不精确相等
 * 常见的是求解最小二乘解，而不是零解
 *  s1 * x1叉乘x1 = s2 * x1叉乘* R * x2 + x1叉乘 t=0 可以求得x2
 * 
 */
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
// #include "extra.h" // used in opencv2 
using namespace std;//标准库　命名空间
using namespace cv; //opencv库命名空间
// ./pose_estimation_2d2d 1.png 2.png
/****************************************************
 * 本程序演示了如何使用2D-2D的特征匹配估计相机运动   基于三角测量
 * **************************************************/

//特征匹配 计算匹配点对
void find_feature_matches (
    const Mat& img_1, const Mat& img_2,
    std::vector<KeyPoint>& keypoints_1,
    std::vector<KeyPoint>& keypoints_2,
    std::vector< DMatch >& matches );

//位置估计 计算旋转和平移  第一张图 到第二章图的坐标变换矩阵和平移矩阵
void pose_estimation_2d2d (
    const std::vector<KeyPoint>& keypoints_1,
    const std::vector<KeyPoint>& keypoints_2,
    const std::vector< DMatch >& matches,
    Mat& R, Mat& t );

// 三角测量
void triangulation (
    const vector<KeyPoint>& keypoint_1,
    const vector<KeyPoint>& keypoint_2,
    const std::vector< DMatch >& matches,
    const Mat& R, const Mat& t,
    vector<Point3d>& points
);

// 像素坐标转相机归一化坐标
Point2f pixel2cam( const Point2d& p, const Mat& K );

int main ( int argc, char** argv )
{
    if ( argc != 3 )//命令行参数　 1.png 　2.png
    {
        cout<<"用法: ./triangulation img1 img2"<<endl;
        return 1;
    }
    //-- 读取图像
    Mat img_1 = imread ( argv[1], CV_LOAD_IMAGE_COLOR );//彩色图模式
    Mat img_2 = imread ( argv[2], CV_LOAD_IMAGE_COLOR );

    //求取特征匹配点对
    vector<KeyPoint> keypoints_1, keypoints_2;//关键点初始化
    vector<DMatch> matches;//特征点匹配对初始化
    find_feature_matches ( img_1, img_2, keypoints_1, keypoints_2, matches );
    cout<<"一共找到了"<<matches.size() <<"组匹配点"<<endl;

    //-- 估计两张图像间运动 R   t
    Mat R,t;//旋转和平移  第一张图 到第二章图的坐标变换矩阵和平移矩阵
    pose_estimation_2d2d ( keypoints_1, keypoints_2, matches, R, t );

    //-- 三角化  由运动变换矩阵 R t 以及三角测量法 得到 特征点的深度
    // s1 * x1 = s2  * R * x2 + t
    // s1 * x1叉乘x1 = s2 * x1叉乘* R * x2 + x1叉乘 t=0 可以求得x2
    vector<Point3d> points;
    triangulation( keypoints_1, keypoints_2, matches, R, t, points );
    
    //-- 验证三角化点与特征点的重投影关系
    Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );// 相机内参,TUM Freiburg2
    for ( int i=0; i<matches.size(); i++ )
    {
        Point2d pt1_cam = pixel2cam( keypoints_1[ matches[i].queryIdx ].pt, K );
        Point2d pt1_cam_3d(points[i].x/points[i].z, points[i].y/points[i].z ); //归一化
        cout<<"第一帧图像中的特征点:  "<<pt1_cam<<endl;
        cout<<"从三角计算得到的3D点重投影得到的二维点： "<<pt1_cam_3d<<", d="<<points[i].z<<endl;
        
        // 第二个图
        Point2f pt2_cam = pixel2cam( keypoints_2[ matches[i].trainIdx ].pt, K );
        Mat pt2_trans = R*( Mat_<double>(3,1) << points[i].x, points[i].y, points[i].z ) + t;
        pt2_trans /= pt2_trans.at<double>(2,0);//归一化
        cout<<"第二帧图像中的特征点: "<<pt2_cam<<endl;
        cout<<"从三角计算得到的3D点重投影得到的二维点： "<<pt2_trans.t()<<endl;
        cout<<endl;
    }
    
    return 0;
}

//特征匹配 计算匹配点对 函数
void find_feature_matches ( const Mat& img_1, const Mat& img_2,
                            std::vector<KeyPoint>& keypoints_1,
                            std::vector<KeyPoint>& keypoints_2,
                            std::vector< DMatch >& matches )
{
    //--------------------第0步:初始化------------------------------------------------------
    Mat descriptors_1, descriptors_2;//描述子
    //  OpenCV3 特征点检测器  描述子生成器 用法
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    // OpenCV2 特征点检测器  描述子生成器 用法
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create("BruteForce-Hamming");;//汉明点对匹配
    
    //------------------第一步:检测 Oriented FAST 角点位置-----------------------------
    detector->detect ( img_1,keypoints_1 );
    detector->detect ( img_2,keypoints_2 );

   //------------------第二步:根据角点位置计算 BRIEF 描述子-------------------------
    descriptor->compute ( img_1, keypoints_1, descriptors_1 );
    descriptor->compute ( img_2, keypoints_2, descriptors_2 );

    //------------------第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> match;
   // BFMatcher matcher ( NORM_HAMMING );
    matcher->match ( descriptors_1, descriptors_2, match );

    //-----------------第四步:匹配点对筛选--------------------------------------------------
    double min_dist=10000, max_dist=0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = match[i].distance;
        if ( dist < min_dist ) min_dist = dist;   //最短距离  最相似
        if ( dist > max_dist ) max_dist = dist;  //最长距离 最不相似
    }

    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if ( match[i].distance <= max ( 2*min_dist, 30.0 ) )
        {
            matches.push_back ( match[i] );
        }
    }
}

//特征匹配 计算匹配点对 函数   第一张图 到第二章图的坐标变换矩阵和平移矩阵
//对极几何
void pose_estimation_2d2d (
    const std::vector<KeyPoint>& keypoints_1,
    const std::vector<KeyPoint>& keypoints_2,
    const std::vector< DMatch >& matches,
    Mat& R, Mat& t )
{
    // 相机内参,TUM Freiburg2
    Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );

    //-- 把匹配点转换为vector<Point2f>的形式
    vector<Point2f> points1;
    vector<Point2f> points2;

    for ( int i = 0; i < ( int ) matches.size(); i++ )
    {
        points1.push_back ( keypoints_1[matches[i].queryIdx].pt );
        points2.push_back ( keypoints_2[matches[i].trainIdx].pt );
    }

    //-- 计算基础矩阵   F
    Mat fundamental_matrix;
    fundamental_matrix = findFundamentalMat ( points1, points2, CV_FM_8POINT );
    cout<<"fundamental_matrix is "<<endl<< fundamental_matrix<<endl;

    //-- 计算本质矩阵   E
    Point2d principal_point ( 325.1, 249.7 );				//相机主点, TUM dataset标定值
    int focal_length = 521;						//相机焦距, TUM dataset标定值
    Mat essential_matrix;
    essential_matrix = findEssentialMat ( points1, points2, focal_length, principal_point );
    cout<<"essential_matrix is "<<endl<< essential_matrix<<endl;

    //-- 计算单应矩阵  H
    Mat homography_matrix;
    homography_matrix = findHomography ( points1, points2, RANSAC, 3 );
    cout<<"homography_matrix is "<<endl<<homography_matrix<<endl;

    //-- 从本质矩阵中恢复旋转和平移信息.
    recoverPose ( essential_matrix, points1, points2, R, t, focal_length, principal_point );
    cout<<"R is "<<endl<<R<<endl;
    cout<<"t is "<<endl<<t<<endl;
}

//三角测量
// s1 * x2 = s2 * x1 * R  + t
// s1  和 s2为深度 误差存在，x1 和 x2 可能不同
void triangulation ( 
    const vector< KeyPoint >& keypoint_1, 
    const vector< KeyPoint >& keypoint_2, 
    const std::vector< DMatch >& matches,
    const Mat& R, const Mat& t, //特征匹配 计算匹配点对 函数   第一张图 到第二章图的坐标变换矩阵和平移矩阵
//对极几何
    vector< Point3d >& points )
{
  // 变换矩阵 在第一幅图像下 的变换矩阵  Pc1  =   Pw  =  T1 * Pw
    Mat T1 = (Mat_<float> (3,4) <<
        1,0,0,0,
        0,1,0,0,
        0,0,1,0);
   // Pc2  =  T2 * Pw  =  [ R  t] * Pw
    Mat T2 = (Mat_<float> (3,4) <<
        R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), t.at<double>(0,0),
        R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), t.at<double>(1,0),
        R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), t.at<double>(2,0)
    );
    
  // 相机内参,TUM Freiburg2   
    //   [fx 0 cx
    //     0 fy cy
    //     0 0  1]
    Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
    vector<Point2f> pts_1, pts_2;//相机坐标下点坐标
    for ( DMatch m:matches )
    {
        // 将像素坐标转换至相机坐标
        pts_1.push_back ( pixel2cam( keypoint_1[m.queryIdx].pt, K) );
        pts_2.push_back ( pixel2cam( keypoint_2[m.trainIdx].pt, K) );
    }
    
    Mat pts_4d;//三角测量得到的 三维空间点 其次坐标
    cv::triangulatePoints( T1, T2, pts_1, pts_2, pts_4d );
    
    // 转换成非齐次坐标
    for ( int i=0; i<pts_4d.cols; i++ )
    {
        Mat x = pts_4d.col(i);
        x /= x.at<float>(3,0); // 归一化
        Point3d p (
            x.at<float>(0,0), 
            x.at<float>(1,0), 
            x.at<float>(2,0) 
        );
        points.push_back( p );
    }
}

// 像素坐标转相机归一化坐标
Point2f pixel2cam ( const Point2d& p, const Mat& K )
{
    return Point2f
    (
        ( p.x - K.at<double>(0,2) ) / K.at<double>(0,0), 
        ( p.y - K.at<double>(1,2) ) / K.at<double>(1,1) 
    );
}

