/*
 *对极几何 求解 两组单目相机 2D图像
 * 2D 点对 求 两相机的 旋转和平移矩阵
 * 空间点 P  两相机 像素点对  p1  p2 两相机 归一化平面上的点对 x1 x2 与P点对应
 * 相机内参数 K  两镜头旋转平移矩阵  R t 或者 变换矩阵 T
 *  p1 = KP  (世界坐标系)     p2 = K( RP + t)  = KTP
 *  而 x1 =  K逆* p1  x2 =  K逆* p2  相机坐标系下 归一化平面上的点     x1= (px -cx)/fx    x2= (py -cy)/fy
 * 所以   x1 = P     x2 =  R  x1  + t
 *  t 外积  x2  = t 外积 R  x1       ；t 外积 t   =0    sin(cet) =0 垂线段投影
 *  x2转置 *  t 外积  x2 = x2转置 * t 外积 R  x1   = 0
 *   有 x2转置 * t 外积 R  x1   = x2转置 * E * x1 =  0 ； E 为本质矩阵
 * p2转置 * K 转置逆 * t 外积 R * K逆 * p1   =p2转置 * F * p1 =  0 ；
 * F 为基础矩阵
 * 
 * x2转置 * E * x1 =  0    x1 x2  为 由 像素坐标转化的归一化坐标
 * 一个点对一个约束 ，8点法  可以算出 E的各个元素 ，
 * 再 奇异值分解 E 得到 R t
 * 
 * 
 * 单应矩阵描述了两个平面间的映射关系
 * p2 = K( RP + t)       有 P在极平面上  平面方程  n转置 * P + d = 0 
 * 得到 -  n转置 * P/d  =1 
 * p2 = K( RP - t *n转置 * P/d)   = K( R - t *n转置 * /d)*P = H *p1
 * 
 * p2 = H *p1
 * 一个点对 2个约束
 * 4 点法求解  单应矩阵 H 再对 H进行分解
 * 
 *
 * / 
*/ 
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
// #include "extra.h" // use this if in OpenCV2 
using namespace std;//标准库　命名空间
using namespace cv; //opencv库命名空间

// ./pose_estimation_2d2d 1.png 2.png
/****************************************************
 * 本程序演示了如何使用2D-2D的特征匹配估计相机运动  
 * **************************************************/
//特征匹配 计算匹配点对
void find_feature_matches (
    const Mat& img_1, const Mat& img_2, // & 为引用  直接使用 参数本身 不进行复制  节省时间
    std::vector<KeyPoint>& keypoints_1,// 
    std::vector<KeyPoint>& keypoints_2,
    std::vector< DMatch >& matches );//keypoint Descriptors Match   描述子匹配
//位置估计 计算旋转和平移
void pose_estimation_2d2d (
    std::vector<KeyPoint> keypoints_1,
    std::vector<KeyPoint> keypoints_2,
    std::vector< DMatch > matches,
    Mat& R, Mat& t );

// 像素坐标转相机归一化坐标
Point2d pixel2cam ( const Point2d& p, const Mat& K );

int main ( int argc, char** argv )
{
    if ( argc != 3 )//命令行参数　 1.png 　2.png
    {
        cout<<"usage: pose_estimation_2d2d img1 img2"<<endl;
        return 1;
    }
    //-- 读取图像
    Mat img_1 = imread ( argv[1], CV_LOAD_IMAGE_COLOR );//彩色图模式
    Mat img_2 = imread ( argv[2], CV_LOAD_IMAGE_COLOR );

    vector<KeyPoint> keypoints_1, keypoints_2;//关键点
    vector<DMatch> matches;//特征点匹配对
    find_feature_matches ( img_1, img_2, keypoints_1, keypoints_2, matches );//得到匹配点对
    cout<<"一共找到了"<<matches.size() <<"组匹配点"<<endl;

    //-- 估计两张图像间运动
    Mat R,t;//旋转和平移 第一张图 到第二章图的坐标变换矩阵和平移矩阵
    pose_estimation_2d2d ( keypoints_1, keypoints_2, matches, R, t );

    //-- 验证E=t^R*scale    变叉乘 为 矩阵乘法   三阶行列式     
    // https://www.cnblogs.com/monoSLAM/p/5349497.html
    //  a   b   c 
    //  d  e    f     = bf - ce , cd - af , ae -bd  =  [0 - c  b; c 0 -a; -b a 0 ] * [ d e f]
    // 向量 t = [ a1 a2 a3] 其
    //  叉乘矩阵 = [0  -a3  a2;
    //                      a3  0  -a1; 
    //                     -a2 a1  0 ]  
    
    Mat t_x = ( Mat_<double> ( 3,3 ) << //t向量的 叉乘矩阵
                0,                      -t.at<double> ( 2,0 ),     t.at<double> ( 1,0 ),
                t.at<double> ( 2,0 ),      0,                      -t.at<double> ( 0,0 ),
                -t.at<double> ( 1.0 ),     t.at<double> ( 0,0 ),      0 );

    cout<<"本质矩阵E = t^R= t叉乘矩阵 * R = "<<endl<<t_x*R<<endl;

    //-- 验证对极约束
    //相机内参数
    //   [fx 0 cx
    //     0 fy cy
    //     0 0  1]
    Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );// 相机内参,TUM Freiburg2
    for ( DMatch m: matches )//Descriptors Match   描述子匹配
    {
        Point2d pt1 = pixel2cam ( keypoints_1[ m.queryIdx ].pt, K );//像素坐标转相机归一化坐标    x1 =  K逆* p1  x2 =  K逆* p2  相机坐标系下 归一化平面上的点 
        Mat y1 = ( Mat_<double> ( 3,1 ) << pt1.x, pt1.y, 1 );// 归一化平面上的点 齐次表示
        Point2d pt2 = pixel2cam ( keypoints_2[ m.trainIdx ].pt, K );
        Mat y2 = ( Mat_<double> ( 3,1 ) << pt2.x, pt2.y, 1 );
        Mat d = y2.t() * t_x * R * y1;//理论上为 0 
        cout << "epipolar constraint = " << d << endl;
    }
    return 0;
}


//特征匹配 计算匹配点对 函数
void find_feature_matches ( const Mat& img_1, const Mat& img_2,
                            std::vector<KeyPoint>& keypoints_1,
                            std::vector<KeyPoint>& keypoints_2,
                            std::vector< DMatch >& matches )
{
    //--------------------第0部初始化------------------------------------------------------
    Mat descriptors_1, descriptors_2;//描述子
    //  OpenCV3 特征点检测器  描述子生成器 用法
    Ptr<FeatureDetector> detector = ORB::create();         //特征点检测器    其他 BRISK   FREAK   
    Ptr<DescriptorExtractor> descriptor = ORB::create();//描述子生成器
    // OpenCV2 特征点检测器  描述子生成器 用法
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );//二进制描述子 汉明点对匹配
    
    //------------------第一步:检测 Oriented FAST 角点位置-----------------------------
    detector->detect ( img_1,keypoints_1 );
    detector->detect ( img_2,keypoints_2 );

    //------------------第二步:根据角点位置计算 BRIEF 描述子-------------------------
    descriptor->compute ( img_1, keypoints_1, descriptors_1 );
    descriptor->compute ( img_2, keypoints_2, descriptors_2 );

    //------------------第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> match;//Descriptors Match   描述子匹配
    //BFMatcher matcher ( NORM_HAMMING );
    matcher->match ( descriptors_1, descriptors_2, match );//各个特征点描述子之间的汉明距离匹配

    //-----------------第四步:匹配点对筛选--------------------------------------------------
    double min_dist=10000, max_dist=0;
    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = matches[i].distance;
        if ( dist < min_dist ) min_dist = dist;  //最短距离  最相似
        if ( dist > max_dist ) max_dist = dist; //最长距离 最不相似
    }
    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );
    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if ( match[i].distance <= max ( 2*min_dist, 30.0 ) )//最大距离
        {
            matches.push_back ( match[i] );
        }
    }
}

// 像素坐标转相机归一化坐标
// 像素坐标转相机归一化坐标    x1 =  K逆* p1  x2 =  K逆* p2  相机坐标系下 归一化平面上的点 
Point2d pixel2cam ( const Point2d& p, const Mat& K )
{
    return Point2d
           (
               ( p.x - K.at<double> ( 0,2 ) ) / K.at<double> ( 0,0 ),// x= (px -cx)/fx
               ( p.y - K.at<double> ( 1,2 ) ) / K.at<double> ( 1,1 )//  y=(py-cy)/fy
           );
}

//特征匹配 计算匹配点对 函数   第一张图 到第二章图的坐标变换矩阵和平移矩阵
//对极几何
void pose_estimation_2d2d ( std::vector<KeyPoint> keypoints_1,
                            std::vector<KeyPoint> keypoints_2,
                            std::vector< DMatch > matches,
                            Mat& R, Mat& t )
{
    // 相机内参,TUM Freiburg2
     //相机内参数
    //   [fx 0 cx
    //     0 fy cy
    //     0 0  1]
    Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );

    //------------把匹配点转换为vector<Point2f>的形式------------------
    vector<Point2f> points1;
    vector<Point2f> points2;
    for ( int i = 0; i < ( int ) matches.size(); i++ )
    {
        points1.push_back ( keypoints_1[matches[i].queryIdx].pt );
        points2.push_back ( keypoints_2[matches[i].trainIdx].pt );
    }

    //-----------计算基础矩阵 F    p2转置*F*p1 = 0   -----------------------------------------------------
    Mat fundamental_matrix;
    fundamental_matrix = findFundamentalMat ( points1, points2, CV_FM_8POINT );//8点发    p2转置*F*p1 = 0 
    cout<<"基础矩阵 fundamental_matrix is "<<endl<< fundamental_matrix<<endl;

    //-----------计算本质矩阵 E   x2转置 * E * x1 = 0  ----------------------------------------------------
    Point2d principal_point ( 325.1, 249.7 );	//相机光心, TUM dataset标定值   cx    cy
    double focal_length = 521;			//相机焦距, TUM dataset标定值  fx     fy
    Mat essential_matrix;
    essential_matrix = findEssentialMat ( points1, points2, focal_length, principal_point );// x2转置 * E * x1 = 0  
    cout<<"本质矩阵 essential_matrix is "<<endl<< essential_matrix<<endl;

    //-----------计算单应矩阵H    p2 = H *p1  ---------------------------------------------------
    Mat homography_matrix;
    homography_matrix = findHomography ( points1, points2, RANSAC, 3 );
    cout<<"单应矩阵 homography_matrix is "<<endl<<homography_matrix<<endl;

    //-- 从本质矩阵中恢复旋转和平移信息. 使用奇异值分解法得到
    recoverPose ( essential_matrix, points1, points2, R, t, focal_length, principal_point );// E = t^R = U C  V   ,U   V 为正交矩阵   C 为奇异值矩阵 C =  diag(1, 1, 0)
    cout<<"R is "<<endl<<R<<endl;
    cout<<"t is "<<endl<<t<<endl;
    
}
