/*
 * ORB特征点检测匹配
 * 其他方法还有SIFT
 * SURF
 */
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>//二维特征提取
#include <opencv2/highgui/highgui.hpp>

using namespace std;//标准库　命名空间
using namespace cv; //opencv库命名空间
// 调用./feature_extraction 1.png 2.png
int main ( int argc, char** argv )
{
    if ( argc != 3 )//命令行参数　 1.png 　2.png
    {
        cout<<"用法:  feature_extraction img1 img2"<<endl;
        return 1;
    }
    //-- 读取图像
    Mat img_1 = imread ( argv[1], CV_LOAD_IMAGE_COLOR );//彩色图模式
    Mat img_2 = imread ( argv[2], CV_LOAD_IMAGE_COLOR );

    //-- 初始化
    std::vector<KeyPoint> keypoints_1, keypoints_2;//关键点容器  二维点
    Mat descriptors_1, descriptors_2;			      //关键点对应的描述子
    Ptr<FeatureDetector> detector = ORB::create();  //cv3下　ORB特征检测    其他 BRISK   FREAK
    //  cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();//cv3下　ORB描述子
    // Ptr<FeatureDetector> detector = FeatureDetector::create(detector_name);
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create(descriptor_name);
   // use this if you are in OpenCV2 
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );
    //二进制描述子汉明距离  匹配

    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect ( img_1,keypoints_1 );
    detector->detect ( img_2,keypoints_2 );

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute ( img_1, keypoints_1, descriptors_1 );
    descriptor->compute ( img_2, keypoints_2, descriptors_2 );

    Mat outimg1;//在原图像画出特征点的图像
    drawKeypoints( img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
    imshow("ORB特征点",outimg1);//显示画上特征点的图像

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离 字符串距离  删除 插入 替换次数
    vector<DMatch> matches;//default默认汉明匹配  容器
    //BFMatcher matcher ( NORM_HAMMING );
    matcher->match ( descriptors_1, descriptors_2, matches );//对两幅照片的特征描述子进行匹配

    //-- 第四步:匹配点对筛选
    double min_dist=10000, max_dist=0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = matches[i].distance;
        if ( dist < min_dist ) min_dist = dist;  //最短距离  最相似
        if ( dist > max_dist ) max_dist = dist; //最长距离 最不相似
    }
    
    // 仅供娱乐的写法
    min_dist = min_element( matches.begin(), matches.end(), [](const DMatch& m1, const DMatch& m2) {return m1.distance<m2.distance;} )->distance;
    max_dist = max_element( matches.begin(), matches.end(), [](const DMatch& m1, const DMatch& m2) {return m1.distance<m2.distance;} )->distance;

    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.
    //但有时候最小距离会非常小,设置一个经验值30作为下限.
    std::vector< DMatch > good_matches;//两幅图像好的特征匹配 点对
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if ( matches[i].distance <= max ( 2*min_dist, 30.0 ) )//最大距离
        {
            good_matches.push_back ( matches[i] );
        }
    }

    //-- 第五步:绘制匹配结果
    Mat img_match;         //全部的匹配点对 图像
    Mat img_goodmatch;//筛选之后的较好的匹配点对图像
    drawMatches ( img_1, keypoints_1, img_2, keypoints_2, matches, img_match );//得到全部的匹配点对 图像
    drawMatches ( img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch );//得到筛选之后的较好的匹配点对图像
    imshow ( "所有匹配点对", img_match );             //显示 全部的匹配点对 图像
    imshow ( "优化后匹配点对", img_goodmatch );//显示筛选之后的较好的匹配点对图像
    waitKey(0);

    return 0;
}
