/* 基于特征点的 物体跟踪
非线性尺度空间的AKAZE特征提取
orb 特征检测 匹配 

 ./planar_tracking blais.mp4 result.avi blais_bb.xml.gz


*/

#include <opencv2/features2d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <iomanip>
#include "stats.h" // Stats structure definition
#include "utils.h" // Drawing and printing functions
using namespace std;
using namespace cv;
const double akaze_thresh = 3e-4;  // AKAZE 检测阈值  1000个点
const double ransac_thresh = 2.5f;  // 随机序列采样 内点阈值 
const double nn_match_ratio = 0.8f; // 近邻匹配点阈值 
const int bb_min_inliers = 100;     // 最小数量的内点数 来画 包围框
const int stats_update_period = 10; // 更新频率

// 定义 跟踪类
class Tracker
{
public:
// 默认构造函数
    Tracker(Ptr<Feature2D> _detector, Ptr<DescriptorMatcher> _matcher) :
        detector(_detector),//特征点检测器
        matcher(_matcher)//描述子匹配器
    {}
// 第一帧图像
    void setFirstFrame(const Mat frame, vector<Point2f> bb, string title, Stats& stats);
// 以后每一帧进行处理 跟踪
    Mat process(const Mat frame, Stats& stats);
// 返回检测器 
    Ptr<Feature2D> getDetector() {
        return detector;
    }
protected:
    Ptr<Feature2D> detector;//特征点检测器
    Ptr<DescriptorMatcher> matcher;//描述子匹配器
    Mat first_frame, first_desc;//
    vector<KeyPoint> first_kp;//关键点
    vector<Point2f> object_bb;//物体的 包围框
};

// 第一帧图像
void Tracker::setFirstFrame(const Mat frame, vector<Point2f> bb, string title, Stats& stats)
{
    first_frame = frame.clone();//复制图像
    detector->detectAndCompute(first_frame, noArray(), first_kp, first_desc);// 检测 + 描述
    stats.keypoints = (int)first_kp.size();// 关键点数量
    drawBoundingBox(first_frame, bb);// 显示 包围框
    putText(first_frame, title, Point(0, 60), FONT_HERSHEY_PLAIN, 5, Scalar::all(0), 4);
    object_bb = bb;// 包围框
}

// 处理
Mat Tracker::process(const Mat frame, Stats& stats)
{
    vector<KeyPoint> kp;//关键点
    Mat desc;//描述子
    detector->detectAndCompute(frame, noArray(), kp, desc);//检测新一帧图像的 关键点和描述子
    stats.keypoints = (int)kp.size();//关键点数量
    vector< vector<DMatch> > matches;//匹配点 索引  二维数组
    vector<KeyPoint> matched1, matched2;// 和上一帧对应 匹配的 关键点坐标
    matcher->knnMatch(first_desc, desc, matches, 2);// 最近领匹配进行匹配 2个匹配
    for(unsigned i = 0; i < matches.size(); i++) {
	// 符合近邻匹配 阈值的 才可能是 关键点
        if(matches[i][0].distance < nn_match_ratio * matches[i][1].distance) {
            matched1.push_back(first_kp[matches[i][0].queryIdx]);//上一帧中对应的关键点
            matched2.push_back(      kp[matches[i][0].trainIdx]);//当前帧中对应的关键点
        }
    }

// 求出初级匹配点对符合 的 单应变换矩阵
    stats.matches = (int)matched1.size();//匹配点的数量
    Mat inlier_mask, homography;
    vector<KeyPoint> inliers1, inliers2;// 符合找到的单应矩阵 的 匹配点
    vector<DMatch> inlier_matches;// 匹配点对是否符合 该单应矩阵
    if(matched1.size() >= 4) {// 求解单应矩阵 需要4个匹配点对以上
        homography = findHomography(Points(matched1), Points(matched2),
                                    RANSAC, ransac_thresh, inlier_mask);
    }
    if(matched1.size() < 4 || homography.empty()) {
        Mat res;// 匹配效果不好  未能求解到 单应矩阵
        hconcat(first_frame, frame, res);
        stats.inliers = 0;
        stats.ratio = 0;
        return res;
    }
// 保留 符合求出的 单应变换矩阵 的 匹配点对 
    for(unsigned i = 0; i < matched1.size(); i++) {
        if(inlier_mask.at<uchar>(i)) {//该匹配点对 符合求出的 单应矩阵
            int new_i = static_cast<int>(inliers1.size());
            inliers1.push_back(matched1[i]);
            inliers2.push_back(matched2[i]);
            inlier_matches.push_back(DMatch(new_i, new_i, 0));
        }
    }
    stats.inliers = (int)inliers1.size();//内点数量
    stats.ratio = stats.inliers * 1.0 / stats.matches;//内点比率

    vector<Point2f> new_bb;//新的物体包围框
    perspectiveTransform(object_bb, new_bb, homography);// 利用 单应变换矩阵 重映射 之前的包围框

    Mat frame_with_bb = frame.clone();// 带有物体包围框的 图像
    if(stats.inliers >= bb_min_inliers) {//内点数量超过阈值 显示 包围框
        drawBoundingBox(frame_with_bb, new_bb);
    }
    // 显示匹配点对
    Mat res;
    drawMatches(first_frame, inliers1, frame_with_bb, inliers2,
                inlier_matches, res,
                Scalar(255, 0, 0), Scalar(255, 0, 0));
    return res;// 返回显示了 包围框 和 匹配点对 的图像
}

// 主函数
int main(int argc, char **argv)
{
    if(argc < 4) {
        cerr << "Usage: " << endl <<
                "akaze_track input_path output_path bounding_box" << endl;
        return 1;
    }

    VideoCapture video_in(argv[1]);//输入视频
    //输出视频
    VideoWriter  video_out(argv[2],
                           (int)video_in.get(CAP_PROP_FOURCC),
                           (int)video_in.get(CAP_PROP_FPS),//帧率
                           Size(2 * (int)video_in.get(CAP_PROP_FRAME_WIDTH),//尺寸
                                2 * (int)video_in.get(CAP_PROP_FRAME_HEIGHT)));
    if(!video_in.isOpened()) {
        cerr << "Couldn't open " << argv[1] << endl;
        return 1;
    }
    if(!video_out.isOpened()) {
        cerr << "Couldn't open " << argv[2] << endl;
        return 1;
    }

    vector<Point2f> bb;// 第一帧所需要跟踪物体 的 包围框
    // 这里可以使用鼠标指定
    FileStorage fs(argv[3], FileStorage::READ);
    if(fs["bounding_box"].empty()) {
        cerr << "Couldn't read bounding_box from " << argv[3] << endl;
        return 1;
    }
    // 第一帧所需要跟踪物体 的 包围框
    fs["bounding_box"] >> bb;
    // 状态
    Stats stats, akaze_stats, orb_stats;
    // AKAZE 特征检测
    Ptr<AKAZE> akaze = AKAZE::create();
    akaze->setThreshold(akaze_thresh);
    // ORB 特征检测 
    Ptr<ORB> orb = ORB::create();
    orb->setMaxFeatures(stats.keypoints);//最多关键点数量

    // 描述子匹配器  汉明字符串距离匹配
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    // 构造跟踪器 对象
    Tracker akaze_tracker(akaze, matcher);
    Tracker orb_tracker(orb, matcher);

    // 第一帧图像
    Mat frame;
    video_in >> frame;
    akaze_tracker.setFirstFrame(frame, bb, "AKAZE", stats);
    orb_tracker.setFirstFrame(frame, bb, "ORB", stats);

    Stats akaze_draw_stats, orb_draw_stats;
    // 总帧数
    int frame_count = (int)video_in.get(CAP_PROP_FRAME_COUNT);
    // 跟踪后的帧 
    Mat akaze_res, orb_res, res_frame;

    for(int i = 1; i < frame_count; i++) {//处理每一帧
        bool update_stats = (i % stats_update_period == 0);
        video_in >> frame;//捕获一帧

//=======AKAZE 跟踪 处理一帧======
        akaze_res = akaze_tracker.process(frame, stats);// AKAZE 跟踪 处理一帧
        akaze_stats += stats;
        if(update_stats) {
            akaze_draw_stats = stats;//更新状态
        }
//=======ORB 跟踪 处理一帧======
        orb->setMaxFeatures(stats.keypoints);//最多关键点数量 根据上一次 检测数量
        orb_res = orb_tracker.process(frame, stats);// ORB 跟踪 处理一帧
        orb_stats += stats;

        if(update_stats) {
            orb_draw_stats = stats;//更新状态
        }

        drawStatistics(akaze_res, akaze_draw_stats);

        drawStatistics(orb_res, orb_draw_stats);

        vconcat(akaze_res, orb_res, res_frame);

        video_out << res_frame;
        cout << i << "/" << frame_count - 1 << endl;
    }

    akaze_stats /= frame_count - 1;

    orb_stats /= frame_count - 1;

    printStatistics("AKAZE", akaze_stats);

    printStatistics("ORB", orb_stats);

    return 0;
}

