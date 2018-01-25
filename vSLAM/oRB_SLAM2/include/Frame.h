/**
* This file is part of ORB-SLAM2.
* 普通帧 每一幅 图像都会生成 一个帧
* 
* 
*/

#ifndef FRAME_H
#define FRAME_H

#include<vector>

#include "MapPoint.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "ORBVocabulary.h"
#include "KeyFrame.h"
#include "ORBextractor.h"

#include <opencv2/opencv.hpp>

namespace ORB_SLAM2
{
// 640 *480 的图像  分成 10 64*48 个网格
#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

class MapPoint;
class KeyFrame;

class Frame
{
public:
    Frame();

    // Copy constructor.
    // 帧 初始化
    Frame(const Frame &frame);

    // 双目相机帧 左图 右图  Constructor for stereo cameras.
    Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth);

    // 深度相机帧 灰度图 深度图 Constructor for RGB-D cameras.
    Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth);

    // 单目相机帧 灰度图Constructor for Monocular cameras.
    Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth);

    // 提取ORB特征点和描述子 Extract ORB on the image. 0 for left image and 1 for right image.
    void ExtractORB(int flag, const cv::Mat &im);

    // Compute Bag of Words representation.
    // 计算 特征词点 描述子
    void ComputeBoW();

    // 相机位姿  Set the camera pose.
    void SetPose(cv::Mat Tcw);

    // 更新相机描述矩阵 旋转 平移 中心点 Computes rotation, translation and camera center matrices from the camera pose.
    void UpdatePoseMatrices();

    // 返回相机中心点 世界坐标系下  Returns the camera center.
    inline cv::Mat GetCameraCenter(){
        return mOw.clone();
    }

    // 世界坐标系转到像极坐标系下 R  Returns inverse of rotation
    inline cv::Mat GetRotationInverse(){
        return mRwc.clone();
    }

    // Check if a MapPoint is in the frustum of the camera  摄像机在给定距离的视锥体
    // and fill variables of the MapPoint to be used by the tracking
    //地图点 是否在当前帧内
    bool isInFrustum(MapPoint* pMP, float viewingCosLimit);

    // Compute the cell of a keypoint (return false if outside the grid)
    bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);

    vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel=-1, const int maxLevel=-1) const;

    // Search a match for each keypoint in the left image to a keypoint in the right image.
    // If there is a match, depth is computed and the right coordinate associated to the left keypoint is stored.
    // 对于双目相机 的 左右图 orb特征匹配程序  计算特征匹配点对 根据视差计算深度
    void ComputeStereoMatches();

    // Associate a "right" coordinate to a keypoint if there is valid depth in the depthmap.
    // 对于 深度相机
    void ComputeStereoFromRGBD(const cv::Mat &imDepth);

    // Backprojects a keypoint (if stereo/depth info available) into 3D world coordinates.
    // 关键点 映射到 3D 相机坐标系下
    cv::Mat UnprojectStereo(const int &i);

public:
    // Vocabulary used for relocalization.
   // 重定位 orb 词典 支持将 描述子 转成 用 单词线性表示的 向量
    ORBVocabulary* mpORBvocabulary;

    // Feature extractor. The right is used only in the stereo case.
    // ORB特征提取器
    ORBextractor* mpORBextractorLeft, *mpORBextractorRight;

    // Frame timestamp. 帧时间戳 来自 原始拍摄图像的 时间戳
    double mTimeStamp;

    // Calibration matrix and OpenCV distortion parameters.
    // 相机内参数
    cv::Mat mK;
    static float fx;
    static float fy;
    static float cx;
    static float cy;
    static float invfx;// 1/fx
    static float invfy;// 1/fy
    cv::Mat mDistCoef;// 畸变校正参数 k1 k2 p1 p2 k3

    // Stereo baseline multiplied by fx. 
    // 双目 基线长度 * 焦距  除以视差 得到深度信息
    float mbf;//  z = bf /d      b 双目相机基线长度  f为焦距  d为视差(同一点在两相机像素平面 水平方向像素单位差值)

    // Stereo baseline in meters.
    float mb;// 基线长度  单位米

    // Threshold close/far points. Close points are inserted from 1 view.
    // Far points are inserted as in the monocular case from 2 views.
    // 远近点 深度阈值
    float mThDepth;

    // Number of KeyPoints.
    // 关键点数量
    int N;

    // Vector of keypoints (original for visualization) and undistorted (actually used by the system).
    // In the stereo case, mvKeysUn is redundant as images must be rectified.
    // In the RGB-D case, RGB images can be distorted.
    std::vector<cv::KeyPoint> mvKeys, mvKeysRight;// 左图关键点  右图关键当年
    std::vector<cv::KeyPoint> mvKeysUn;//校正后的关键点

    // Corresponding stereo coordinate and depth for each keypoint.
    // "Monocular" keypoints have a negative value.
    std::vector<float> mvuRight;
    std::vector<float> mvDepth;// 双目相机 关键点对应的深度

    // Bag of Words Vector structures.
    // 词带
    DBoW2::BowVector mBowVec;
    DBoW2::FeatureVector mFeatVec;

    // ORB descriptor, each row associated to a keypoint.
    // ORB关键点描述子
    cv::Mat mDescriptors, mDescriptorsRight;

    // MapPoints associated to keypoints, NULL pointer if no association.
    std::vector<MapPoint*> mvpMapPoints;

    // Flag to identify outlier associations.
    std::vector<bool> mvbOutlier;

    // Keypoints are assigned to cells in a grid to reduce matching complexity when projecting MapPoints.
    // 关键点被 分配到 64*48个网格内  来降低匹配复杂度
    static float mfGridElementWidthInv;
    static float mfGridElementHeightInv;
    // 格点 64*48个数组 每个数组内是一个容器 存储着 关键点 的 id
    std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

    // Camera pose.
    // 相机位姿
    cv::Mat mTcw;

    // Current and Next Frame id.
    static long unsigned int nNextId;
    long unsigned int mnId;

    // Reference Keyframe. 参考关键帧
    KeyFrame* mpReferenceKF;

    // Scale pyramid info.
    // 图像金字塔
    int mnScaleLevels;
    float mfScaleFactor;
    float mfLogScaleFactor;
    vector<float> mvScaleFactors;
    vector<float> mvInvScaleFactors;
    vector<float> mvLevelSigma2;
    vector<float> mvInvLevelSigma2;

    // Undistorted Image Bounds (computed once).
    static float mnMinX;
    static float mnMaxX;
    static float mnMinY;
    static float mnMaxY;

    static bool mbInitialComputations;


private:

    // Undistort keypoints given OpenCV distortion parameters.
    // Only for the RGB-D case. Stereo must be already rectified!
    // (called in the constructor).
    void UndistortKeyPoints();

    // Computes image bounds for the undistorted image (called in the constructor).
    void ComputeImageBounds(const cv::Mat &imLeft);

    // Assign keypoints to the grid for speed up feature matching (called in the constructor).
    void AssignFeaturesToGrid();

    // Rotation, translation and camera center
    cv::Mat mRcw;// 旋转矩阵 世界坐标系 到  相机坐标系 
    cv::Mat mtcw;// 平移向量
    cv::Mat mRwc;//旋转矩阵  相机坐标系 到  世界坐标系 
    cv::Mat mOw; //==mtwc
};

}// namespace ORB_SLAM

#endif // FRAME_H
