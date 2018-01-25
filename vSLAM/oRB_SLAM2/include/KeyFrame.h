/**
* This file is part of ORB-SLAM2.
* 关键帧
* 
* 普通帧里面精选出来的具有代表性的帧
* 
* 
*/

#ifndef KEYFRAME_H
#define KEYFRAME_H

#include "MapPoint.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "ORBVocabulary.h"
#include "ORBextractor.h"
#include "Frame.h"
#include "KeyFrameDatabase.h"

#include <mutex>


namespace ORB_SLAM2
{

class Map;// 地图
class MapPoint;// 地图点
class Frame;// 普通帧
class KeyFrameDatabase;//关键帧数据库  存储关键点 位姿态等信息 用于匹配

class KeyFrame
{
public:
  // 初始化关键帧 
    KeyFrame(Frame &F, Map* pMap, KeyFrameDatabase* pKFDB);

    // Pose functions
    void SetPose(const cv::Mat &Tcw);
    cv::Mat GetPose();// 位姿
    cv::Mat GetPoseInverse();// 位姿
    cv::Mat GetCameraCenter();// 单目 相机中心
    cv::Mat GetStereoCenter();// 双目 相机中心
    cv::Mat GetRotation();// 旋转矩阵
    cv::Mat GetTranslation();// 平移向量

    // Bag of Words Representation
    // 计关键点 描述子 的  词典线性表示向量
    void ComputeBoW();

    // Covisibility graph functions
    // 可视化 添加 线连接  关键点连线
    void AddConnection(KeyFrame* pKF, const int &weight);
    void EraseConnection(KeyFrame* pKF);// 删除 线连接
    void UpdateConnections();// 跟新线连接
    void UpdateBestCovisibles();
    std::set<KeyFrame *> GetConnectedKeyFrames();
    std::vector<KeyFrame* > GetVectorCovisibleKeyFrames();
    std::vector<KeyFrame*> GetBestCovisibilityKeyFrames(const int &N);
    std::vector<KeyFrame*> GetCovisiblesByWeight(const int &w);
    int GetWeight(KeyFrame* pKF);

    // Spanning tree functions 生成 关键帧 树 
    void AddChild(KeyFrame* pKF);// 添加孩子
    void EraseChild(KeyFrame* pKF);// 删除孩子
    void ChangeParent(KeyFrame* pKF);// 跟换父亲
    std::set<KeyFrame*> GetChilds();// 得到孩子
    KeyFrame* GetParent();//得到父亲
    bool hasChild(KeyFrame* pKF);// 有孩子吗

    // Loop Edges 环边
    void AddLoopEdge(KeyFrame* pKF);// 添加
    std::set<KeyFrame*> GetLoopEdges();// 获取

    // MapPoint observation functions
    // 地图点 观测函数
    void AddMapPoint(MapPoint* pMP, const size_t &idx);// 添加 地图点
    void EraseMapPointMatch(const size_t &idx);// 删除地图点匹配
    void EraseMapPointMatch(MapPoint* pMP);
    void ReplaceMapPointMatch(const size_t &idx, MapPoint* pMP);// 替换地图点匹配
    std::set<MapPoint*> GetMapPoints();// 得到地图点集合 set 指针
    std::vector<MapPoint*> GetMapPointMatches();// 得到地图点匹配
    int TrackedMapPoints(const int &minObs);// 跟踪到的地图点
    MapPoint* GetMapPoint(const size_t &idx);//得到单个地图点

    // KeyPoint functions
    // 关键点 方程
    std::vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r) const;
    cv::Mat UnprojectStereo(int i);

    // Image
    // 在图像里吗
    bool IsInImage(const float &x, const float &y) const;

    // Enable/Disable bad flag changes
    // 使能/关能 
    void SetNotErase();
    void SetErase();

    // Set/check bad flag
    // 设置标志  检查标志
    void SetBadFlag();
    bool isBad();

    // Compute Scene Depth (q=2 median). Used in monocular.
    // 单目 环境 深度中指
    float ComputeSceneMedianDepth(const int q);

    // 比较函数
    static bool weightComp( int a, int b){
        return a>b;
    }
   // 关键帧 先后顺序 
    static bool lId(KeyFrame* pKF1, KeyFrame* pKF2){
        return pKF1->mnId<pKF2->mnId;
    }


    // The following variables are accesed from only 1 thread or never change (no mutex needed).
    // 变量
public:

    static long unsigned int nNextId;// 下一个关键帧 id
    long unsigned int mnId;// 本关键帧 id
    const long unsigned int mnFrameId;// 帧id

    const double mTimeStamp;// 时间戳

    // Grid (to speed up feature matching)
    // 640 *480 图像 分成 64 × 48 个格子 加速 特征匹配
    const int mnGridCols;// 64 列
    const int mnGridRows;// 48 行
    const float mfGridElementWidthInv;// 每一像素 占有的 格子数量
    const float mfGridElementHeightInv;

    // Variables used by the tracking
    long unsigned int mnTrackReferenceForFrame;// 参考帧
    long unsigned int mnFuseTargetForKF;

    // Variables used by the local mapping
    // 本地地图  最小化重投影误差 BA参数
    long unsigned int mnBALocalForKF;
    long unsigned int mnBAFixedForKF;

    // Variables used by the keyframe database
    // 关键帧数据库 变量参数
    long unsigned int mnLoopQuery;
    int mnLoopWords;
    float mLoopScore;
    long unsigned int mnRelocQuery;
    int mnRelocWords;
    float mRelocScore;

    // Variables used by loop closing
    // 闭环检测 变量
    cv::Mat mTcwGBA;
    cv::Mat mTcwBefGBA;
    long unsigned int mnBAGlobalForKF;

    // Calibration parameters
    // 校准参数 相机内参 基线*焦距 基线 深度
    const float fx, fy, cx, cy, invfx, invfy, mbf, mb, mThDepth;

    // Number of KeyPoints
    // 关键点数量
    const int N;// 

    // KeyPoints, stereo coordinate and descriptors (all associated by an index)
    // 关键点  校正后的关键点 匹配点横坐标 深度值  描述子
    const std::vector<cv::KeyPoint> mvKeys;// 关键点
    const std::vector<cv::KeyPoint> mvKeysUn;// 校正后的关键点
    const std::vector<float> mvuRight; // negative value for monocular points
    const std::vector<float> mvDepth; // negative value for monocular points
    const cv::Mat mDescriptors;// 描述子

    //BoW
    // 词带模型
    DBoW2::BowVector mBowVec;     // 描述子 词典单词 线性表示 的向量
    DBoW2::FeatureVector mFeatVec;// feature vector of nodes and feature indexes

    // Pose relative to parent (this is computed when bad flag is activated)
    cv::Mat mTcp;// 父位姿 到 当前关键帧 转换矩阵

    // Scale  图像金字塔尺度信息
    const int mnScaleLevels;
    const float mfScaleFactor;
    const float mfLogScaleFactor;
    const std::vector<float> mvScaleFactors;
    const std::vector<float> mvLevelSigma2;
    const std::vector<float> mvInvLevelSigma2;

    // Image bounds and calibration
    // 图像四个顶点  畸变校正后 得到 的图像 尺寸
    const int mnMinX;
    const int mnMinY;
    const int mnMaxX;
    const int mnMaxY;
    const cv::Mat mK;


    // The following variables need to be accessed trough a mutex to be thread safe.
protected:

    // SE3 Pose and camera center
    cv::Mat Tcw;// 世界 到 相机
    cv::Mat Twc;// 相机 到 世界
    cv::Mat Ow;//

    cv::Mat Cw; // 相机中心点坐标 Stereo middel point. Only for visualization

    // MapPoints associated to keypoints
    // 地图点
    std::vector<MapPoint*> mvpMapPoints;

    // BoW
    // 关键帧数据库  特征字典
    KeyFrameDatabase* mpKeyFrameDB;
    ORBVocabulary* mpORBvocabulary;

    // Grid over the image to speed up feature matching
    // 格点 64*48个容器 每个容器内是一个容器 存储着 关键点 的 id
    std::vector< std::vector <std::vector<size_t> > > mGrid;

    std::map<KeyFrame*,int> mConnectedKeyFrameWeights;// 帧 和 权重  key ：value 数对
    std::vector<KeyFrame*> mvpOrderedConnectedKeyFrames;
    std::vector<int> mvOrderedWeights;

    // Spanning Tree and Loop Edges
    // 生成树
    bool mbFirstConnection;
    KeyFrame* mpParent;//父节点
    std::set<KeyFrame*> mspChildrens;//子节点
    std::set<KeyFrame*> mspLoopEdges;//环边

    // Bad flags
    bool mbNotErase;
    bool mbToBeErased;
    bool mbBad;    

    float mHalfBaseline; // Only for visualization  基线的一半 用于显示 

    Map* mpMap;

    std::mutex mMutexPose;
    std::mutex mMutexConnections;
    std::mutex mMutexFeatures;
};

} //namespace ORB_SLAM

#endif // KEYFRAME_H
