/**
* This file is part of ORB-SLAM2.
*  单目相机初始化
*  基础矩阵F(随机采样序列 8点法求解) 和 单应矩阵计算( 采用归一化的直接线性变换（normalized DLT）)  相机运动
* 用于平面场景的单应性矩阵H和用于非平面场景的基础矩阵F，
* 然后通过一个评分规则来选择合适的模型，恢复相机的旋转矩阵R和平移向量t。
* 
*/
#ifndef INITIALIZER_H
#define INITIALIZER_H

#include<opencv2/opencv.hpp>
#include "Frame.h"


namespace ORB_SLAM2
{

// THIS IS THE INITIALIZER FOR MONOCULAR SLAM. NOT USED IN THE STEREO OR RGBD CASE.
class Initializer
{
    typedef pair<int,int> Match;// pair  键值对 

public:

    // Fix the reference frame
   // 固定参考帧
    Initializer(const Frame &ReferenceFrame, float sigma = 1.0, int iterations = 200);

    // Computes in parallel a fundamental matrix and a homography
    // 两种方法
    // 计算 基础矩阵 F 和 单应矩阵 H       2D-2D点对映射关系
    // Xc = H * Xr               p2转置 * F * p1 = 0
    // Selects a model and tries to recover the motion and the structure from motion
    // 选择一种方法 恢复 运动
    bool Initialize(const Frame &CurrentFrame, const vector<int> &vMatches12,
                    cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated);


private:
   // 计算单应矩阵 H  随机采样8点对 调用 ComputeH21计算单应  CheckHomography 计算得分  迭代求 得分最高的 H
    void FindHomography(vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21);
   // 计算基础矩阵 F  随机采样8点对 调用 ComputeF21计算基础矩阵  CheckFundamental 计算得分 迭代求 得分最高的 F
    void FindFundamental(vector<bool> &vbInliers, float &score, cv::Mat &F21);
   // 计算单应矩阵 H
    cv::Mat ComputeH21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2);
    // 计算基础矩阵 F
    cv::Mat ComputeF21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2);
    // 计算单应矩阵 得分
    float CheckHomography(const cv::Mat &H21, const cv::Mat &H12, vector<bool> &vbMatchesInliers, float sigma);
    // 计算 基础矩阵 得分
    float CheckFundamental(const cv::Mat &F21, vector<bool> &vbMatchesInliers, float sigma);

    
    // 基础矩阵 恢复  R  t -----F ----> 本质矩阵E 从本质矩阵恢复  R  t
    bool ReconstructF(vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K,
                      cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated);

    // 单应矩阵  恢复  R  t 
    bool ReconstructH(vector<bool> &vbMatchesInliers, cv::Mat &H21, cv::Mat &K,
                      cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated);

    // 三角化计算 深度 获取3D点坐标
    /*
 平面二维点摄影矩阵到三维点  P1 = K × [I 0]    P2 = K * [R  t]
  kp1 = P1 * p3dC1       p3dC1  特征点匹配对 对应的 世界3维点
  kp2 = P2 * p3dC1  
  kp1 叉乘  P1 * p3dC1 =0
  kp2 叉乘  P2 * p3dC1 =0  
 p = ( x,y,1)
 其叉乘矩阵为
     //  叉乘矩阵 = [0  -1  y;
    //              1   0  -x; 
    //              -y   x  0 ]  
  一个方程得到两个约束
  对于第一行 0  -1  y; 会与P的三行分别相乘 得到四个值 与齐次3d点坐标相乘得到 0
  有 (y * P.row(2) - P.row(1) ) * D =0
      (-x *P.row(2) + P.row(0) ) * D =0 ===> (x *P.row(2) - P.row(0) ) * D =0
    两个方程得到 4个约束
    A × D = 0
    对A进行奇异值分解 求解线性方程 得到 D  （D是3维齐次坐标，需要除以第四个尺度因子 归一化）
 */
    void Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D);

    // 标准化点坐标
    // 标准化矩阵  * 点坐标    =   标准化后的的坐标              去均值点坐标 * 绝对矩倒数
   //  点坐标    =    标准化矩阵 逆矩阵 * 标准化后的的坐标
    void Normalize(const vector<cv::KeyPoint> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T);

    // 检查 R t
    int CheckRT(const cv::Mat &R, const cv::Mat &t, const vector<cv::KeyPoint> &vKeys1, const vector<cv::KeyPoint> &vKeys2,
                       const vector<Match> &vMatches12, vector<bool> &vbInliers,
                       const cv::Mat &K, vector<cv::Point3f> &vP3D, float th2, vector<bool> &vbGood, float &parallax);
    // 从本质矩阵 恢复 R t
    // E = t^R = U C  V   ,U   V 为正交矩阵   C 为奇异值矩阵 C =  diag(1, 1, 0)
    void DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t);


    // Keypoints from Reference Frame (Frame 1)
    vector<cv::KeyPoint> mvKeys1;

    // Keypoints from Current Frame (Frame 2)
    vector<cv::KeyPoint> mvKeys2;

    // Current Matches from Reference to Current
    // 参考帧和当前帧的匹配点对
    vector<Match> mvMatches12;//匹配信息
    vector<bool> mvbMatched1;// 是否匹配上

    // Calibration 相机内参数
    cv::Mat mK;

    // Standard Deviation and Variance
    float mSigma, mSigma2;// 标准差 和 方差

    // Ransac max iterations
    int mMaxIterations;// 随机采样序列 最大迭代次数
   //  基础矩阵F(随机采样序列 8点法求解) 和 单应矩阵计算(随机采样序列 4点法求解)  相机运动
   
    // Ransac sets  随机点对序列
    vector<vector<size_t> > mvSets;   

};

} //namespace ORB_SLAM

#endif // INITIALIZER_H
