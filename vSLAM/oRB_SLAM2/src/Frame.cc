/**
* This file is part of ORB-SLAM2.
*  普通帧 每一幅 图像都会生成 一个帧
* 
************双目相机帧*************
* 左右图
* orb特征提取
* 计算匹配点对
* 由视差计算对应关键点的深度距离
* 
* 双目 立体匹配
* 1】为左目每个特征点建立带状区域搜索表，限定搜索区域，（前已进行极线校正）
* 2】在限定区域内 通过描述子进行 特征点匹配，得到每个特征点最佳匹配点（scaleduR0）
* 3】通过SAD滑窗得到匹配修正量 bestincR
* 4】(bestincR, dist)  (bestincR - 1, dist)  (bestincR +1, dist) 三点拟合出抛物线，得到亚像素修正量 deltaR
* 5】最终匹配点位置 为 : scaleduR0 + bestincR  + deltaR
* 
* 视差 Disparity 和深度Depth
* z = bf /d      b 双目相机基线长度  f为焦距  d为视差(同一点在两相机像素平面 水平方向像素单位差值)
* 
* 为特征点分配网格
******深度相机 帧******************
* 深度值 由未校正特征点 坐标对于应深度图 中的 值确定
* 匹配点坐标横坐标值 为 特征点坐标 横坐标值 - 视差 = 特征点坐标 横坐标值  - bf / 深度
* 为特征点分配网格
********单目相机帧****************
* 深度值容器初始化
* 匹配点坐标 容器初始化
* 为特征点分配网格
* 
* 
* 
*/

#include "Frame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include <thread>

namespace ORB_SLAM2
{

long unsigned int Frame::nNextId=0;
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

Frame::Frame()
{}

//Copy Constructor  默认初始化 主要是 直接赋值 写入到类内 变量
Frame::Frame(const Frame &frame)
    :mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight),
     mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
     mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
     mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn),  mvuRight(frame.mvuRight),
     mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
     mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
     mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mnId(frame.mnId),
     mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
     mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
     mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors),
     mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2)
{
    for(int i=0;i<FRAME_GRID_COLS;i++)//64列
        for(int j=0; j<FRAME_GRID_ROWS; j++)//48行
            mGrid[i][j]=frame.mGrid[i][j];//存vector的数组 格子特征点数 赋值 

    if(!frame.mTcw.empty())
        SetPose(frame.mTcw);
}

//  双目的立体匹配 
Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mpReferenceKF(static_cast<KeyFrame*>(NULL))
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    // 特征点匹配 图像金字塔参数
    mnScaleLevels = mpORBextractorLeft->GetLevels();// 特征提取 图像金字塔 层数
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();//尺度
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    // 左右相机图像 ORB特征提取   未校正的图像  得到 关键点位置后 直接对 关键点坐标进行校正
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft);// 左相机     提取 orb特征点 和描述子  线程 关键点 mvKeys   描述子mDescriptors
    thread threadRight(&Frame::ExtractORB,this,1,imRight);// 又相机 提取 orb特征点 和描述子                       mvKeysRight     mDescriptorsRight
    threadLeft.join();//加入到线程
    threadRight.join();

    N = mvKeys.size();//左图关键点数量 
    if(mvKeys.empty())
        return;

    UndistortKeyPoints();// 对关键点坐标进行校正  只是校正了 左图的 关键点   mvKeys -----> 畸变校正------>  mvKeysUn 

    ComputeStereoMatches();// 计算匹配点对  根据视差计算深度信息  d= mbf/d

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));    
    mvbOutlier = vector<bool>(N,false);


    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)//第一帧 进行计算
    {
       // 对于未校正的图像 计算校正后 图像的 尺寸 mnMinX  , mnMaxX   mnMinY,  mnMaxY
       ComputeImageBounds(imLeft);
       // 640*480 图像 分成 64*48 个网格
       // 计算每个 像素 占据的 网格量   像素差×该量 得到 网格下标
        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);// 每个像素占用的网格数
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;// 双目相机基线长度
   
   // 按照 特征点 的像素坐标  分配到各个网格内
   // 每个网格记录了 特征点的 序列下标
    AssignFeaturesToGrid();
}

// 深度相机 帧结构 灰度图 深度图
Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    // Frame ID
   // int类型  帧id ++
    mnId=nNextId++;

    // Scale Level Info
     // 特征点匹配 图像金字塔参数
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();    
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    // 图像 ORB特征提取   未校正的图像  得到 关键点位置后 直接对 关键点坐标进行校正
    ExtractORB(0,imGray);//  

    N = mvKeys.size();// 特征点数量

    if(mvKeys.empty())
        return;//没有提取到特征点
    // 对关键点坐标进行校正    关键点   mvKeys -----> 畸变校正------>  mvKeysUn 
    UndistortKeyPoints();
    
// 深度相机 计算  深度值 根据未校正的关键点 在深度图中的值 获得
// 匹配点横坐标 有原特征点校正的后横坐标 -  视差；    视差 = bf / 深度
    ComputeStereoFromRGBD(imDepth);

    // 关键点 转成 的地图点
    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)//第一帧 进行计算
    {
      // 对于未校正的图像 计算校正后 图像的 尺寸 mnMinX  , mnMaxX   mnMinY,  mnMaxY
        ComputeImageBounds(imGray);
      // 640*480 图像 分成 64*48 个网格
      // 计算每个 像素 占据的 网格量   像素差×该量 得到 网格下标
        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;// 双目相机基线长度
   
   // 按照 特征点 的像素坐标  分配到各个网格内
   // 每个网格记录了 特征点的 序列下标
    AssignFeaturesToGrid();
}

// 单目图像的帧 一个灰度图
Frame::Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    // 特征点匹配 图像金字塔参数
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    // 图像 ORB特征提取   未校正的图像  得到 关键点位置后 直接对 关键点坐标进行校正
    ExtractORB(0,imGray);

    N = mvKeys.size();// 特征点数量

    if(mvKeys.empty())
        return;

    // 对关键点坐标进行校正    关键点   mvKeys -----> 畸变校正------>  mvKeysUn 
    UndistortKeyPoints();

    // Set no stereo information
    // 初始化 匹配点 横坐标  和对应特征点的深度  单目一开始算不出来 深度 和 匹配点
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);
    
    // 地图点
    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)//第一帧 进行计算
    {
      // 对于未校正的图像 计算校正后 图像的 尺寸 mnMinX  , mnMaxX   mnMinY,  mnMaxY
        ComputeImageBounds(imGray);
      // 640*480 图像 分成 64*48 个网格
      // 计算每个 像素 占据的 网格量   像素差×该量 得到 网格下标
        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;// 双目相机基线长度
   
   // 按照 特征点 的像素坐标  分配到各个网格内
   // 每个网格记录了 特征点的 序列下标
    AssignFeaturesToGrid();
}
// Assign keypoints to the grid for speed up feature matching
// 关键点按网格分配 来加速 匹配
void Frame::AssignFeaturesToGrid()
{
  // 640 *480 的图像  分成 10 64*48 个网格  总关键点 个数 N  每个网格 分到的 关键点个数
    int nReserve = 0.5f*N/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
            mGrid[i][j].reserve(nReserve);//每个小网格容器 又变成 分关键点个数个大小的 子容器 可以动态调整大小

    for(int i=0;i<N;i++)
    {
        const cv::KeyPoint &kp = mvKeysUn[i];//每一个 校正后的关键点

        int nGridPosX, nGridPosY;
        if(PosInGrid(kp,nGridPosX,nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i);//网格容器内 填充关键点的 序列  动态调整大小
    }
}

// 关键点提取 + 描述子
void Frame::ExtractORB(int flag, const cv::Mat &im)
{
    if(flag==0)
        (*mpORBextractorLeft)(im,cv::Mat(),mvKeys,mDescriptors);//左图提取器 提取 关键点 和描述子
    else
        (*mpORBextractorRight)(im,cv::Mat(),mvKeysRight,mDescriptorsRight);// 右图提取器  提取 关键点 和描述子
}

// 更新相机位置姿
void Frame::SetPose(cv::Mat Tcw)
{
    mTcw = Tcw.clone();// 设置位姿 变换矩阵  到 类内 变量
    // Tcw_.copyTo(mTcw);// 拷贝到 类内变量 w2c
    UpdatePoseMatrices();// 更新旋转矩阵 平移向量 世界中心点
}
void Frame::UpdatePoseMatrices()
{ 
    mRcw = mTcw.rowRange(0,3).colRange(0,3);// 世界 到 相机 旋转矩阵
    mRwc = mRcw.t();// t() 逆矩阵                        // 相机 到 世界  旋转矩阵
    mtcw = mTcw.rowRange(0,3).col(3);              // 世界 到 相机 平移向量
    // mtwc  = - mtcw
    mOw = -mRwc*mtcw;// 相机中心点在世界坐标系坐标  相机00点--->mRwc------>mtwc--------
}

// 检查地图点 是否在 当前视野中
// 相机坐标系下 点 深度小于0 点不在视野中
// 像素坐标系下 点 横纵坐标在 校正后的图像尺寸内
bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
    pMP->mbTrackInView = false;// 初始 设置为 不在视野内

    // 3D in absolute coordinates
    cv::Mat P = pMP->GetWorldPos();// 世界坐标系下的点

    // 3D in camera coordinates
    // 3D点转换到 相机坐标系下
    const cv::Mat Pc = mRcw*P+mtcw;  // 世界坐标系 转到 相机坐标系
    const float &PcX = Pc.at<float>(0);
    const float &PcY= Pc.at<float>(1);
    const float &PcZ = Pc.at<float>(2);

    // Check positive depth
    if(PcZ<0.0f)// 深度为 负  错误
        return false;

    // Project in image and check it is not outside
    // 相机像素坐标系下的点   应该在 校正的图像内
    const float invz = 1.0f/PcZ;
    const float u=fx*PcX*invz+cx;
    const float v=fy*PcY*invz+cy;
    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    // Check distance is in the scale invariance region of the MapPoint
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const cv::Mat PO = P-mOw;
    const float dist = cv::norm(PO);
    if(dist<minDistance || dist>maxDistance)
        return false;

   // Check viewing angle
    cv::Mat Pn = pMP->GetNormal();
    const float viewCos = PO.dot(Pn)/dist;
    if(viewCos<viewingCosLimit)
        return false;

    // Predict scale in the image
    const int nPredictedLevel = pMP->PredictScale(dist,this);

    // Data used by the tracking
    pMP->mbTrackInView = true;//在视野中
    pMP->mTrackProjX = u;// 投影点 像素横坐标
    pMP->mTrackProjXR = u - mbf*invz;// 匹配点 像素 横坐标 = 投影点 像素横坐标 - 视差 = 投影点 像素横坐标 - bf / 深度
    pMP->mTrackProjY = v;// 投影点 像素 纵坐标
    pMP->mnTrackScaleLevel= nPredictedLevel;
    pMP->mTrackViewCos = viewCos;

    return true;
}

vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX>=FRAME_GRID_COLS)
        return vIndices;

    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
        return vIndices;

    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mGrid[ix][iy];
            if(vCell.empty())
                continue;

            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                if(bCheckLevels)
                {
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kpUn.octave>maxLevel)
                            continue;
                }

                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}


// 检查点是否在 某个划分的格子内 
bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
   // 被 划分到的 格子坐标
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);
    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;// 不在 某个格子里
    return true;//在某个格子里
}

// 用单词(ORB单词词典) 线性表示 描述子  相当于  一个句子 用几个单词 来表示
void Frame::ComputeBoW()
{
    if(mBowVec.empty())//词典表示向量为空
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);//mat类型转换到 vector类型描述子向量
         // Feature vector associate features with nodes in the 4th level (from leaves up)
        // We assume the vocabulary tree has 6 levels, change the 4 otherwise
	mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);// 计算 描述子向量 用词典线性表示的向量
    }
}

// 对元素图像中的点 利用 畸变 参数进行校正 得到校正后的坐标 
// 不是对 整个图像进行校正(时间长)
void Frame::UndistortKeyPoints()
{
    if(mDistCoef.at<float>(0)==0.0)// 畸变校正参数没有 
    {
        mvKeysUn=mvKeys;//校正后的点 和 未校正 的点坐标 二位像素坐标
        return;
    }
    // Fill matrix with points
    // 关键点坐标（未校正） 转成 opencv mat类型
    cv::Mat mat(N,2,CV_32F);
    for(int i=0; i<N; i++)
    {
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    // 校正关键点坐标 Undistort points
   // 函数的功能：将一些 来自 原始图像的2维点坐标进行校正 得到  相应的矫正点坐标。
    mat=mat.reshape(2);
    cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
    mat=mat.reshape(1);

    // Fill undistorted keypoint vector 填充校正后的坐标
    mvKeysUn.resize(N);
    for(int i=0; i<N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }
}

// 对于未校正的图像 计算校正后 图像的 尺寸
void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    if(mDistCoef.at<float>(0)!=0.0)// 畸变校正参数 有
    {
        cv::Mat mat(4,2,CV_32F);
	// 图像四个顶点位置
	// (0,0)  (col,0)  (0,row)   (col,row)
        mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols; mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; mat.at<float>(3,1)=imLeft.rows;

        // Undistort corners
	// 对图像四个点进行 畸变校正
        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

        mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));
        mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));
        mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));
        mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));

    }
    else // 无畸变校正参数  也就是校正后的图像 图像大小不变
    {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

/*
 * 1】为左目每个特征点建立带状区域搜索表，限定搜索区域，（前已进行极线校正）
* 2】在限定区域内 通过描述子进行 特征点匹配，得到每个特征点最佳匹配点（scaleduR0）   bestIdxR  uR0 = mvKeysRight[bestIdxR].pt.x;   scaleduR0 = round(uR0*scaleFactor);
* 3】通过SAD滑窗得到匹配修正量 bestincR
* 4】(bestincR, dist)  (bestincR - 1, dist)  (bestincR +1, dist) 三点拟合出抛物线，得到亚像素修正量 deltaR
* 5】最终匹配点位置 为 : scaleduR0 + bestincR  + deltaR
 */
// 双目匹配  特征点对匹配
void Frame::ComputeStereoMatches()
{
    mvuRight = vector<float>(N,-1.0f);// 左图关键点 对应 右图匹配点
    mvDepth = vector<float>(N,-1.0f);// 关键点对于的深度

    const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2;// 匹配距离

    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

    //Assign keypoints to row table
    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());

    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);

    const int Nr = mvKeysRight.size();// 右图关键点数量

    // 生成 右图 关键点 匹配块 限制 搜索 带状 区域
    for(int iR=0; iR<Nr; iR++)
    {
        const cv::KeyPoint &kp = mvKeysRight[iR];
        const float &kpY = kp.pt.y;
        const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];
        const int maxr = ceil(kpY+r);
        const int minr = floor(kpY-r);

        for(int yi=minr;yi<=maxr;yi++)
            vRowIndices[yi].push_back(iR);// 右图 带状区域
    }

    // Set limits for search
    const float minZ = mb;
    const float minD = 0;
    const float maxD = mbf/minZ;

    // For each left keypoint search a match in the right image
    // 在右图限定区域内为 左图 关键点 匹配一个 关键点
    vector<pair<int, int> > vDistIdx;
    vDistIdx.reserve(N);

    for(int iL=0; iL<N; iL++)// 每一个 左图关键点 和 限定区域 右图关键点 进行描述子匹配 找到 距离最近的 匹配点  + 修正量
    {
       // 创建引用
        const cv::KeyPoint &kpL = mvKeys[iL];
        const int &levelL = kpL.octave;
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

        const vector<size_t> &vCandidates = vRowIndices[vL];// 关键点匹配 候选区域

        if(vCandidates.empty())
            continue;

        const float minU = uL-maxD;
        const float maxU = uL-minD;

        if(maxU<0)
            continue;

        int bestDist = ORBmatcher::TH_HIGH;
        size_t bestIdxR = 0;

        const cv::Mat &dL = mDescriptors.row(iL);//左图关键点 描述子

        // Compare descriptor to right keypoints  和右图 关键点 描述子进行比较
        for(size_t iC=0; iC<vCandidates.size(); iC++)// 每一个 右图 候选区域 关键点
        {
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = mvKeysRight[iR];//右图关键点

            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                continue;

            const float &uR = kpR.pt.x;

            if(uR>=minU && uR<=maxU)
            {
                const cv::Mat &dR = mDescriptorsRight.row(iR);// 右图  关键点描述子
                const int dist = ORBmatcher::DescriptorDistance(dL,dR);// 二进制描述子 相似度距离   汉明距离

                if(dist<bestDist)
                {
                    bestDist = dist;//保存最小的 匹配距离
                    bestIdxR = iR;// 最小匹配距离对应的 右图点 bestIdxR
                }
            }
        }
        
//
        // Subpixel match by correlation
        // 亚像素修正量
        if(bestDist<thOrbDist)// 匹配距离较小
        {
            // coordinates in image pyramid at keypoint scale
	   // 图像金字塔中的 匹配点坐标 
            const float uR0 = mvKeysRight[bestIdxR].pt.x;
            const float scaleFactor = mvInvScaleFactors[kpL.octave];
            const float scaleduL = round(kpL.pt.x*scaleFactor);
            const float scaledvL = round(kpL.pt.y*scaleFactor);
            const float scaleduR0 = round(uR0*scaleFactor);

            // sliding window search
	    // 滑窗方法
            const int w = 5;
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);
            IL.convertTo(IL,CV_32F);
            IL = IL - IL.at<float>(w,w) *cv::Mat::ones(IL.rows,IL.cols,CV_32F);

            int bestDist = INT_MAX;
            int bestincR = 0;
            const int L = 5;// 10*10窗口
            vector<float> vDists;
            vDists.resize(2*L+1);

            const float iniu = scaleduR0+L-w;
            const float endu = scaleduR0+L+w+1;
            if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                continue;

            for(int incR=-L; incR<=+L; incR++)
            {
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);
                IR.convertTo(IR,CV_32F);
                IR = IR - IR.at<float>(w,w) *cv::Mat::ones(IR.rows,IR.cols,CV_32F);

                float dist = cv::norm(IL,IR,cv::NORM_L1);
                if(dist<bestDist)
                {
                    bestDist =  dist;// 窗口内 最近的点
                    bestincR = incR;//
                }

                vDists[L+incR] = dist;// 距离
            }

            if(bestincR==-L || bestincR==L)
                continue;

            // Sub-pixel match (Parabola fitting)
            const float dist1 = vDists[L+bestincR-1];
            const float dist2 = vDists[L+bestincR];
            const float dist3 = vDists[L+bestincR+1];

            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

            if(deltaR<-1 || deltaR>1)
                continue;

            // Re-scaled coordinate
	    // 最终 右图 匹配点 横坐标   带状块区域 描述子匹配 坐标 + 滑窗校正  + 抛物线拟合校正
            float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

	    
            float disparity = (uL-bestuR);// 双目相机 匹配点对 视差

            if(disparity>=minD && disparity<maxD)
            {
                if(disparity<=0)
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
                mvDepth[iL]=mbf/disparity; //根据视差得到深度信息    z = bf /d      b 双目相机基线长度  f为焦距  d为视差(同一点在两相机像素平面 水平方向像素单位差值)
                mvuRight[iL] = bestuR;
                vDistIdx.push_back(pair<int,int>(bestDist,iL));
            }
        }
    }
// 以上 计算了 所有 关键点的匹配点对(以及最佳匹配距离) 和 对应的深度

    sort(vDistIdx.begin(),vDistIdx.end());// 所有关键点 匹配距离排序
    const float median = vDistIdx[vDistIdx.size()/2].first;// 距离中值
    const float thDist = 1.5f*1.4f*median;

    for(int i=vDistIdx.size()-1;i>=0;i--)
    {
        if(vDistIdx[i].first<thDist)
            break;
        else // 距离过大的匹配点对 任务是误匹配  设置为 -1 用以标记
        {
            mvuRight[vDistIdx[i].second]=-1;
            mvDepth[vDistIdx[i].second]=-1;
        }
    }
}

// 深度相机 计算  深度值 根据未校正的关键点 在深度图中的值 获得
// 匹配点横坐标 有原特征点校正的后横坐标 -  视差；    视差 = bf / 深度
void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
{
    mvuRight = vector<float>(N,-1);// 初始匹配点
    mvDepth = vector<float>(N,-1);// 初始深度

    for(int i=0; i<N; i++)
    {
        const cv::KeyPoint &kp = mvKeys[i];// 关键点 未校正  用于从深度图 获取深度值
        const cv::KeyPoint &kpU = mvKeysUn[i];//校正后的 关键点

        const float &v = kp.pt.y;
        const float &u = kp.pt.x;

        const float d = imDepth.at<float>(v,u);// 对应的深度值 

        if(d>0)
        {
            mvDepth[i] = d;
            mvuRight[i] = kpU.pt.x-mbf/d;// 深度 = bf / 视差  ---> 视差 = bf / 深度  ----> 原校正后的坐标 - 视差 得到匹配点 x方向坐标值
        }
    }
}

// 得到 世界坐标系 坐标
cv::Mat Frame::UnprojectStereo(const int &i)
{
    const float z = mvDepth[i];// 深度值
    if(z>0)
    {
      // 像素坐标值
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;
      // 像极坐标系 坐标
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);
        return mRwc*x3Dc+mOw;
    }
    else
        return cv::Mat();
}

} //namespace ORB_SLAM
