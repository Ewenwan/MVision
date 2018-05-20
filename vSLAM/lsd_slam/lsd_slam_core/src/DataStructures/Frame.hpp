/**
* 帧这玩意儿贯穿始终，是slam中最基本的数据结构，
* 我觉得想要理解这个类，应
* 该从类中的结构体Data开始
* 
* 每张图像创建 5层的图像金字塔  每一层的尺度 变为上一层的1/2
* 图像的 内参数 也上上一层的 1/2
* 内参数求逆得到 内参数逆矩阵
* 
* 一、图像金字塔构建方法为 ：
* 上一层 的 四个像素的值的平均值合并成一个像素为下一层的像素
* 
* 	int wh = width*height;// 当前层 像素总数
*	const float* s;
*	for(int y=0; y<wh; y += width*2)// 隔行
*	{
*		for(int x=0; x<width; x+= 2)// 隔列下采样
*		{
*			s = source + x + y;// 上一层 像素对应位置
*			*dest = (s[0] +
*					s[1] +
*					s[width] +
*					s[1+width]) * 0.25f;// 四个像素的值的平均值合并成一个
*			dest++;
*		}
*	}
* 
* 二、梯度金字塔构建方法（四个值  dx ， dy， i， null)
* 使用同一层的 图像  左右像素求得x方向梯度  上下求得 方向梯度 
*           *(img_pt-width)
*  val_m1  *(img_pt)   val_p1
*           *(img_pt+width)
* 1.  (val_p1 - val_m1)/2    = x 方向梯度
* 2.  0.5f*(*(img_pt+width) - *(img_pt-width)) = y方向梯度
* 3.  val_00 = *(img_pt)   当前 点像素值
* 4. 第四维度 没有存储数据    gradxyii_pt  Eigen::Vector4f
*
* 
* 三、临近最大合成梯度 值 地图构建 一个合成梯度值
*  创建 梯度图内 临近四点中梯度最大值 的 最大值梯度 图 ， 并记录梯度值较大的可以映射 成 地图点的数量
* 在梯度图中 求去合成梯度 g=sqrt(gx^2+gy^2)  ，求的 上中下 三个梯度值中的最大值，形成临时梯度最大值图
* 在临时梯度最大值图 中求 的  左中右 三个梯度值中的最大值，形成最后的 最大梯度值地图
*  并记录 最大梯度大小超过阈值的点 可以映射成地图点  
* 
* 四、构建 第0层 逆深度均值图 和方差图
* 1. 使用 真实 深度值  取反得到逆深度值，方差初始为一个设定值
* 2. 没有真实值是，也可以使用高斯分布均值初始化 逆深度均值图 和方差图
* 
* 五、高层逆深度均值金字塔图 和逆深度方差金字塔图的构建
* 
*  根据逆深度 构建  逆深度均值图 方差图(高斯分布)  金字塔
*       current   -----> 右边一个
*       下边                下右边       上一层四个位置 
*  上一层 逆方差和  /  上一层 逆深度均值 (四个位置处) 和  得到深度信息 再 取逆得到 逆深度均值
*  上一层 逆深度 方差和 取逆得到 本层 逆深度方差 * 
* 
*/

#pragma once
#include "util/SophusUtil.h"
#include "util/settings.h"
#include <boost/thread/recursive_mutex.hpp>
#include <boost/thread/shared_mutex.hpp>
#include "DataStructures/FramePoseStruct.h"
#include "DataStructures/FrameMemory.h"
#include "unordered_set"
#include "util/settings.h"


namespace lsd_slam
{


    class DepthMapPixelHypothesis;
    class TrackingReference;
    /**
    */

    class Frame
    {
    public:
	    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	    friend class FrameMemory;// 友元类   FrameMemory 可以访问 Frame 

    // 类构造函数   id   宽 高 相机内参数  时间戳    图像  0~255
	    Frame(int id, int width, int height, const Eigen::Matrix3f& K, double timestamp, const unsigned char* image);

	    Frame(int id, int width, int height, const Eigen::Matrix3f& K, double timestamp, const float* image);
    // 类析构函数
	    ~Frame();
	    
    // 更新 逆深度信息	
	    /** Sets or updates idepth and idepthVar on level zero. Invalidates higher levels. */
	    void setDepth(const DepthMapPixelHypothesis* newDepth);
    // 均值信息
	    /** Calculates mean information for statistical purposes. */
	    void calculateMeanInformation();
    // 设置 深度信息	
	    /** Sets ground truth depth (real, not inverse!) from a float array on level zero. Invalidates higher levels. */
	    void setDepthFromGroundTruth(const float* depth, float cov_scale = 1.0f);
    // 两帧 组成双目 三角测量 相机内参数  图像金字塔等级	
	    /** Prepares this frame for stereo comparisons with the other frame (computes some intermediate values that will be needed) */
	    void prepareForStereoWith(Frame* other, Sim3 thisToOther, const Eigen::Matrix3f& K, const int level);

	    

	    // Accessors
	    /** Returns the unique frame id. */
	    inline int id() const;
	    
	    /** Returns the frame's image width. */
	    inline int width(int level = 0) const;
	    /** Returns the frame's image height. */
	    inline int height(int level = 0) const;
	    
	    /** Returns the frame's intrinsics matrix. */
	    inline const Eigen::Matrix3f& K(int level = 0) const;// 相机内参数
	    /** Returns the frame's inverse intrinsics matrix. */
	    inline const Eigen::Matrix3f& KInv(int level = 0) const;// 相机内参数逆 
	    /** Returns K(0, 0). */
	    inline float fx(int level = 0) const;
	    /** Returns K(1, 1). */
	    inline float fy(int level = 0) const;
	    /** Returns K(0, 2). */
	    inline float cx(int level = 0) const;
	    /** Returns K(1, 2). */
	    inline float cy(int level = 0) const;
	    /** Returns KInv(0, 0). */
	    inline float fxInv(int level = 0) const;
	    /** Returns KInv(1, 1). */
	    inline float fyInv(int level = 0) const;
	    /** Returns KInv(0, 2). */
	    inline float cxInv(int level = 0) const;
	    /** Returns KInv(1, 2). */
	    inline float cyInv(int level = 0) const;
	    
	    /** Returns the frame's recording timestamp. */
	    inline double timestamp() const;// 时间戳
	    
	    inline float* image(int level = 0);// float 类型的image 
	    inline const Eigen::Vector4f* gradients(int level = 0);
	    inline const float* maxGradients(int level = 0);
	    inline bool hasIDepthBeenSet() const;
	    inline const float* idepth(int level = 0);// 逆深度
	    inline const float* idepthVar(int level = 0);
	    inline const unsigned char* validity_reAct();
	    inline const float* idepth_reAct();
	    inline const float* idepthVar_reAct();

	    inline bool* refPixelWasGood();
	    inline bool* refPixelWasGoodNoCreate();
	    inline void clear_refPixelWasGood();

	    /** Flags for use with require() and requirePyramid(). See the Frame class
	      * documentation for their exact meaning. */
	    enum DataFlags
	    {
		    IMAGE			= 1<<0,
		    GRADIENTS		= 1<<1,
		    MAX_GRADIENTS	= 1<<2,
		    IDEPTH			= 1<<3,//逆深度值
		    IDEPTH_VAR		= 1<<4,
		    REF_ID			= 1<<5,// 参考帧
		    
		    ALL = IMAGE | GRADIENTS | MAX_GRADIENTS | IDEPTH | IDEPTH_VAR | REF_ID
	    };
	    

	    void setPermaRef(TrackingReference* reference);
	    void takeReActivationData(DepthMapPixelHypothesis* depthMap);


	    // shared_lock this as long as any minimizable arrays are being used.
	    // the minimizer will only minimize frames after getting
	    // an exclusive lock on this.
	    inline boost::shared_lock<boost::shared_mutex> getActiveLock()
	    {
		    return FrameMemory::getInstance().activateFrame(this);// 激活帧
	    }


	    /*
	    * ==================================================================================
	    * Here are ALL central pose and scale informations.
	    * generally, everything is stored relative to the frame
	    */
	    FramePoseStruct* pose;// 帧位姿
	    Sim3 getScaledCamToWorld(int num=0) { return pose->getCamToWorld();}// 相机到世界 的坐标变换
	    bool hasTrackingParent() { return pose->trackingParent != nullptr;}// 跟踪父亲帧
	    Frame* getTrackingParent() { return pose->trackingParent->frame;}// 参考帧

	    Sim3 lastConstraintTrackedCamToWorld;



	    /** Pointers to all adjacent Frames in graph. empty for non-keyframes.*/
	    std::unordered_set< Frame*, std::hash<Frame*>, std::equal_to<Frame*>,
		    Eigen::aligned_allocator< Frame* > > neighbors;// 邻居帧

	    /** Multi-Map indicating for which other keyframes with which initialization tracking failed.*/
	    std::unordered_multimap< Frame*, Sim3, std::hash<Frame*>, std::equal_to<Frame*>,
		    Eigen::aligned_allocator< std::pair<const Frame*,Sim3> > > trackingFailed;


	    // flag set when depth is updated.
	    bool depthHasBeenUpdatedFlag;


	    // Tracking Reference for quick test. Always available, never taken out of memory.
	    // this is used for re-localization and re-Keyframe positioning.
	    boost::mutex permaRef_mutex;
	    Eigen::Vector3f* permaRef_posData;	// (x,y,z)
	    Eigen::Vector2f* permaRef_colorAndVarData;	// (I, Var)
	    int permaRefNumPts;



	    // Temporary values
	    int referenceID;
	    int referenceLevel;
	    float distSquared;
	    Eigen::Matrix3f K_otherToThis_R;
	    Eigen::Vector3f K_otherToThis_t;
	    Eigen::Vector3f otherToThis_t;
	    Eigen::Vector3f K_thisToOther_t;
	    Eigen::Matrix3f thisToOther_R;
	    Eigen::Vector3f otherToThis_R_row0;
	    Eigen::Vector3f otherToThis_R_row1;
	    Eigen::Vector3f otherToThis_R_row2;
	    Eigen::Vector3f thisToOther_t;



	    // statistics
	    float initialTrackedResidual;
	    int numFramesTrackedOnThis;
	    int numMappedOnThis;
	    int numMappedOnThisTotal;
	    float meanIdepth;
	    int numPoints;
	    int idxInKeyframes;
	    float edgeErrorSum, edgesNum;
	    int numMappablePixels;
	    float meanInformation;

    private:

	    void require(int dataFlags, int level = 0);
	    void release(int dataFlags, bool pyramidsOnly, bool invalidateOnly);
// 初始化
	    void initialize(int id, int width, int height, const Eigen::Matrix3f& K, double timestamp);
	    void setDepth_Allocate();
// 图像金字塔	    
	    void buildImage(int level);
	    void releaseImage(int level);
// 梯度金字塔	    
	    void buildGradients(int level);
	    void releaseGradients(int level);
// 最大梯度值金字塔	    
	    void buildMaxGradients(int level);
	    void releaseMaxGradients(int level);
// 逆深度均值&方差金字塔	    
	    void buildIDepthAndIDepthVar(int level);
	    void releaseIDepth(int level);
	    void releaseIDepthVar(int level);
	    
	    void printfAssert(const char* message) const;
	    
	    // 结构体Data
	    struct Data
	    {
		    int id;// 帧id
		    // 定义了一个金字塔，PYRAMID_LEVELS这个宏在setting.h头文件中
		    int width[PYRAMID_LEVELS], height[PYRAMID_LEVELS];// 5层金字塔
		  // 相机内参数
		    Eigen::Matrix3f K[PYRAMID_LEVELS], KInv[PYRAMID_LEVELS];//相机内参数 内参数的逆
		    float fx[PYRAMID_LEVELS], fy[PYRAMID_LEVELS], cx[PYRAMID_LEVELS], cy[PYRAMID_LEVELS];
		    float fxInv[PYRAMID_LEVELS], fyInv[PYRAMID_LEVELS], cxInv[PYRAMID_LEVELS], cyInv[PYRAMID_LEVELS];
		  // 时间戳
		    double timestamp;

		    
		    float* image[PYRAMID_LEVELS];// 图像金字塔
		    bool imageValid[PYRAMID_LEVELS];// 每一层 图像金字塔 的图像 指针 是否已经指向了 相应的图像数据
		    
		    Eigen::Vector4f* gradients[PYRAMID_LEVELS];// 梯度金字塔 [gx gy i null]
		    bool gradientsValid[PYRAMID_LEVELS];
		    
		    float* maxGradients[PYRAMID_LEVELS];// 最大梯度值金字塔
		    bool maxGradientsValid[PYRAMID_LEVELS];
		    

		    bool hasIDepthBeenSet;

		    // negative depthvalues are actually allowed, so setting this to -1 does NOT invalidate the pixel's depth.
		    // a pixel is valid iff idepthVar[i] > 0.
		    float* idepth[PYRAMID_LEVELS];//  逆深度金字塔
		    bool idepthValid[PYRAMID_LEVELS];
		    
		    // MUST contain -1 for invalid pixel (that dont have depth)!!
		    float* idepthVar[PYRAMID_LEVELS];// 逆深度方差 金字塔
		    bool idepthVarValid[PYRAMID_LEVELS];

		    // data needed for re-activating the frame. theoretically, this is all data the
		    // frame contains.
		    unsigned char* validity_reAct;
		    float* idepth_reAct;
		    float* idepthVar_reAct;
		    bool reActivationDataValid;


		    // data from initial tracking, indicating which pixels in the reference frame ware good or not.
		    // deleted as soon as frame is used for mapping.
		    bool* refPixelWasGood;
	    };
	    Data data;


	    // used internally. locked while something is being built, such that no
	    // two threads build anything simultaneously. not locked on require() if nothing is changed.
	    boost::mutex buildMutex;//   两个互斥锁
	    boost::shared_mutex activeMutex;
	    bool isActive;// 是否活的flag

	    /** Releases everything which can be recalculated, but keeps the minimal
	      * representation in memory. Use release(Frame::ALL, false) to store on disk instead.
	      * ONLY CALL THIS, if an exclusive lock on activeMutex is owned! */
	    bool minimizeInMemory();
    };



    inline int Frame::id() const
    {
	    return data.id;
    }

    inline int Frame::width(int level) const
    {
	    return data.width[level];
    }

    inline int Frame::height(int level) const
    {
	    return data.height[level];
    }

    inline const Eigen::Matrix3f& Frame::K(int level) const
    {
	    return data.K[level];
    }
    inline const Eigen::Matrix3f& Frame::KInv(int level) const
    {
	    return data.KInv[level];
    }
    inline float Frame::fx(int level) const
    {
	    return data.fx[level];
    }
    inline float Frame::fy(int level) const
    {
	    return data.fy[level];
    }
    inline float Frame::cx(int level) const
    {
	    return data.cx[level];
    }
    inline float Frame::cy(int level) const
    {
	    return data.cy[level];
    }
    inline float Frame::fxInv(int level) const
    {
	    return data.fxInv[level];
    }
    inline float Frame::fyInv(int level) const
    {
	    return data.fyInv[level];
    }
    inline float Frame::cxInv(int level) const
    {
	    return data.cxInv[level];
    }
    inline float Frame::cyInv(int level) const
    {
	    return data.cyInv[level];
    }

    inline double Frame::timestamp() const
    {
	    return data.timestamp;
    }


    inline float* Frame::image(int level)
    {
	    if (! data.imageValid[level])
		    require(IMAGE, level);
	    return data.image[level];
    }
    inline const Eigen::Vector4f* Frame::gradients(int level)
    {
	    if (! data.gradientsValid[level])
		    require(GRADIENTS, level);
	    return data.gradients[level];
    }
    inline const float* Frame::maxGradients(int level)
    {
	    if (! data.maxGradientsValid[level])
		    require(MAX_GRADIENTS, level);
	    return data.maxGradients[level];
    }
    inline bool Frame::hasIDepthBeenSet() const
    {
	    return data.hasIDepthBeenSet;
    }
    inline const float* Frame::idepth(int level)
    {
	    if (! data.hasIDepthBeenSet)
	    {
		    printfAssert("Frame::idepth(): idepth has not been set yet!");
		    return nullptr;
	    }
	    if (! data.idepthValid[level])
		    require(IDEPTH, level);
	    return data.idepth[level];
    }
    inline const unsigned char* Frame::validity_reAct()
    {
	    if( !data.reActivationDataValid)
		    return 0;
	    return data.validity_reAct;
    }
    inline const float* Frame::idepth_reAct()
    {
	    if( !data.reActivationDataValid)
		    return 0;
	    return data.idepth_reAct;
    }
    inline const float* Frame::idepthVar_reAct()
    {
	    if( !data.reActivationDataValid)
		    return 0;
	    return data.idepthVar_reAct;
    }
    inline const float* Frame::idepthVar(int level)
    {
	    if (! data.hasIDepthBeenSet)
	    {
		    printfAssert("Frame::idepthVar(): idepth has not been set yet!");
		    return nullptr;
	    }
	    if (! data.idepthVarValid[level])
		    require(IDEPTH_VAR, level);
	    return data.idepthVar[level];
    }


    inline bool* Frame::refPixelWasGood()
    {
	    if( data.refPixelWasGood == 0)
	    {
		    boost::unique_lock<boost::mutex> lock2(buildMutex);

		    if(data.refPixelWasGood == 0)
		    {
			    int width = data.width[SE3TRACKING_MIN_LEVEL];
			    int height = data.height[SE3TRACKING_MIN_LEVEL];
			    data.refPixelWasGood = (bool*)FrameMemory::getInstance().getBuffer(sizeof(bool) * width * height);

			    memset(data.refPixelWasGood, 0xFFFFFFFF, sizeof(bool) * (width * height));
		    }
	    }
	    return data.refPixelWasGood;
    }


    inline bool* Frame::refPixelWasGoodNoCreate()
    {
	    return data.refPixelWasGood;
    }

    inline void Frame::clear_refPixelWasGood()
    {
	    FrameMemory::getInstance().returnBuffer(reinterpret_cast<float*>(data.refPixelWasGood));
	    data.refPixelWasGood=0;
    }


}
