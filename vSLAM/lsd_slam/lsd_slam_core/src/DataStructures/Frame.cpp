/*This file is part of LSD-SLAM.
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
*  上一层 逆深度 方差和 取逆得到 本层 逆深度方差 
* 
*/

#include "DataStructures/Frame.h"
#include "DataStructures/FrameMemory.h"
#include "DepthEstimation/DepthMapPixelHypothesis.h"
#include "Tracking/TrackingReference.h"

namespace lsd_slam
{

      int privateFrameAllocCount = 0;
      // 构造函数有两个，主要是最后一个参数给的有所不同，实际上它代表的是两种不同格式的图片数据
      // 构造函数  0~255 的图像  
      Frame::Frame(int id, int width, int height, const Eigen::Matrix3f& K, double timestamp, const unsigned char* image)
      {
	// 类参数变量初始化 主要初始化 图像金字塔的 内参数  内参数逆
	      initialize(id, width, height, K, timestamp);
      // 获取图像内存空间	
	      data.image[0] = FrameMemory::getInstance().getFloatBuffer(data.width[0]*data.height[0]);//float类型存储 指针
	      float* maxPt = data.image[0] + data.width[0]*data.height[0];// 存储区域 float 的个数 对应像素个数
      // 使用指针复制图像的内一个像素
	      for(float* pt = data.image[0]; pt < maxPt; pt++)// 拷贝每一个像素
	      {
		      *pt = *image;// 拷贝每一个像素
		      image++;// 指针指向下一个
	      }
	      data.imageValid[0] = true;// 第0层图像金字塔 已经有图像数据了

	      privateFrameAllocCount++;// 私有帧数量++

	      if(enablePrintDebugInfo && printMemoryDebugInfo)// 调试信息
		      printf("ALLOCATED frame %d, now there are %d\n", this->id(), privateFrameAllocCount);
      }
      // 构造函数  0~1 的图像  
      Frame::Frame(int id, int width, int height, const Eigen::Matrix3f& K, double timestamp, const float* image)
      {
	// 类参数变量初始化 主要初始化 图像金字塔的 内参数  内参数逆
	      initialize(id, width, height, K, timestamp);
      // 获取图像内存空间	
	      data.image[0] = FrameMemory::getInstance().getFloatBuffer(data.width[0]*data.height[0]);
      // 使用memcpy 复制图像内容 两个指针指向的类型一直
	      memcpy(data.image[0], image, data.width[0]*data.height[0] * sizeof(float));//  指针类型相同 使用 memcpy拷贝内容 
	      data.imageValid[0] = true;// 第0层图像金字塔 已经有图像数据了

	      privateFrameAllocCount++;// 私有帧数量++

	      if(enablePrintDebugInfo && printMemoryDebugInfo)// 调试信息
		      printf("ALLOCATED frame %d, now there are %d\n", this->id(), privateFrameAllocCount);
      }
      // 析构函数
      Frame::~Frame()
      {
      // 调试信息
	      if(enablePrintDebugInfo && printMemoryDebugInfo)
		      printf("DELETING frame %d\n", this->id());// 打印信息 删除 帧
      // 回收内存
	      FrameMemory::getInstance().deactivateFrame(this);// 这里只是回收内存

	      if(!pose->isRegisteredToGraph)
		      delete pose;// 还没有  注册放入到地图 就删除位姿
	      else
		      pose->frame = 0;
      // 删除图像金字塔的每一层 图像
	      for (int level = 0; level < PYRAMID_LEVELS; ++ level)// 每一层图像金字塔
	      {
		      FrameMemory::getInstance().returnBuffer(data.image[level]);// 图像
		      FrameMemory::getInstance().returnBuffer(reinterpret_cast<float*>(data.gradients[level]));// 梯度
		      FrameMemory::getInstance().returnBuffer(data.maxGradients[level]);//
		      FrameMemory::getInstance().returnBuffer(data.idepth[level]);// 逆深度 均值
		      FrameMemory::getInstance().returnBuffer(data.idepthVar[level]);// 逆深度方差
	      }
      // 删除 深度 逆深度均值   逆深度方差
	      FrameMemory::getInstance().returnBuffer((float*)data.validity_reAct);
	      FrameMemory::getInstance().returnBuffer(data.idepth_reAct);
	      FrameMemory::getInstance().returnBuffer(data.idepthVar_reAct);
      // 最后再释放permaRef_colorAndVarData和permaRef_posData,这两个参数是位置，颜色和方差的引用，用于重定位，
      // 注意：这个参数只是在initialize中初始化为空指针
	      if(permaRef_colorAndVarData != 0)
		      delete permaRef_colorAndVarData;// 删除 颜色和方差的引用
	      if(permaRef_posData != 0)
		      delete permaRef_posData;// 位置 的 引用 

	      privateFrameAllocCount--;/// 私有帧数量--
	      if(enablePrintDebugInfo && printMemoryDebugInfo)// 调试信息
		      printf("DELETED frame %d, now there are %d\n", this->id(), privateFrameAllocCount);
      }


      void Frame::takeReActivationData(DepthMapPixelHypothesis* depthMap)
      {
// 上锁
	      boost::shared_lock<boost::shared_mutex> lock = getActiveLock();
// 申请内存
	      if(data.validity_reAct == 0)
		      data.validity_reAct = (unsigned char*) FrameMemory::getInstance().getBuffer(data.width[0]*data.height[0]);

	      if(data.idepth_reAct == 0)
		      data.idepth_reAct = FrameMemory::getInstance().getFloatBuffer((data.width[0]*data.height[0]));

	      if(data.idepthVar_reAct == 0)
		      data.idepthVar_reAct = FrameMemory::getInstance().getFloatBuffer((data.width[0]*data.height[0]));

// 赋值
	      float* id_pt = data.idepth_reAct;// 起始位置
	      float* id_pt_max = data.idepth_reAct + (data.width[0]*data.height[0]);// 最大位置
	      float* idv_pt = data.idepthVar_reAct;
	      unsigned char* val_pt = data.validity_reAct;

	      for (; id_pt < id_pt_max; ++ id_pt, ++ idv_pt, ++ val_pt, ++depthMap)
	      {
		      if(depthMap->isValid)// 深度图 有效
		      {
			      *id_pt = depthMap->idepth;// 深度值
			      *idv_pt = depthMap->idepth_var;// 深度值方差
			      *val_pt = depthMap->validity_counter;
		      }
		      else if(depthMap->blacklisted < MIN_BLACKLIST)
		      {
			      *idv_pt = -2;
		      }
		      else
		      {
			      *idv_pt = -1;
		      }
	      }

	      data.reActivationDataValid = true;
      }


// 参考帧
      void Frame::setPermaRef(TrackingReference* reference)
      {
	      assert(reference->frameID == id());
	      reference->makePointCloud(QUICK_KF_CHECK_LVL);

	      permaRef_mutex.lock();// 上锁

	      if(permaRef_colorAndVarData != 0)
		      delete permaRef_colorAndVarData;
	      if(permaRef_posData != 0)
		      delete permaRef_posData;

	      permaRefNumPts = reference->numData[QUICK_KF_CHECK_LVL];
	      permaRef_colorAndVarData = new Eigen::Vector2f[permaRefNumPts];
	      permaRef_posData = new Eigen::Vector3f[permaRefNumPts];

	      memcpy(permaRef_colorAndVarData,
			      reference->colorAndVarData[QUICK_KF_CHECK_LVL],
			      sizeof(Eigen::Vector2f) * permaRefNumPts);

	      memcpy(permaRef_posData,
			      reference->posData[QUICK_KF_CHECK_LVL],
			      sizeof(Eigen::Vector3f) * permaRefNumPts);

	      permaRef_mutex.unlock();
      }
      
// 求解 深度值标准差逆( 深度值方差逆  开根号 )  均值(求和 取平均)
      void Frame::calculateMeanInformation()
      {
	      return;

	      if(numMappablePixels < 0)
		      maxGradients(0);

	      const float* idv = idepthVar(0);// 深度方差 起始位置
	      const float* idv_max = idv + width(0)*height(0);// 最大位置
	      float sum = 0; int goodpx = 0;
	      for(const float* pt=idv; pt < idv_max; pt++)
	      {
		      if(*pt > 0)
		      {
			      sum += sqrtf(1.0f / *pt);// 深度值方差逆  开根号 得到 深度值标准差逆  和 
			      goodpx++;
		      }
	      }

	      meanInformation = sum / goodpx;// 深度值标准差逆  均值
      }
      
////////////////////////////////////////////////////////////////////////////////////////////
// 使用 逆深度图深度值假设值(高斯分布平滑后的数据) 设置 第0层的 逆深度均值和方差  对深度估计值进行设定
      void Frame::setDepth(const DepthMapPixelHypothesis* newDepth)
      {
     // 首先调用了锁，把数据都锁了起来
	      boost::shared_lock<boost::shared_mutex> lock = getActiveLock();
	      boost::unique_lock<boost::mutex> lock2(buildMutex);
   // 申请第0层的 深度均值和方差 的存储空间
	      if(data.idepth[0] == 0)
		      data.idepth[0] = FrameMemory::getInstance().getFloatBuffer(data.width[0]*data.height[0]);
	      if(data.idepthVar[0] == 0)
		      data.idepthVar[0] = FrameMemory::getInstance().getFloatBuffer(data.width[0]*data.height[0]);

	      float* pyrIDepth = data.idepth[0];//  第一个内存地址的指针
	      float* pyrIDepthVar = data.idepthVar[0];
	      float* pyrIDepthMax = pyrIDepth + (data.width[0]*data.height[0]);// 最大的内存地址
	      
	      float sumIdepth=0;// 逆深度均值和
	      int numIdepth=0;// 有 有效逆深度值 的 像素点数量

	      for (; pyrIDepth < pyrIDepthMax; ++ pyrIDepth, ++ pyrIDepthVar, ++ newDepth) //, ++ pyrRefID)
	      {
		      if (newDepth->isValid && newDepth->idepth_smoothed >= -0.05)// 预设值的值符合 一些前提条件
		      {
			      *pyrIDepth = newDepth->idepth_smoothed;// 逆深度均值
			      *pyrIDepthVar = newDepth->idepth_var_smoothed;// 逆深度方差

			      numIdepth++;// 有 有效逆深度值 的 像素点数量
			      sumIdepth += newDepth->idepth_smoothed;// 逆深度均值和
		      }
		      // 预设值不符合规定，设置异常值 -1
		      else
		      {
			      *pyrIDepth = -1;
			      *pyrIDepthVar = -1;
		      }
	      }
	      
	      meanIdepth = sumIdepth / numIdepth;// 逆深度均值
	      numPoints = numIdepth;// 有 有效逆深度值 的 像素点数量


	      data.idepthValid[0] = true;// 第0层逆深度值 已经得到
	      data.idepthVarValid[0] = true;// 第0层逆深度均值 已经得到
	      release(IDEPTH | IDEPTH_VAR, true, true);// 最后调用release,释放第0层以上层的深度估计值
	      data.hasIDepthBeenSet = true;
	      depthHasBeenUpdatedFlag = true;// 更新标志
      }

// 从 深度真实值 设置 第0层的逆深度值和方差
      void Frame::setDepthFromGroundTruth(const float* depth, float cov_scale)
      {
      // 首先调用了锁，把数据都锁了起来	
	      boost::shared_lock<boost::shared_mutex> lock = getActiveLock();
	      const float* pyrMaxGradient = maxGradients(0);
	      boost::unique_lock<boost::mutex> lock2(buildMutex);
     // 申请第0层的 深度均值和方差 的存储空间	      
	      if(data.idepth[0] == 0)
		      data.idepth[0] = FrameMemory::getInstance().getFloatBuffer(data.width[0]*data.height[0]);
	      if(data.idepthVar[0] == 0)
		      data.idepthVar[0] = FrameMemory::getInstance().getFloatBuffer(data.width[0]*data.height[0]);

	      float* pyrIDepth = data.idepth[0];//  第一个内存地址的指针
	      float* pyrIDepthVar = data.idepthVar[0];

	      int width0 = data.width[0];// 深度值图 宽度
	      int height0 = data.height[0];// 高度

	      for(int y=0;y<height0;y++)// 每一行
	      {
		      for(int x=0;x<width0;x++)// 每一列
		      {
			     // 有效位置
			      if (x > 0 && x < width0-1 && y > 0 && y < height0-1 && // pyramidMaxGradient is not valid for the border
					      pyrMaxGradient[x+y*width0] >= MIN_ABS_GRAD_CREATE &&
					      !isnanf(*depth) && *depth > 0)
			      {
				      *pyrIDepth = 1.0f / *depth;// 逆深度值
				      *pyrIDepthVar = VAR_GT_INIT_INITIAL * cov_scale;// 逆深度方差
			      }
			      // 无效位置
			      else
			      {
				      *pyrIDepth = -1;
				      *pyrIDepthVar = -1;
			      }

			      ++ depth;
			      ++ pyrIDepth;
			      ++ pyrIDepthVar;
		      }
	      }
   // 设置标志	      
	      data.idepthValid[0] = true;// 第0层 逆深度值 已经得到
	      data.idepthVarValid[0] = true;
      // 	data.refIDValid[0] = true;
	      // Invalidate higher levels, they need to be updated with the new data
	      release(IDEPTH | IDEPTH_VAR, true, true);// 最后调用release,释放第0层以上层的深度估计值
	      data.hasIDepthBeenSet = true;
      }

////////////////////////////////////////////////
// 这个函数是用来设置变换的，准备 两帧之前的 旋转 平移  变换矩阵 为双目三角测量得到深度做准备
// 第一个参数传入是哪一帧，
// 第二个参数是这一帧的相似变换矩阵   ----> sR, t
// 第三个参数是相机参数  K ，
// 第四个参数是金字塔等级 level
      void Frame::prepareForStereoWith(Frame* other, Sim3 thisToOther, const Eigen::Matrix3f& K, const int level)
      {
	      Sim3 otherToThis = thisToOther.inverse();
// 1.  other 变换到 当前帧    K*T = K*(R +t) = K*R + K*t
	      //otherToThis = data.worldToCam * other->data.camToWorld;
	      K_otherToThis_R = K * otherToThis.rotationMatrix().cast<float>() * otherToThis.scale();// K*R
	      otherToThis_t = otherToThis.translation().cast<float>();
	      K_otherToThis_t = K * otherToThis_t;// K*t
// 2. 当前帧  变换到 other  K*T = K*(R +t) = K*R + K*t
	      thisToOther_t = thisToOther.translation().cast<float>();
	      K_thisToOther_t = K * thisToOther_t;
	      thisToOther_R = thisToOther.rotationMatrix().cast<float>() * thisToOther.scale();
// 3.  other 变换到 当前帧 旋转矩阵的每一行    R逆 = R转置
	      otherToThis_R_row0 = thisToOther_R.col(0);// 第0列 转置 第0行
	      otherToThis_R_row1 = thisToOther_R.col(1);// 第1列 转置 第1行
	      otherToThis_R_row2 = thisToOther_R.col(2);// 第2列 转置 第2行
// 4. 两帧之间的 距离 平方 t * t'
	      distSquared = otherToThis.translation().dot(otherToThis.translation());
//参考帧
	      referenceID = other->id();
	      referenceLevel = level;
      }
      
      

//  是某种请求函数
// 判断需要怎样的数据，如果这个数据没有，就调用相应的构建函数build*
      void Frame::require(int dataFlags, int level)
      {
	      if ((dataFlags & IMAGE) && ! data.imageValid[level])
	      {
		      buildImage(level);// 第几层的图像没有 则创建  递归创建 从最大的图开始 依次 下采样 获取 金字塔图像
	      }
	      if ((dataFlags & GRADIENTS) && ! data.gradientsValid[level])
	      {
		      buildGradients(level);// 第几层的 梯度没有 则创建 
	      }
	      if ((dataFlags & MAX_GRADIENTS) && ! data.maxGradientsValid[level])
	      {
		      buildMaxGradients(level);// 第几层的  最大梯度 没有 则创建 
	      }
	      if (((dataFlags & IDEPTH) && ! data.idepthValid[level])
		      || ((dataFlags & IDEPTH_VAR) && ! data.idepthVarValid[level]))
	      {
		      buildIDepthAndIDepthVar(level);//  第几层的  逆深度   没有 则创建 
	      }
      }
      
// 是某种释放函数
      void Frame::release(int dataFlags, bool pyramidsOnly, bool invalidateOnly)
      {
	      for (int level = (pyramidsOnly ? 1 : 0); level < PYRAMID_LEVELS; ++ level)
	      {
		      if ((dataFlags & IMAGE) && data.imageValid[level])
		      {
			      data.imageValid[level] = false;
			      if(!invalidateOnly)
				      releaseImage(level);// 释放 图像 
		      }
		      if ((dataFlags & GRADIENTS) && data.gradientsValid[level])
		      {
			      data.gradientsValid[level] = false;
			      if(!invalidateOnly)
				      releaseGradients(level);// 释放 梯度图
		      }
		      if ((dataFlags & MAX_GRADIENTS) && data.maxGradientsValid[level])
		      {
			      data.maxGradientsValid[level] = false;
			      if(!invalidateOnly)
				      releaseMaxGradients(level);// 释放最大梯度图
		      }
		      if ((dataFlags & IDEPTH) && data.idepthValid[level])
		      {
			      data.idepthValid[level] = false;
			      if(!invalidateOnly)
				      releaseIDepth(level);// 释放逆深度图
		      }
		      if ((dataFlags & IDEPTH_VAR) && data.idepthVarValid[level])
		      {
			      data.idepthVarValid[level] = false;
			      if(!invalidateOnly)
				      releaseIDepthVar(level);// 释放逆深度图 值
		      }
	      }
      }
      
      
 // 是最小化储存函数
 // 就是释放一些内存
      bool Frame::minimizeInMemory()
      {
	      if(activeMutex.timed_lock(boost::posix_time::milliseconds(10)))
	      {
		      buildMutex.lock();
		      // 打印信息
		      if(enablePrintDebugInfo && printMemoryDebugInfo)
			      printf("minimizing frame %d\n",id());

		      release(IMAGE | IDEPTH | IDEPTH_VAR, true, false);
		      release(GRADIENTS | MAX_GRADIENTS, false, false);

		      clear_refPixelWasGood();

		      buildMutex.unlock();
		      activeMutex.unlock();
		      return true;
	      }
	      return false;
      }

// 类参数变量初始化 主要初始化 图像金字塔的 内参数  内参数逆
      void Frame::initialize(int id, int width, int height, const Eigen::Matrix3f& K, double timestamp)
      {
	      data.id = id;// id
	      
	      pose = new FramePoseStruct(this);// 帧位姿
	    // 相机内参
	      data.K[0] = K;
	      data.fx[0] = K(0,0);
	      data.fy[0] = K(1,1);
	      data.cx[0] = K(0,2);
	      data.cy[0] = K(1,2);
	    // 相机内参数逆
	      data.KInv[0] = K.inverse();// 逆矩阵
	      data.fxInv[0] = data.KInv[0](0,0);
	      data.fyInv[0] = data.KInv[0](1,1);
	      data.cxInv[0] = data.KInv[0](0,2);
	      data.cyInv[0] = data.KInv[0](1,2);
	      
	      data.timestamp = timestamp;// 时间戳

	      data.hasIDepthBeenSet = false;
	      depthHasBeenUpdatedFlag = false;
	      
	      referenceID = -1;
	      referenceLevel = -1;
	      
	      numMappablePixels = -1;
	    /// 初始化金字塔
	      for (int level = 0; level < PYRAMID_LEVELS; ++ level)
	      {
		      data.width[level] = width >> level;// 右移位 相等于 除以2
		      data.height[level] = height >> level;

		      data.imageValid[level] = false;
		      data.gradientsValid[level] = false;
		      data.maxGradientsValid[level] = false;
		      data.idepthValid[level] = false;
		      data.idepthVarValid[level] = false;

		      data.image[level] = 0;
		      data.gradients[level] = 0;
		      data.maxGradients[level] = 0;
		      data.idepth[level] = 0;
		      data.idepthVar[level] = 0;
		      data.reActivationDataValid = false;

      // 		data.refIDValid[level] = false;
		      // 初始化相机金字塔
		      if (level > 0)
		      {
			      data.fx[level] = data.fx[level-1] * 0.5;// 相应的 内参数  是 上一层的一般  大小变小了 变为原来的1/2
			      data.fy[level] = data.fy[level-1] * 0.5;
			      data.cx[level] = (data.cx[0] + 0.5) / ((int)1<<level) - 0.5;// 也是 变为 1/2
			      data.cy[level] = (data.cy[0] + 0.5) / ((int)1<<level) - 0.5;

			      data.K[level]  << data.fx[level], 0.0, data.cx[level], 0.0, data.fy[level], data.cy[level], 0.0, 0.0, 1.0;	// synthetic
			      data.KInv[level] = (data.K[level]).inverse();

			      data.fxInv[level] = data.KInv[level](0,0);
			      data.fyInv[level] = data.KInv[level](1,1);
			      data.cxInv[level] = data.KInv[level](0,2);
			      data.cyInv[level] = data.KInv[level](1,2);
		      }
	      }

	      data.validity_reAct = 0;
	      data.idepthVar_reAct = 0;
	      data.idepth_reAct = 0;

	      data.refPixelWasGood = 0;

	      permaRefNumPts = 0;
	      permaRef_colorAndVarData = 0;
	      permaRef_posData = 0;

	      meanIdepth = 1;
	      numPoints = 0;

	      numFramesTrackedOnThis = numMappedOnThis = numMappedOnThisTotal = 0;

	      idxInKeyframes = -1;

	      edgeErrorSum = edgesNum = 1;

	      lastConstraintTrackedCamToWorld = Sim3();

	      isActive = false;
      }

      void Frame::setDepth_Allocate()
      {
	      return;
      }

      // 创建对应层图像金字塔的  图像
      // 递归构建底层金字塔，因为上层金字塔是以底层为基础的 
      // 分配内存 从上一层  隔行隔列下采样  方案是下层金字塔的四个像素的值的平均值合并成一个 
      // 建立每一层 图像金字塔
      void Frame::buildImage(int level)
      {
	      if (level == 0)
	      {
		      printf("Frame::buildImage(0): Loading image from disk is not implemented yet! No-op.\n");
		      return;
	      }
	      
	      require(IMAGE, level - 1);// 首先是递归构建底层金字塔，因为上层金字塔是以底层为基础的
	      // 递归到底层后，调用buildMutex互斥锁，
	      boost::unique_lock<boost::mutex> lock2(buildMutex);// 创建线程锁 上锁

	      if(data.imageValid[level])
		      return;
	    // 打印调试信息
	      if(enablePrintDebugInfo && printFrameBuildDebugInfo)
		      printf("CREATE Image lvl %d for frame %d\n", level, id());
	      // 当前层的 图像宽 高
	      int width = data.width[level - 1];// 0,1,2,3,4
	      int height = data.height[level - 1];
	      const float* source = data.image[level - 1];// 上一层的图像
	      
      // 之后才是判断这个等级的金字塔是否已经构建，然后向内存管理的对象申请内存，之后构建整个金字塔
	      if (data.image[level] == 0)// 指针为空 当前层 未分配存储空间
		    // 分配 存储空间
		      data.image[level] = FrameMemory::getInstance().getFloatBuffer(data.width[level] * data.height[level]);
	      float* dest = data.image[level];// 目标地址

      #if defined(ENABLE_SSE)
	      // I assume all all subsampled width's are a multiple of 8.
	      // if this is not the case, this still works except for the last * pixel, which will produce a segfault.
	      // in that case, reduce this loop and calculate the last 0-3 dest pixels by hand....
	      if (width % 8 == 0)
	      {
		      __m128 p025 = _mm_setr_ps(0.25f,0.25f,0.25f,0.25f);

		      const float* maxY = source+width*height;
		      for(const float* y = source; y < maxY; y+=width*2)
		      {
			      const float* maxX = y+width;
			      for(const float* x=y; x < maxX; x += 8)
			      {
				      // i am calculating four dest pixels at a time.

				      __m128 top_left = _mm_load_ps((float*)x);
				      __m128 bot_left = _mm_load_ps((float*)x+width);
				      __m128 left = _mm_add_ps(top_left,bot_left);

				      __m128 top_right = _mm_load_ps((float*)x+4);
				      __m128 bot_right = _mm_load_ps((float*)x+width+4);
				      __m128 right = _mm_add_ps(top_right,bot_right);

				      __m128 sumA = _mm_shuffle_ps(left,right, _MM_SHUFFLE(2,0,2,0));
				      __m128 sumB = _mm_shuffle_ps(left,right, _MM_SHUFFLE(3,1,3,1));

				      __m128 sum = _mm_add_ps(sumA,sumB);
				      sum = _mm_mul_ps(sum,p025);

				      _mm_store_ps(dest, sum);
				      dest += 4;
			      }
		      }

		      data.imageValid[level] = true;
		      return;
	      }
      #elif defined(ENABLE_NEON)
	      // I assume all all subsampled width's are a multiple of 8.
	      // if this is not the case, this still works except for the last * pixel, which will produce a segfault.
	      // in that case, reduce this loop and calculate the last 0-3 dest pixels by hand....
	      if (width % 8 == 0)
	      {
		      static const float p025[] = {0.25, 0.25, 0.25, 0.25};
		      int width_iteration_count = width / 8;
		      int height_iteration_count = height / 2;
		      const float* cur_px = source;
		      const float* next_row_px = source + width;
		      
		      __asm__ __volatile__
		      (
			      "vldmia %[p025], {q10}                        \n\t" // p025(q10)
			      
			      ".height_loop:                                \n\t"
			      
				      "mov r5, %[width_iteration_count]             \n\t" // store width_iteration_count
				      ".width_loop:                                 \n\t"
				      
					      "vldmia   %[cur_px]!, {q0-q1}             \n\t" // top_left(q0), top_right(q1)
					      "vldmia   %[next_row_px]!, {q2-q3}        \n\t" // bottom_left(q2), bottom_right(q3)
		      
					      "vadd.f32 q0, q0, q2                      \n\t" // left(q0)
					      "vadd.f32 q1, q1, q3                      \n\t" // right(q1)
		      
					      "vpadd.f32 d0, d0, d1                     \n\t" // pairwise add into sum(q0)
					      "vpadd.f32 d1, d2, d3                     \n\t"
					      "vmul.f32 q0, q0, q10                     \n\t" // multiply with 0.25 to get average
					      
					      "vstmia %[dest]!, {q0}                    \n\t"
				      
				      "subs     %[width_iteration_count], %[width_iteration_count], #1 \n\t"
				      "bne      .width_loop                     \n\t"
				      "mov      %[width_iteration_count], r5    \n\t" // restore width_iteration_count
				      
				      // Advance one more line
				      "add      %[cur_px], %[cur_px], %[rowSize]    \n\t"
				      "add      %[next_row_px], %[next_row_px], %[rowSize] \n\t"
			      
			      "subs     %[height_iteration_count], %[height_iteration_count], #1 \n\t"
			      "bne      .height_loop                       \n\t"

			      : /* outputs */ [cur_px]"+&r"(cur_px),
							      [next_row_px]"+&r"(next_row_px),
							      [width_iteration_count]"+&r"(width_iteration_count),
							      [height_iteration_count]"+&r"(height_iteration_count),
							      [dest]"+&r"(dest)
			      : /* inputs  */ [p025]"r"(p025),
							      [rowSize]"r"(width * sizeof(float))
			      : /* clobber */ "memory", "cc", "r5",
							      "q0", "q1", "q2", "q3", "q10"
		      );

		      data.imageValid[level] = true;
		      return;
	      }
      #endif

	      int wh = width*height;// 当前层 像素总数
	      const float* s;
	      for(int y=0; y<wh; y += width*2)
	      {
		      for(int x=0; x<width; x+= 2)// 从上一层 隔行隔列下采样
		      {
			      s = source + x + y;// 上一层 像素对应位置
			      *dest = (s[0] +
					      s[1] +
					      s[width] +
					      s[1+width]) * 0.25f;// 四个像素的值的平均值合并成一个
			      dest++;
		      }
	      }

	      data.imageValid[level] = true;// 该层 图像已经构建
      }

      // 释放图像
      void Frame::releaseImage(int level)
      {
	      if (level == 0)
	      {
		      printf("Frame::releaseImage(0): Storing image on disk is not supported yet! No-op.\n");
		      return;
	      }
	      FrameMemory::getInstance().returnBuffer(data.image[level]);
	      data.image[level] = 0;// 指针为0
      }

      //  根据 图像金字塔 构建梯度金字塔  xy方向梯度  是 中心差分
      void Frame::buildGradients(int level)
      {
      // 梯度图像需要 同一层的 图像
	      require(IMAGE, level);
	      boost::unique_lock<boost::mutex> lock2(buildMutex);

	      if(data.gradientsValid[level])// 空指针 为分配内存 返回
		      return;
      // 打印调试信息
	      if(enablePrintDebugInfo && printFrameBuildDebugInfo)
		      printf("CREATE Gradients lvl %d for frame %d\n", level, id());
      // 当前层图像的宽度和高度 
	      int width = data.width[level];
	      int height = data.height[level];
      // 存储指针为空，则 申请内存空间
	      if(data.gradients[level] == 0)
		    //  x 方向梯度  y方向梯度  当前点像素值 第四维度没有存储数据  gradxyii_pt
		      data.gradients[level] = (Eigen::Vector4f*)FrameMemory::getInstance().getBuffer(sizeof(Eigen::Vector4f) * width * height);
	      
	      const float* img_pt = data.image[level] + width;// 第二行开始 的 图像值   应为要计算y方向梯度
	      const float* img_pt_max = data.image[level] + width*(height-1);// 图像 最大的指针地址
	      Eigen::Vector4f* gradxyii_pt = data.gradients[level] + width;// 梯度值对应的 空间指针
	      
	      // in each iteration i need -1,0,p1,mw,pw
	      float val_m1 = *(img_pt-1);// 是左右两个像素点的梯度
	      float val_00 = *img_pt;//  当前中心点
	      float val_p1;// 是中心差分
      //            *(img_pt-width)
      //  val_m1  *(img_pt)   val_p1
      //           *(img_pt+width)
      //   (val_p1 - val_m1)/2  = x 方向梯度
      // 0.5f*(*(img_pt+width) - *(img_pt-width)) = y方向梯度
      // val_00 = *(img_pt)   当前 点像素值
      // 第四维度 没有存储数据    gradxyii_pt  Eigen::Vector4f
	      for(; img_pt < img_pt_max; img_pt++, gradxyii_pt++)
	      {
		      val_p1 = *(img_pt+1);

		      *((float*)gradxyii_pt) = 0.5f*(val_p1 - val_m1);// 是左右两个像素点的梯度 x方向梯度
		      *(((float*)gradxyii_pt)+1) = 0.5f*(*(img_pt+width) - *(img_pt-width));// +/- 一行的长度就得到 上下的坐标 y方向梯度
		      *(((float*)gradxyii_pt)+2) = val_00;// 像素值
		      val_m1 = val_00;// 迭代
		      val_00 = val_p1;
	      }

	      data.gradientsValid[level] = true;
      }
      // 释放 梯度 金字塔
      void Frame::releaseGradients(int level)
      {
	      FrameMemory::getInstance().returnBuffer(reinterpret_cast<float*>(data.gradients[level]));
	      data.gradients[level] = 0;
      }
      
      
// 创建 梯度图内 临近四点中梯度最大值 的 最大值梯度 图 ， 并记录梯度值较大的可以映射 成 地图点的数量
// 在梯度图中  求的  上中下 三个梯度值中的最大值，形成临时梯度最大值图
// 在临时梯度最大值图 中求 的  左中右 三个梯度值中的最大值，形成最后的 最大梯度值地图
// 并记录 最大梯度大小超过阈值的点 可以映射成地图点  
      void Frame::buildMaxGradients(int level)
      {
	// 需要同一层级的 梯度图
	      require(GRADIENTS, level);
	      boost::unique_lock<boost::mutex> lock2(buildMutex);
       // 已经 得到过了，就不要计算了
	      if(data.maxGradientsValid[level]) return;
       // 打印调试信息
	      if(enablePrintDebugInfo && printFrameBuildDebugInfo)
		      printf("CREATE AbsGrad lvl %d for frame %d\n", level, id());
      // 当前层级的 宽度和高度
	      int width = data.width[level];
	      int height = data.height[level];
      // 未申请内存则申请内存
	      if (data.maxGradients[level] == 0)
		      data.maxGradients[level] = FrameMemory::getInstance().getFloatBuffer(width * height);
	      // 临时内存地址
	      float* maxGradTemp = FrameMemory::getInstance().getFloatBuffer(width * height);


    // 1. 秋去合成梯度大小 sqrt(dx^2 + dy^2)   write abs gradients in real data.
	      Eigen::Vector4f* gradxyii_pt = data.gradients[level] + width;// 梯度 第二行开始的 地址  ， x梯度 y梯度 像素值 空
	      float* maxgrad_pt = data.maxGradients[level] + width;// 对应最大梯度值  存储起始地址
	      float* maxgrad_pt_max = data.maxGradients[level] + width*(height-1);// 对应最大梯度值  最大存储地址

	      for(; maxgrad_pt < maxgrad_pt_max; maxgrad_pt++, gradxyii_pt++)
	      {
		      float dx = *((float*)gradxyii_pt);// x 方向梯度
		      float dy = *(1+(float*)gradxyii_pt);// y方向梯度
		      *maxgrad_pt = sqrtf(dx*dx + dy*dy);// 和成梯度
	      }

// 2. 求每个梯度值上中下三个位置中的最大值形成的 临时最大梯度图
        // smear up/down direction into temp buffer
	      maxgrad_pt = data.maxGradients[level] + width+1;//第二行 第二个 开始  位置
	      maxgrad_pt_max = data.maxGradients[level] + width*(height-1)-1;// 最大位置
	      float* maxgrad_t_pt = maxGradTemp + width+1;// 对应位置的 临时变量地址
	      for(;maxgrad_pt<maxgrad_pt_max; maxgrad_pt++, maxgrad_t_pt++)
	      {
		      float g1 = maxgrad_pt[-width];// 上方位置的 合成梯度
		      float g2 = maxgrad_pt[0];// 当前点 合成梯度 
		      if(g1 < g2) g1 = g2;// 保留g1 和 g2 中的最大值
		      float g3 = maxgrad_pt[width];// 下方位置的 合成梯度
		      if(g1 < g3)
			      *maxgrad_t_pt = g3; // 临时最大值
		      else
			      *maxgrad_t_pt = g1;// 临时最大值
	      }

	      float numMappablePixels = 0;
// 3. smear left/right direction into real data
	      maxgrad_pt = data.maxGradients[level] + width+1;//第二行 第二个 开始  位置
	      maxgrad_pt_max = data.maxGradients[level] + width*(height-1)-1;// 最大位置
	      maxgrad_t_pt = maxGradTemp + width+1;// 对应位置的 临时变量地址
	      for(;maxgrad_pt<maxgrad_pt_max; maxgrad_pt++, maxgrad_t_pt++)
	      {
		      float g1 = maxgrad_t_pt[-1];// 左边
		      float g2 = maxgrad_t_pt[0];// 中间
		      if(g1 < g2) g1 = g2;// 保留g1 和 g2 中的最大值
		      float g3 = maxgrad_t_pt[1];// 右边
		      if(g1 < g3)
		      {
			      *maxgrad_pt = g3;// 最大值
			      if(g3 >= MIN_ABS_GRAD_CREATE)
				      numMappablePixels++;// 梯度大小超过阈值的 计数
		      }
		      else
		      {
			      *maxgrad_pt = g1;
			      if(g1 >= MIN_ABS_GRAD_CREATE)
				      numMappablePixels++;// 梯度大小超过阈值的 计数
		      }
	      }

	      if(level==0)
		      this->numMappablePixels = numMappablePixels;// 梯度值较大的点可以 求的 地图点

	      FrameMemory::getInstance().returnBuffer(maxGradTemp);// 删除临时内存

	      data.maxGradientsValid[level] = true;// 得到周围四点 最大梯度值 地图
      }
// 深度 最大梯度值地图
      void Frame::releaseMaxGradients(int level)
      {
	      FrameMemory::getInstance().returnBuffer(data.maxGradients[level]);
	      data.maxGradients[level] = 0;
      }

// 根据最初逆深度均值图 构建  逆深度均值图 方差图(高斯分布)  金字塔
//    current   -----> 右边一个
//      下边         下右边      上一层四个位置处的和 
//  上一层 逆方差和 / 上一层 逆深度均值 (四个位置处) 和 得到深度信息 再 取逆得到 逆深度均值
//  上一层 逆深度 方差和 取逆得到 本层 逆深度方差 
      void Frame::buildIDepthAndIDepthVar(int level)
      {
	      if (! data.hasIDepthBeenSet)
	      {
		      printfAssert("Frame::buildIDepthAndIDepthVar(): idepth has not been set yet!\n");
		      return;
	      }
	      if (level == 0)
	      {
		      printf("Frame::buildIDepthAndIDepthVar(0): Loading depth from disk is not implemented yet! No-op.\n");
		      return;
	      }
      // 递归构建 从最开始的 逆深度图开始构建
	      require(IDEPTH, level - 1);// 需要上一层的逆深度图
	      boost::unique_lock<boost::mutex> lock2(buildMutex);
	      
	      if(data.idepthValid[level] && data.idepthVarValid[level])// 该层 逆深度均值 和方差 已经创建 就返回
		      return;
      // 打印调试信息
	      if(enablePrintDebugInfo && printFrameBuildDebugInfo)
		      printf("CREATE IDepth lvl %d for frame %d\n", level, id());
      // 当前层的 宽度和高度 
	      int width = data.width[level];
	      int height = data.height[level];
      // 逆深度均值和 逆深度方差图 内存是否分配
      //  若未分配内存，则分配内存
	      if (data.idepth[level] == 0)
		      data.idepth[level] = FrameMemory::getInstance().getFloatBuffer(width * height);
	      if (data.idepthVar[level] == 0)
		      data.idepthVar[level] = FrameMemory::getInstance().getFloatBuffer(width * height);
      // 上一层 图像宽度
	      int sw = data.width[level - 1];

	      const float* idepthSource = data.idepth[level - 1];// 上一层 逆深度均值  指针
	      const float* idepthVarSource = data.idepthVar[level - 1];// 上一层 逆深度方差 指针
	      float* idepthDest = data.idepth[level];// 当前层 逆深度均值 指针
	      float* idepthVarDest = data.idepthVar[level];// 当前层 逆深度方差 指针
      //
      //    current   -----> 右边一个
      //      下边         下右边      上一层四个位置处的和 
      //
	      for(int y=0;y<height;y++)// 每一行
	      {
		      for(int x=0;x<width;x++)// 每一列
		      {
			      int idx = 2*(x+y*sw);// 在上一层中的偏移量×2  因为上一层的尺寸是当前层的 2倍
			      int idxDest = (x+y*width);// 当前层的 偏移量

			      float idepthSumsSum = 0;// 逆深度均值 之和
			      float ivarSumsSum = 0;// 逆深度方差 的逆之 和
			      int num=0;

			      // build sums
			      float ivar;// 逆方差
			      float var = idepthVarSource[idx];//  上一层 逆深度方差
			      if(var > 0)
			      {
				      ivar = 1.0f / var;//  逆方差
				      ivarSumsSum += ivar;// 逆方差 之和
				      idepthSumsSum += ivar * idepthSource[idx];// 加权 逆深度之和 以逆方差为权重
				      num++;
			      }

			      var = idepthVarSource[idx+1];// 右边一个
			      if(var > 0)
			      {
				      ivar = 1.0f / var;
				      ivarSumsSum += ivar;
				      idepthSumsSum += ivar * idepthSource[idx+1];
				      num++;
			      }

			      var = idepthVarSource[idx+sw];// 下边的一个
			      if(var > 0)
			      {
				      ivar = 1.0f / var;
				      ivarSumsSum += ivar;
				      idepthSumsSum += ivar * idepthSource[idx+sw];
				      num++;
			      }

			      var = idepthVarSource[idx+sw+1];// 下右边的一个
			      if(var > 0)
			      {
				      ivar = 1.0f / var;
				      ivarSumsSum += ivar;
				      idepthSumsSum += ivar * idepthSource[idx+sw+1];
				      num++;
			      }
			      
			      if(num > 0)
			      {
				      float depth = ivarSumsSum / idepthSumsSum;// 上一层 逆方差和 / 上一层 逆深度和 得到深度信息
				      idepthDest[idxDest] = 1.0f / depth;// 深度取逆 得到 逆深度
				      idepthVarDest[idxDest] = num / ivarSumsSum;// 上一层 逆深度 方差和 取逆得到 本层 逆深度方法 
			      }
			      else
			      {
				      idepthDest[idxDest] = -1;
				      idepthVarDest[idxDest] = -1;
			      }
		      }
	      }

	      data.idepthValid[level] = true;
	      data.idepthVarValid[level] = true;
      }
// 释放 逆深度均值图 层
      void Frame::releaseIDepth(int level)
      {
	      if (level == 0)
	      {
		      printf("Frame::releaseIDepth(0): Storing depth on disk is not supported yet! No-op.\n");
		      return;
	      }
	      
	      FrameMemory::getInstance().returnBuffer(data.idepth[level]);
	      data.idepth[level] = 0;
      }

// 释放逆深度 方差 图 层
      void Frame::releaseIDepthVar(int level)
      {
	      if (level == 0)
	      {
		      printf("Frame::releaseIDepthVar(0): Storing depth variance on disk is not supported yet! No-op.\n");
		      return;
	      }
	      FrameMemory::getInstance().returnBuffer(data.idepthVar[level]);
	      data.idepthVar[level] = 0;
      }

      void Frame::printfAssert(const char* message) const
      {
	      assert(!message);
	      printf("%s\n", message);
      }
 }
