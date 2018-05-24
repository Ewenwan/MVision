/**
* This file is part of LSD-SLAM.
* 跟踪的第一步就是 为当前帧 旋转 跟踪的参考帧
*  包含 每一层金字塔上的：
* 1.   关键点位置坐标                   posData[i]           (x,y,z)
* 2.   关键点像素梯度信息               gradData[i]          (dx, dy)
* 3.   关键点像素值 和 逆深度方差信息    colorAndVarData[i]   (I, Var)
* 4.   关键点位置对应的 灰度像素点       pointPosInXYGrid[i]  x + y*width
*       上面四个都是指针数组
* 5.   产生的 物理世界中的点的数量       numData[i]
* 
* 1. 帧坐标系下的关键点　3d 位置坐标 的产生  由深度值和像素坐标
*       P*T*K =  (u,v,1)    P 世界坐标系下的点坐标
*       Q*I*K =  (u,v,1)    Q 当前坐标系下的点坐标
*       Q  =  (X/Z，Y/Z，1)  这里Z就相当于 当前相机坐标系下 点的Z轴方向的深度值D
* 
*        (X，Y，Z) = D  *  (u,v,1)*K逆 =1/(1/D)* (u*fx_inv + cx_inv, v+fy_inv+cy_inv, 1)
* 
*       *posDataPT = (1.0f / pyrIdepthSource[idx]) * Eigen::Vector3f(fxInvLevel*x+cxInvLevel,fyInvLevel*y+cyInvLevel,1);
* 
* 2. 关键点像素梯度信息   的产生 
*     在帧类中有计算直接取来就行了 
*     *gradDataPT = pyrGradSource[idx].head<2>();
* 
* 3.  关键点像素值 和 逆深度方差信息
*     分别在 帧的 图像金字塔 和 逆深度方差金字塔中已经存在，直接取过来
*     *colorAndVarDataPT = Eigen::Vector2f(pyrColorSource[idx], pyrIdepthVarSource[idx]);
* 4. 关键点位置对应的 灰度像素点   直接就是像素所在的位置编号  x + y*width
* 
* 5. 产生的 物理世界中的点的数量
*    首尾指针位置之差就是  三维点 的数量
*    numData[level] = posDataPT - posData[level]; 
*/

#include "Tracking/TrackingReference.h"
#include "DataStructures/Frame.h"// 帧类 图像金字塔 梯度金字塔 逆深度金字塔等
#include "DepthEstimation/DepthMapPixelHypothesis.h"// 逆深度初始化  高斯分布均值初始化 等
#include "GlobalMapping/KeyFrameGraph.h"// 全局地图
#include "util/globalFuncs.h"
#include "IOWrapper/ImageDisplay.h"// 图像显示

namespace lsd_slam
{
      // 类初始化函数
      TrackingReference::TrackingReference()
      {
	      frameID=-1;
	      keyframe = 0;
	      wh_allocated = 0;
	      for (int level = 0; level < PYRAMID_LEVELS; ++ level)
	      {
		      posData[level] = nullptr;
		      gradData[level] = nullptr;
		      colorAndVarData[level] = nullptr;
		      pointPosInXYGrid[level] = nullptr;
		      numData[level] = 0;
	      }
      }

      // 清除帧上所有的 每一金字塔层级对应的 坐标点 梯度 像素值 方差等信息 对应的内存
      void TrackingReference::releaseAll()
      {
	      for (int level = 0; level < PYRAMID_LEVELS; ++ level)// 遍历每一个 金字塔层级
	      {
		      if(posData[level] != nullptr) delete[] posData[level];// delete[]  删除指针数组 位置数据
		      if(gradData[level] != nullptr) delete[] gradData[level];// 梯度数据
		      if(colorAndVarData[level] != nullptr) delete[] colorAndVarData[level];// 颜色和方差数据
		      if(pointPosInXYGrid[level] != nullptr)//位置对应的网格数据
			      Eigen::internal::aligned_free((void*)pointPosInXYGrid[level]);
		      numData[level] = 0;// 清理 计数
	      }
	      wh_allocated = 0;// 第 0 层金字塔 像素总数 这里全部清理了 所以为0 
      }
      // 清理 点云 计数
      void TrackingReference::clearAll()
      {
	      for (int level = 0; level < PYRAMID_LEVELS; ++ level)
		      numData[level] = 0;
      }

      //  析构函数
      TrackingReference::~TrackingReference()
      {
	      boost::unique_lock<boost::mutex> lock(accessMutex);
	      invalidate();//关键帧内存锁 解锁
	      releaseAll();// 清除帧上所有的 每一金字塔层级对应的 坐标点 梯度 像素值 方差等信息 对应的内存
      }

      // 导入 帧 更新帧的资源配置的记录
      void TrackingReference::importFrame(Frame* sourceKF)
      {
	// 上锁   在这个时候 只有本程序能够使用 相应的内存资源
	      boost::unique_lock<boost::mutex> lock(accessMutex);
	// 获取锁
	      keyframeLock = sourceKF->getActiveLock();
	      keyframe = sourceKF;// 帧
	      frameID=keyframe->id();// 身份证 编号
      // 判断是否需要重置
      // 如果宽度和高度的乘积和先前的高度与宽度的乘积一样大，那么就不用重新配置了
	      // reset allocation if dimensions differ (shouldnt happen usually)
	      if(sourceKF->width(0) * sourceKF->height(0) != wh_allocated)
	      {
		      releaseAll();// 如果不同，那么就释放先前分配的资源
		    // 然后把资源配置的记录 记录成当前的配置数
		      wh_allocated = sourceKF->width(0) * sourceKF->height(0);
	      }
	      clearAll();// 最后再清空 点云的记录数据numData
      // 内存解锁
	      lock.unlock();
      }


      // 关键帧内存锁 解锁
      void TrackingReference::invalidate()
      {
	      if(keyframe != 0)
		      keyframeLock.unlock();// 解锁
	      keyframe = 0;
      }
      
/* 1. 帧坐标系下的关键点　3d 位置坐标 的产生  由深度值和像素坐标
*       P*T*K =  (u,v,1)    P 世界坐标系下的点坐标
*       Q*I*K =  (u,v,1)    Q 当前坐标系下的点坐标
*       Q  =  (X/Z，Y/Z，1)  这里Z就相当于 当前相机坐标系下 点的Z轴方向的深度值D
* 
*        (X，Y，Z) = D  *  (u,v,1)*K逆 =1/(1/D)* (u*fx_inv + cx_inv, v+fy_inv+cy_inv, 1)
* 
*       *posDataPT = (1.0f / pyrIdepthSource[idx]) * Eigen::Vector3f(fxInvLevel*x+cxInvLevel,fyInvLevel*y+cyInvLevel,1);
*/ 
      void TrackingReference::makePointCloud(int level)
      {
	// 指针 不为空
	      assert(keyframe != 0);
	// 内存上锁
	      boost::unique_lock<boost::mutex> lock(accessMutex);
      // 判断当前层的点云是否已经生成过
	      if(numData[level] > 0)
		      return;	// already exists.
      // 当前层 的大小
	      int w = keyframe->width(level);// 
	      int h = keyframe->height(level);
      // 当前层的 相机内参数
	      float fxInvLevel = keyframe->fxInv(level);
	      float fyInvLevel = keyframe->fyInv(level);
	      float cxInvLevel = keyframe->cxInv(level);
	      float cyInvLevel = keyframe->cyInv(level);
      // 逆深度 方差 像素值 梯度
	      const float* pyrIdepthSource = keyframe->idepth(level);// 逆深度
	      const float* pyrIdepthVarSource = keyframe->idepthVar(level);// 逆深度方差
	      const float* pyrColorSource = keyframe->image(level);// 像素 灰度值
	      const Eigen::Vector4f* pyrGradSource = keyframe->gradients(level);// 梯度 x方向梯度 y反向梯度等
      //  申请内存 位置点(x,y,z)  网格点  梯度  颜色和方差 信息申请内存
	      if(posData[level] == nullptr) posData[level] = new Eigen::Vector3f[w*h];// 位置点(x,y,z) 
	      if(pointPosInXYGrid[level] == nullptr)// 关键点位置对应的网格点    x + y*width
		      pointPosInXYGrid[level] = (int*)Eigen::internal::aligned_malloc(w*h*sizeof(int));;
	      if(gradData[level] == nullptr) gradData[level] = new Eigen::Vector2f[w*h];// 梯度信息    (dx, dy)
	      if(colorAndVarData[level] == nullptr) colorAndVarData[level] = new Eigen::Vector2f[w*h];// 关键点像素值 和 方差信息  (I, Var)
      // 获取起始指针 
	      Eigen::Vector3f* posDataPT = posData[level];// 位置
	      int* idxPT = pointPosInXYGrid[level];// 关键点位置对应的 灰度像素点 位置编号
	      Eigen::Vector2f* gradDataPT = gradData[level];// 梯度信息 
	      Eigen::Vector2f* colorAndVarDataPT = colorAndVarData[level];// 关键点像素值 和 方差信息

	      for(int x=1; x<w-1; x++)// 每一列
		      for(int y=1; y<h-1; y++)// 每一行
		      {
			      int idx = x + y*w;// 像素的id编号
		    // 跳过逆深度信息 为0的点 或者方差小于0的点
			      if(pyrIdepthVarSource[idx] <= 0 || pyrIdepthSource[idx] == 0) continue;
		    // 像素位置* k逆*深度 得到 当前像极坐标系下的 3d点坐标
			      *posDataPT = (1.0f / pyrIdepthSource[idx]) * Eigen::Vector3f(fxInvLevel*x+cxInvLevel,fyInvLevel*y+cyInvLevel,1);
		    // 图像中点 的x方向梯度 和y方向梯度，在帧类中有计算直接取来就行了 
			      *gradDataPT = pyrGradSource[idx].head<2>();
		    // 分别在 帧的 图像金字塔 和 逆深度方差金字塔中已经存在，直接取过来	
			      *colorAndVarDataPT = Eigen::Vector2f(pyrColorSource[idx], pyrIdepthVarSource[idx]);
		    // 就是像素点的编号  起始点为1 
			      *idxPT = idx;
		    // 指针++  移动到下一个存储位置
			      posDataPT++;
			      gradDataPT++;
			      colorAndVarDataPT++;
			      idxPT++;
		      }

	      numData[level] = posDataPT - posData[level];// 首尾指针位置之差就是  三维点 的数量
      }

 }
