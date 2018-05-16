/**
* This file is part of LSD-SLAM.
* 内存管理 
* 最科学的欣赏源码方式，必然是先看内存管理，
* 即进入文件夹下的第二个文件FrameMemory.h
* 这个文件只有一个类FrameMemory,接口也不多
* 申请空间
* 释放删除空间
* 关键帧添加删除剪枝
*/

#pragma once// 只编译一次
#include <unordered_map>//无序集合map  哈希函数实现
#include <vector>// 向量 动态内存管理的 数组
#include <boost/thread/mutex.hpp>// 多线程 thread mutex
#include <deque>// 双端队列
#include <list>//列表 
#include <boost/thread/shared_mutex.hpp>// 共享 多线程
#include <Eigen/Core> //For EIGEN MACRO 矩阵运算

namespace lsd_slam
{

/** Singleton class for re-using buffers in the Frame class. */
class Frame;
class FrameMemory// 这个类偷偷地管理了每一帧的所有内存
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	/** Returns the global instance. Creates it when the method is first called. */
	// 注意到构造函数被私有化，显然这是初始化对象的函数
       // 返回帧内存对象实例  注意为  函数返回值为 引用
	static FrameMemory& getInstance();
	
	// 获取指定大小的内存空间============
	/** Allocates or fetches a buffer with length: size * sizeof(float).
	  * Corresponds to "buffer = new float[size]". */
	float* getFloatBuffer(unsigned int size);
	/** Allocates or fetches a buffer with length: size * sizeof(float).
	  * Corresponds to "buffer = new float[size]". */
	void* getBuffer(unsigned int sizeInByte);
	
	
	// 释放内存=======
	/** Returns an allocated buffer back to the global storage for re-use.
	  * Corresponds to "delete[] buffer". */
	void returnBuffer(void* buffer);// returnBuffer只是把内存还回去(放入map availableBuffers 中 )
	void releaseBuffes();// 用来释放内存的
	
       // 激活帧 ?？ 添加/删除关键帧  关键帧剪枝 
	boost::shared_lock<boost::shared_mutex> activateFrame(Frame* frame);// 激活
	void deactivateFrame(Frame* frame);// 失活
	void pruneActiveFrames();//激活 帧  剪枝


private:
	FrameMemory();
	void* allocateBuffer(unsigned int sizeInByte);// Eigen的内存管理函数 把buffer的首地址和尺寸映射起来
	
	boost::mutex accessMutex;// 访问内存 锁
	std::unordered_map< void*, unsigned int > bufferSizes;// 总大小 首地址 内存空间大小
	std::unordered_map< unsigned int, std::vector< void* > > availableBuffers;// 可用大小


	boost::mutex activeFramesMutex;// 激活 关键帧锁
	std::list<Frame*> activeFrames;// 激活的帧  关键帧???
};

}
