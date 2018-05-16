/**
* This file is part of LSD-SLAM.
* 内存管理 
* 最科学的欣赏源码方式，必然是先看内存管理，
* 即进入文件夹下的第二个文件FrameMemory.h
* 这个文件只有一个类FrameMemory,接口也不多
* 
* 申请空间
* 释放删除空间
* 关键帧添加删除剪枝
*/

#include "DataStructures/FrameMemory.h"
#include "DataStructures/Frame.h"// 帧结构

namespace lsd_slam//命名空间
{

FrameMemory::FrameMemory()//类
{
}
//===========================================
// 返回帧内存对象实例  注意为  函数返回值为 引用
FrameMemory& FrameMemory::getInstance()
{
	static FrameMemory theOneAndOnly;// 静态变量 存在与静态区， 只被定义一次，存在于整个进程的生存期
	// 也就是说，一个进程里面，有且仅有一个FrameMemory的对象 theOneAndOnly
	return theOneAndOnly;
}
// getFloatBuffer明显调用了getBuffer这个函数 
// 返回指定 字节大小的 空间
void* FrameMemory::getBuffer(unsigned int sizeInByte)
{
  // 首先 直接调用boost里面的互斥锁，把这段函数里面的这段内存锁上了 
	boost::unique_lock<boost::mutex> lock(accessMutex);
// 判断可用buffer中是否有sizeInByte这个大小的内存，如果有，那么返回1，没有返回0，
// 所以，搜索到会进入if内部，否则进入else内部 	
	if (availableBuffers.count(sizeInByte) > 0)
	{
	  // 首先是获取sizeInByte所对应大小的内存的地址的引用，也就是需要的内存的首地址
		std::vector< void* >& availableOfSize = availableBuffers.at(sizeInByte);
	    // 之后会判断
	      // 如果是空的
		if (availableOfSize.empty())
		{
			void* buffer = allocateBuffer(sizeInByte);
			// 调用allocateBUffer申请一段内存，注意在allocateBuffer内部调用了Eigen的内存管理函数
			// (底层实际上还是malloc，如果失败会抛出一个throw_std_bad_alloc)
//			assert(buffer != 0);
			// 之后做一个映射，把buffer的首地址和尺寸映射起来，
			// 之后返回buffer的首地址，这样便可以得到一个buffer
			return buffer;
		}
	      // 如果不是空，那么直接得到一个那个尺寸的内存，然后返回
		else
		{
			void* buffer = availableOfSize.back();
			availableOfSize.pop_back();
//			assert(buffer != 0);
			return buffer;
		}
	}
// 进入else : 如果没有这个尺寸的内存，就调用allocateBuffer申请一段，之后返回
	else
	{
		void* buffer = allocateBuffer(sizeInByte);
//		assert(buffer != 0);
		return buffer;
	}
}
// 获取指定大小的 float 单位的 内存孔家
float* FrameMemory::getFloatBuffer(unsigned int size)
{
	return (float*)getBuffer(sizeof(float) * size);
}

// 调用allocateBUffer申请一段内存，注意在allocateBuffer内部调用了Eigen的内存管理函数
// (底层实际上还是malloc，如果失败会抛出一个throw_std_bad_alloc)
// 之后做一个映射，把buffer的首地址和尺寸映射起来，
void* FrameMemory::allocateBuffer(unsigned int size)
{
	//printf("allocateFloatBuffer(%d)\n", size);
	void* buffer = Eigen::internal::aligned_malloc(size);// Eigen的内存管理函数
	bufferSizes.insert(std::make_pair(buffer, size));// 之后做一个映射，把buffer的首地址和尺寸映射起来
	return buffer;
}

// 用来释放内存的
void FrameMemory::releaseBuffes()
{
// 1. 首先 直接调用boost里面的互斥锁，把这段函数里面的这段内存锁上了 
	boost::unique_lock<boost::mutex> lock(accessMutex);
	int total = 0;


	for(auto p : availableBuffers)// 可以空间
	{
		if(printMemoryDebugInfo)
			printf("deleting %d buffers of size %d!\n", (int)p.second.size(), (int)p.first);

		total += p.second.size() * p.first;// 总字节大小

		for(unsigned int i=0;i<p.second.size();i++)
		{
			Eigen::internal::aligned_free(p.second[i]);// Eigen的内存管理函数
			bufferSizes.erase(p.second[i]);// 删除
		}
		p.second.clear();
	}
	availableBuffers.clear();

	if(printMemoryDebugInfo)
		printf("released %.1f MB!\n", total / (1000000.0f));
}
// returnBuffer只是把内存还回去(放入map availableBuffers 中 )
void FrameMemory::returnBuffer(void* buffer)
{
	if(buffer==0) return;// 空指针
 // 1. 首先 直接调用boost里面的互斥锁，把这段函数里面的这段内存锁上了 
	boost::unique_lock<boost::mutex> lock(accessMutex);
	
	unsigned int size = bufferSizes.at(buffer);// 指针指向的空间的大小 map  buffer的首地址和尺寸映射起来
	//printf("returnFloatBuffer(%d)\n", size);
	if (availableBuffers.count(size) > 0)
		availableBuffers.at(size).push_back(buffer);
	else
	{
		std::vector< void* > availableOfSize;
		availableOfSize.push_back(buffer);
		availableBuffers.insert(std::make_pair(size, availableOfSize));
	}
}

// 添加关键帧
// 成员 std::list
// 激活 帧 内存 调用boost里面的互斥锁 上锁
boost::shared_lock<boost::shared_mutex> FrameMemory::activateFrame(Frame* frame)
{
  // 激活 帧 内存 调用boost里面的互斥锁 上锁
	boost::unique_lock<boost::mutex> lock(activeFramesMutex);
	if(frame->isActive)
		activeFrames.remove(frame);
	activeFrames.push_front(frame);// 从最前面添加
	frame->isActive = true;// 激活标志
	// 共享锁？？？
	return boost::shared_lock<boost::shared_mutex>(frame->activeMutex);
}
// 灭火 帧   去除关键帧
void FrameMemory::deactivateFrame(Frame* frame)
{
  // 灭火 帧 内存 调用boost里面的互斥锁 上锁
	boost::unique_lock<boost::mutex> lock(activeFramesMutex);
	if(!frame->isActive) return;
	activeFrames.remove(frame);

	while(!frame->minimizeInMemory())
		printf("cannot deactivateFrame frame %d, as some acvite-lock is lingering. May cause deadlock!\n", frame->id());	// do it in a loop, to make shure it is really, really deactivated.

	frame->isActive = false;
}

// 对 关键帧 剪枝 
void FrameMemory::pruneActiveFrames()
{
  // 激活 帧 内存 调用boost里面的互斥锁 上锁
	boost::unique_lock<boost::mutex> lock(activeFramesMutex);

	while((int)activeFrames.size() > maxLoopClosureCandidates + 20)
	{
		if(!activeFrames.back()->minimizeInMemory())
		{
			if(!activeFrames.back()->minimizeInMemory())
			{
				printf("failed to minimize frame %d twice. maybe some active-lock is lingering?\n",activeFrames.back()->id());
				return;	 // pre-emptive return if could not deactivate.
			}
		}
		activeFrames.back()->isActive = false;
		activeFrames.pop_back();
	}
}

}
