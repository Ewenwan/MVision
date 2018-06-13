/*
主要用了Farneback计算光流，
把金字塔的方法也写进去了，
金字塔方法主要是为了消除不同尺寸的影响，
让描述子有更好的泛化能力。

博客参考：
https://blog.csdn.net/ironyoung/article/details/60884929
光流源码：
https://searchcode.com/file/30099587/opencv_source/src/cv/cvoptflowgf.cpp


孔径问题（Aperture Problem）： 
- http://blog.csdn.net/hankai1024/article/details/23433157 
- 形象理解：从小孔中观察一块移动的黑色幕布观察不到任何变化。
但实际情况是幕布一直在移动中 
- 解决方案：从不同尺度（图像金字塔）上对图像进行观察，
由高到低逐层利用上一层已求得信息来计算下一层信息。


光流基础概念
真实的三维空间中，描述物体运动状态的物理概念是运动场。三维空间中的每一个点，经过某段时间的运动之后会到达一个新的位置，而这个位移过程可以用运动场来描述。

而在计算机视觉的空间中，计算机所接收到的信号往往是二维图片信息。
由于缺少了一个维度的信息，所以其不再适用以运动场描述。
光流场（optical flow）就是用于描述三维空间中的运动物体表现到二维图像中，
所反映出的像素点的运动向量场。

光流法是利用图像序列中的像素在时间域上的变化、
相邻帧之间的相关性来找到的上一帧跟当前帧间存在的对应关系，
计算出相邻帧之间物体的运动信息的一种方法。
光流法理解的关键点有：

核心问题：同一个空间中的点，在下一帧即将出现的位置
重要假设：光流的变化（向量场）几乎是光滑
角点处的光流能够通过角点的邻域完全确定下来，
因此角点处的运动信息最为可靠；其次是边界的信息


理论基础
图像建模：
	将图像视为二维信号的函数（输出图像是灰度图像），
	因变量是二维坐标位置： x=(x, y)转置
	并且利用二次多项式对于图像进行近似建模的话，会得到：
		f(x) = x转置 * A * x + b转置 * x  + c
		其中，A 是一个2×2的矩阵，b是一个2×1的矩阵
		c = r1
		b= (r2, r3)
		A= (r4, r6;r6 r5)
		f(x) = =r1 + r2*x + r3*y + r4*x^2 + r5*y^2 + r6*x*y = 实际灰度值

	来求解出r1~r66个参数
	
求解空间转换：
	如果将原有（笛卡尔坐标系）图像的二维信号空间，
	转换到以 (1,x,y,x2,y2,xy) 作为基函数的空间，
	则表示图像需要一个六维向量作为系数，
	代入不同像素点的位置 x,y 求出不同像素点的灰度值。

	Farneback 算法对于每帧图像中的每个像素点周围设定一个邻域(2n+1)×(2n+1)
	利用邻域内的共(2n+1)2个像素点作为最小二乘法的样本点.
	拟合得到中心像素点的六维系数。
	因此对于图像中的每个像素点，都能得到一个六维向量。
	在一个邻域内灰度值的 (2n+1)×(2n+1) 矩阵中，
	将矩阵按列优先次序拆分组合成 (2n+1)^2 × 1 的向量 f，
	同时已知 (1,x,y,x2,y2,xy) 作为基函数的转换矩阵 B 维度为 (2n+1)^2 × 6
	（也可以视为 6 个列向量 bi 共同组成的矩阵），
	邻域内共有的系数向量 r 维度为 6×1, (1,x,y,x2,y2,xy)，则有： 
	f=B×r=(b1,b2,b3,b4,b5,b6)×r


权重分配：
	利用最小二乘法求解时，并非是邻域内每个像素点样本误差都
	对中心点有着同样的影响力，
	函数中利用二维高斯分布将影响力赋予权重。

 在一个邻域内二维高斯分布的 (2n+1)×(2n+1) 权重矩阵中，
 将矩阵按列优先次序拆分组合成 (2n+1)^2 × 1 的权重向量 a。
 因此原本的基函数的转换矩阵 B 将变为： 
 
 B=(a*b1, a*b2, a*b3, a*b4, a*b5, a*b6)
 
 
 对偶 (dual) 转换：
	 为了“进一步加快求解”得到系数矩阵 r，
	 博士论文 4.3 小节中提出使用对偶的方式再次转换基函数矩阵 B，
	 此时的对偶转换矩阵为 G，
	 经过转换后的基函数矩阵的列向量为 bi~。
	 博士论文中 G 的计算方式为
	 
	 G = ((a*b1,b1),...,(a*b6,b6);
			...
		  (a*b1,b1),...,(a*b6,b6))
	 而通过对偶转换之后，计算系数向量 r 的方式就简单了很多，
	 其中 ⋆ 表示两个信号的互相关（实质上类似于两个函数的卷积 
	 https://zh.wikipedia.org/wiki/%E4%BA%92%E7%9B%B8%E5%85%B3 ）
	 过程;
	 
函数输出
	子函数 FarnebackPolyExp 输出得到的是单张图像中每个像素点的
	系数向量 r（不包括常数项系数 r1，因为之后的计算光流过程中没有用到）

 
*/
#ifndef OPTICALFLOW_H_
#define OPTICALFLOW_H_

#include "DenseTrackStab.h"

#include <time.h>

using namespace cv;
// 命名空间
namespace my
{
// 利用 FarnebackPolyExp 函数分别得到了视频前后帧中每个像素点的系数向量
	static void
	FarnebackPolyExp( const Mat& src, Mat& dst, int n, double sigma )
	{
		int k, x, y;

		assert( src.type() == CV_32FC1 );
		int width = src.cols;
		int height = src.rows;
		AutoBuffer<float> kbuf(n*6 + 3), _row((width + n*2)*3);
		float* g = kbuf + n;
		float* xg = g + n*2 + 1;
		float* xxg = xg + n*2 + 1;
		float *row = (float*)_row + n*3;

		if( sigma < FLT_EPSILON )
			sigma = n*0.3;
		
//【基于邻域】产生二维高斯分布的基础是一维高斯分布，
// 一维高斯分布存储于数组 g 中，并且进行了求和后归一化：
		double s = 0.;
		for( x = -n; x <= n; x++ )
		{
			g[x] = (float)std::exp(-x*x/(2*sigma*sigma));
			s += g[x];//求和
		}
		s = 1./s;// 归一化
		
		for( x = -n; x <= n; x++ )
		{
			g[x] = (float)(g[x]*s);//1*高斯权重
			xg[x] = (float)(x*g[x]);// x*高斯权重
			xxg[x] = (float)(x*x*g[x]);// x^2*高斯权重
		}
		
// 【基于邻域】求解对偶转换矩阵 G
// 此时的二维高斯分布权重已经被按照列向量拉伸成为了一个 (2n+1)^2×1 的向量。    
		Mat_<double> G = Mat_<double>::zeros(6, 6);
		// 一维高斯分布数组 g 生成二维高斯分布权重 嵌套两层循环即可
		for( y = -n; y <= n; y++ )//一维水平高斯分布
			for( x = -n; x <= n; x++ )// 一维垂直高斯分布
			{
// 循环时可以利用矩阵的对称性减少计算量。并且因为邻域中心点为 (0,0) 点，
// 所以循环求和时G(0,1)+=g[y]*g[x]*x这类计算在x和-x上相互抵消，
// 结果必然为 0 无需计算：
				G(0,0) += g[y]*g[x];
				G(1,1) += g[y]*g[x]*x*x;
				G(3,3) += g[y]*g[x]*x*x*x*x;
				G(5,5) += g[y]*g[x]*x*x*y*y;
			}

		//G[0][0] = 1.;
		G(2,2) = G(0,3) = G(0,4) = G(3,0) = G(4,0) = G(1,1);
		G(4,4) = G(3,3);
		G(3,4) = G(4,3) = G(5,5);
// 实际计算中 G 的特殊结构会使得 G−1 的特殊结构，
// 所以只需要保存逆矩阵中几个特殊位置的元素即可。证明过程见博士论文的附录 A
		// invG:
		// [ x        e  e    ]
		// [    y             ]
		// [       y          ]
		// [ e        z       ]
		// [ e           z    ]
		// [                u ]
		Mat_<double> invG = G.inv(DECOMP_CHOLESKY);
		double ig11 = invG(1,1), ig03 = invG(0,3), ig33 = invG(3,3), ig55 = invG(5,5);

		dst.create( height, width, CV_32FC(5) );
// 分别在 vertical 和 horizontal 两个方向进行卷积计算，
// 卷积结果分别存在 row 数组和 b1~b6 中
		for( y = 0; y < height; y++ )
		{
			float g0 = g[0], g1, g2;
			float *srow0 = (float*)(src.data + src.step*y), *srow1 = 0;
			float *drow = (float*)(dst.data + dst.step*y);

			// vertical part of convolution
			for( x = 0; x < width; x++ )
			{
				row[x*3] = srow0[x]*g0;
				row[x*3+1] = row[x*3+2] = 0.f;
			}

			for( k = 1; k <= n; k++ )
			{
				g0 = g[k]; g1 = xg[k]; g2 = xxg[k];
				srow0 = (float*)(src.data + src.step*std::max(y-k,0));
				srow1 = (float*)(src.data + src.step*std::min(y+k,height-1));

				for( x = 0; x < width; x++ )
				{
					float p = srow0[x] + srow1[x];
					float t0 = row[x*3] + g0*p;
					float t1 = row[x*3+1] + g1*(srow1[x] - srow0[x]);
					float t2 = row[x*3+2] + g2*p;

					row[x*3] = t0;
					row[x*3+1] = t1;
					row[x*3+2] = t2;
				}
			}

			// horizontal part of convolution
			for( x = 0; x < n*3; x++ )
			{
				row[-1-x] = row[2-x];
				row[width*3+x] = row[width*3+x-3];
			}

			for( x = 0; x < width; x++ )
			{
				g0 = g[0];
				// r1 ~ 1, r2 ~ x, r3 ~ y, r4 ~ x^2, r5 ~ y^2, r6 ~ xy
				double b1 = row[x*3]*g0, b2 = 0, b3 = row[x*3+1]*g0,
					b4 = 0, b5 = row[x*3+2]*g0, b6 = 0;

				for( k = 1; k <= n; k++ )
				{
					double tg = row[(x+k)*3] + row[(x-k)*3];
					g0 = g[k];
					b1 += tg*g0;
					b4 += tg*xxg[k];
					b2 += (row[(x+k)*3] - row[(x-k)*3])*xg[k];
					b3 += (row[(x+k)*3+1] + row[(x-k)*3+1])*g0;
					b6 += (row[(x+k)*3+1] - row[(x-k)*3+1])*xg[k];
					b5 += (row[(x+k)*3+2] + row[(x-k)*3+2])*g0;
				}

				// do not store r1
				drow[x*5+1] = (float)(b2*ig11);
				drow[x*5] = (float)(b3*ig11);
				drow[x*5+3] = (float)(b1*ig03 + b4*ig33);
				drow[x*5+2] = (float)(b1*ig03 + b5*ig33);
				drow[x*5+4] = (float)(b6*ig55);
			}
		}

		row -= n*3;
	}
/*
子函数 FarnebackUpdateMatrices 中需要的变量参数包括： 
1. _R0：输入前一帧图像（编号：0） 
2. _R1：输入后一帧图像（编号：1） 
3. _flow：已知的前一帧图像光流场（用于 Blur 子函数的迭代，以及主函数中图像金字塔的不同层数间迭代） 
4. _M：储存中间变量，不断更新 
5. _y0：图像求解光流的起始行 
6. _y1：图像求解光流的终止行

需要利用 FarnebackPolyExp 函数得到的 每个像素点的系数向量来计算得到光流场。

*/ 
	static void
	FarnebackUpdateMatrices( const Mat& _R0, const Mat& _R1, const Mat& _flow, Mat& matM, int _y0, int _y1 )
	{
		const int BORDER = 5;
		static const float border[BORDER] = {0.14f, 0.14f, 0.4472f, 0.4472f, 0.4472f};

		int x, y, width = _flow.cols, height = _flow.rows;
		const float* R1 = (float*)_R1.data;
		size_t step1 = _R1.step/sizeof(R1[0]);

		matM.create(height, width, CV_32FC(5));

		for( y = _y0; y < _y1; y++ )
		{
			const float* flow = (float*)(_flow.data + y*_flow.step);
			const float* R0 = (float*)(_R0.data + y*_R0.step);
			float* M = (float*)(matM.data + y*matM.step);

			for( x = 0; x < width; x++ )
			{
				float dx = flow[x*2], dy = flow[x*2+1];
				float fx = x + dx, fy = y + dy;

				int x1 = cvFloor(fx), y1 = cvFloor(fy);
				const float* ptr = R1 + y1*step1 + x1*5;
				float r2, r3, r4, r5, r6;

				fx -= x1; fy -= y1;
	// 二次插值得到新一帧图像位置中的系数向量：
				if( (unsigned)x1 < (unsigned)(width-1) &&
					(unsigned)y1 < (unsigned)(height-1) )
				{
					float a00 = (1.f-fx)*(1.f-fy), a01 = fx*(1.f-fy),
						  a10 = (1.f-fx)*fy, a11 = fx*fy;

					r2 = a00*ptr[0] + a01*ptr[5] + a10*ptr[step1] + a11*ptr[step1+5];
					r3 = a00*ptr[1] + a01*ptr[6] + a10*ptr[step1+1] + a11*ptr[step1+6];
					r4 = a00*ptr[2] + a01*ptr[7] + a10*ptr[step1+2] + a11*ptr[step1+7];
					r5 = a00*ptr[3] + a01*ptr[8] + a10*ptr[step1+3] + a11*ptr[step1+8];
					r6 = a00*ptr[4] + a01*ptr[9] + a10*ptr[step1+4] + a11*ptr[step1+9];

					r4 = (R0[x*5+2] + r4)*0.5f;
					r5 = (R0[x*5+3] + r5)*0.5f;
					r6 = (R0[x*5+4] + r6)*0.25f;
				}
				else
				{
					r2 = r3 = 0.f;
					r4 = R0[x*5+2];
					r5 = R0[x*5+3];
					r6 = R0[x*5+4]*0.5f;
				}

				r2 = (R0[x*5] - r2)*0.5f;
				r3 = (R0[x*5+1] - r3)*0.5f;

				r2 += r4*dy + r6*dx;
				r3 += r6*dy + r5*dx;
// 处理不同尺度上的缩放，这里借鉴的是 multi-scale 思路，详见博士论文的 7.7 小节，目的是为了提高本算法的鲁棒性：
				if( (unsigned)(x - BORDER) >= (unsigned)(width - BORDER*2) ||
					(unsigned)(y - BORDER) >= (unsigned)(height - BORDER*2))
				{
					float scale = (x < BORDER ? border[x] : 1.f)*
						(x >= width - BORDER ? border[width - x - 1] : 1.f)*
						(y < BORDER ? border[y] : 1.f)*
						(y >= height - BORDER ? border[height - y - 1] : 1.f);

					r2 *= scale; r3 *= scale; r4 *= scale;
					r5 *= scale; r6 *= scale;
				}
// 计算中间变量 G 与 h，存储到矩阵 M 中：
				M[x*5]   = r4*r4 + r6*r6; // G(1,1)
				M[x*5+1] = (r4 + r5)*r6;  // G(1,2)=G(2,1)
				M[x*5+2] = r5*r5 + r6*r6; // G(2,2)
				M[x*5+3] = r4*r2 + r6*r3; // h(1)
				M[x*5+4] = r6*r2 + r5*r3; // h(2)
			}
		}
	}
	
//光流的变化（向量场）几乎是光滑的。因此利用中间变量 G 与 h 求解光流场 dout 前，
//需要进行一次局部模糊化处理（主函数中 winsize 输入变量控制），
//可选均值模糊 (FarnebackUpdateFlow_Blur)、高斯模糊 (FarnebackUpdateFlow_GaussianBlur)。
//对于模糊后的中间变量，可以直接求解光流场： 
	static void
	FarnebackUpdateFlow_GaussianBlur( const Mat& _R0, const Mat& _R1,
									  Mat& _flow, Mat& matM, int block_size,
									  bool update_matrices )
	{
		int x, y, i, width = _flow.cols, height = _flow.rows;
		int m = block_size/2;
		int y0 = 0, y1;
		int min_update_stripe = std::max((1 << 10)/width, block_size);
		double sigma = m*0.3, s = 1;

		AutoBuffer<float> _vsum((width+m*2+2)*5 + 16), _hsum(width*5 + 16);
		AutoBuffer<float, 4096> _kernel((m+1)*5 + 16);
		AutoBuffer<float*, 1024> _srow(m*2+1);
		float *vsum = alignPtr((float*)_vsum + (m+1)*5, 16), *hsum = alignPtr((float*)_hsum, 16);
		float* kernel = (float*)_kernel;
		const float** srow = (const float**)&_srow[0];
		kernel[0] = (float)s;

		for( i = 1; i <= m; i++ )
		{
			float t = (float)std::exp(-i*i/(2*sigma*sigma) );
			kernel[i] = t;
			s += t*2;
		}

		s = 1./s;
		for( i = 0; i <= m; i++ )
			kernel[i] = (float)(kernel[i]*s);

	#if CV_SSE2
		float* simd_kernel = alignPtr(kernel + m+1, 16);
		volatile bool useSIMD = checkHardwareSupport(CV_CPU_SSE);
		if( useSIMD )
		{
			for( i = 0; i <= m; i++ )
				_mm_store_ps(simd_kernel + i*4, _mm_set1_ps(kernel[i]));
		}
	#endif

		// compute blur(G)*flow=blur(h)
		for( y = 0; y < height; y++ )
		{
			double g11, g12, g22, h1, h2;
			float* flow = (float*)(_flow.data + _flow.step*y);

			// vertical blur
			for( i = 0; i <= m; i++ )
			{
				srow[m-i] = (const float*)(matM.data + matM.step*std::max(y-i,0));
				srow[m+i] = (const float*)(matM.data + matM.step*std::min(y+i,height-1));
			}

			x = 0;
	#if CV_SSE2
			if( useSIMD )
			{
				for( ; x <= width*5 - 16; x += 16 )
				{
					const float *sptr0 = srow[m], *sptr1;
					__m128 g4 = _mm_load_ps(simd_kernel);
					__m128 s0, s1, s2, s3;
					s0 = _mm_mul_ps(_mm_loadu_ps(sptr0 + x), g4);
					s1 = _mm_mul_ps(_mm_loadu_ps(sptr0 + x + 4), g4);
					s2 = _mm_mul_ps(_mm_loadu_ps(sptr0 + x + 8), g4);
					s3 = _mm_mul_ps(_mm_loadu_ps(sptr0 + x + 12), g4);

					for( i = 1; i <= m; i++ )
					{
						__m128 x0, x1;
						sptr0 = srow[m+i], sptr1 = srow[m-i];
						g4 = _mm_load_ps(simd_kernel + i*4);
						x0 = _mm_add_ps(_mm_loadu_ps(sptr0 + x), _mm_loadu_ps(sptr1 + x));
						x1 = _mm_add_ps(_mm_loadu_ps(sptr0 + x + 4), _mm_loadu_ps(sptr1 + x + 4));
						s0 = _mm_add_ps(s0, _mm_mul_ps(x0, g4));
						s1 = _mm_add_ps(s1, _mm_mul_ps(x1, g4));
						x0 = _mm_add_ps(_mm_loadu_ps(sptr0 + x + 8), _mm_loadu_ps(sptr1 + x + 8));
						x1 = _mm_add_ps(_mm_loadu_ps(sptr0 + x + 12), _mm_loadu_ps(sptr1 + x + 12));
						s2 = _mm_add_ps(s2, _mm_mul_ps(x0, g4));
						s3 = _mm_add_ps(s3, _mm_mul_ps(x1, g4));
					}

					_mm_store_ps(vsum + x, s0);
					_mm_store_ps(vsum + x + 4, s1);
					_mm_store_ps(vsum + x + 8, s2);
					_mm_store_ps(vsum + x + 12, s3);
				}

				for( ; x <= width*5 - 4; x += 4 )
				{
					const float *sptr0 = srow[m], *sptr1;
					__m128 g4 = _mm_load_ps(simd_kernel);
					__m128 s0 = _mm_mul_ps(_mm_loadu_ps(sptr0 + x), g4);

					for( i = 1; i <= m; i++ )
					{
						sptr0 = srow[m+i], sptr1 = srow[m-i];
						g4 = _mm_load_ps(simd_kernel + i*4);
						__m128 x0 = _mm_add_ps(_mm_loadu_ps(sptr0 + x), _mm_loadu_ps(sptr1 + x));
						s0 = _mm_add_ps(s0, _mm_mul_ps(x0, g4));
					}
					_mm_store_ps(vsum + x, s0);
				}
			}
	#endif
			for( ; x < width*5; x++ )
			{
				float s0 = srow[m][x]*kernel[0];
				for( i = 1; i <= m; i++ )
					s0 += (srow[m+i][x] + srow[m-i][x])*kernel[i];
				vsum[x] = s0;
			}

			// update borders
			for( x = 0; x < m*5; x++ )
			{
				vsum[-1-x] = vsum[4-x];
				vsum[width*5+x] = vsum[width*5+x-5];
			}

			// horizontal blur
			x = 0;
	#if CV_SSE2
			if( useSIMD )
			{
				for( ; x <= width*5 - 8; x += 8 )
				{
					__m128 g4 = _mm_load_ps(simd_kernel);
					__m128 s0 = _mm_mul_ps(_mm_loadu_ps(vsum + x), g4);
					__m128 s1 = _mm_mul_ps(_mm_loadu_ps(vsum + x + 4), g4);

					for( i = 1; i <= m; i++ )
					{
						g4 = _mm_load_ps(simd_kernel + i*4);
						__m128 x0 = _mm_add_ps(_mm_loadu_ps(vsum + x - i*5),
											   _mm_loadu_ps(vsum + x + i*5));
						__m128 x1 = _mm_add_ps(_mm_loadu_ps(vsum + x - i*5 + 4),
											   _mm_loadu_ps(vsum + x + i*5 + 4));
						s0 = _mm_add_ps(s0, _mm_mul_ps(x0, g4));
						s1 = _mm_add_ps(s1, _mm_mul_ps(x1, g4));
					}

					_mm_store_ps(hsum + x, s0);
					_mm_store_ps(hsum + x + 4, s1);
				}
			}
	#endif
			for( ; x < width*5; x++ )
			{
				float sum = vsum[x]*kernel[0];
				for( i = 1; i <= m; i++ )
					sum += kernel[i]*(vsum[x - i*5] + vsum[x + i*5]);
				hsum[x] = sum;
			}

			for( x = 0; x < width; x++ )
			{
				g11 = hsum[x*5];
				g12 = hsum[x*5+1];
				g22 = hsum[x*5+2];
				h1 = hsum[x*5+3];
				h2 = hsum[x*5+4];

				double idet = 1./(g11*g22 - g12*g12 + 1e-3);

				flow[x*2] = (float)((g11*h2-g12*h1)*idet);
				flow[x*2+1] = (float)((g22*h1-g12*h2)*idet);
			}

			y1 = y == height - 1 ? height : y - block_size;
			if( update_matrices && (y1 == height || y1 >= y0 + min_update_stripe) )
			{
				FarnebackUpdateMatrices( _R0, _R1, _flow, matM, y0, y1 );
				y0 = y1;
			}
		}
	}
// 对光流进行中值滤波
	// in-place median blur for optical flow
	void MedianBlurFlow(Mat& flow, const int ksize)
	{
		Mat channels[2];//水平和垂直光流
		split(flow, channels);//分割
		//进行中值滤波  opencv内的函数
		medianBlur(channels[0], channels[0], ksize);
		medianBlur(channels[1], channels[1], ksize);
		//合并中值滤波后的两个通道
		merge(channels, 2, flow);
	}

	void FarnebackPolyExpPyr(const Mat& img, std::vector<Mat>& poly_exp_pyr,
							 std::vector<float>& fscales, int poly_n, double poly_sigma)
	{
		Mat fimg;

		for(int k = 0; k < poly_exp_pyr.size(); k++)
		{
			double sigma = (fscales[k]-1)*0.5;
			int smooth_sz = cvRound(sigma*5)|1;
			smooth_sz = std::max(smooth_sz, 3);

			int width = poly_exp_pyr[k].cols;
			int height = poly_exp_pyr[k].rows;

			Mat R, I;

			img.convertTo(fimg, CV_32F);
			GaussianBlur(fimg, fimg, Size(smooth_sz, smooth_sz), sigma, sigma);
			resize(fimg, I, Size(width, height), CV_INTER_LINEAR);

			FarnebackPolyExp(I, R, poly_n, poly_sigma);
			R.copyTo(poly_exp_pyr[k]);
		}
	}
	
//计算光流===============================
// prev_poly_exp_pyr 上一帧灰度图金字塔
// poly_exp_pyr      当前帧灰度图金字塔
// flow_pyr          计算得到的光流金字塔
	void calcOpticalFlowFarneback(std::vector<Mat>& prev_poly_exp_pyr, std::vector<Mat>& poly_exp_pyr,
								  std::vector<Mat>& flow_pyr, int winsize, int iterations)
	{
		int i, k;
		Mat prevFlow, flow;//上一层光流 当前层光流

		for( k = flow_pyr.size() - 1; k >= 0; k-- )//每一层
		{
			int width = flow_pyr[k].cols;//尺寸 列数 宽度
			int height = flow_pyr[k].rows;// 行数 高度

			flow.create( height, width, CV_32FC2 );// 2个通道

			if( !prevFlow.data )
				flow = Mat::zeros( height, width, CV_32FC2 );//初始化为0
			else {
				// 各行各列下采样
				resize( prevFlow, flow, Size(width, height), 0, 0, INTER_LINEAR );
				flow *= scale_stride;
			}

			Mat R[2], M;

			prev_poly_exp_pyr[k].copyTo(R[0]);
			poly_exp_pyr[k].copyTo(R[1]);
			//更新
			FarnebackUpdateMatrices( R[0], R[1], flow, M, 0, flow.rows );

			for( i = 0; i < iterations; i++ )//迭代
				FarnebackUpdateFlow_GaussianBlur( R[0], R[1], flow, M, winsize, i < iterations - 1 );
			//均值滤波
			MedianBlurFlow(flow, 5);
			prevFlow = flow;//上一层光流
			flow.copyTo(flow_pyr[k]);
		}
	}

}

#endif /*OPTICALFLOW_H_*/
