/**
* This file is part of LSD-SLAM.
* 最小二乘 优化  更新 update
*  第一个类型LSGX，这里面有3个类，分别是LSG4,LSG6和LSG7，
* 他们定义了4个参数6个参数以及7个参数的最小二乘法  优化算法
* 
* 
* 
* 
* 
* X86框架下 	SSE 单指令多数据流式扩展 优化
*                     使用SSE内嵌原语  + 使用SSE汇编
* ARM平台 NEON 指令  优化
* 参考 http://zyddora.github.io/2016/02/28/neon_1/
* NEON就是一种基于SIMD思想的ARM技术
* Single Instruction Multiple Data (SIMD)顾名思义就是“一条指令处理多个数据 ， 并行处理技术
* 
* 16×    128-bit 寄存器(Q0-Q15)；
* 或32× 64-bit  寄存器(D0-D31)
* 或上述寄存器的组合。
*  实际上D寄存器和Q寄存器是重叠的
* 
* 所有的支持NEON指令都有一个助记符V，下面以32位指令为例，说明指令的一般格式：
* V{<mod>}<op>{<shape>}{<cond>}{.<dt>}{<dest>}, src1, src2
* 1. <mod>  模式  Q  H   D   R 系列
* 2. <op>     操作   ADD, SUB, MUL    加法  减法  乘法
* 3. <shape> 数据形状  
*                    Long (L),         操作双字vectors，生成四倍长字vectors   结果的宽度一般比操作数加倍，同类型
*                    Wide (W),       操作双字 + 四倍长字，生成四倍长字  结果和第一个操作数都是第二个操作数的两倍宽度
*                    Narrow (N)    操作四倍长字，生成双字  结果宽度一般是操作数的一半
* 5. .<dt>      数据类型        s8, u8, f32 
* 
* 6. <dest>  目的地址 
* 7. <src1>  源操作数1
* 8. <src2>  源操作数2
* 
*/

#include "least_squares.h"
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>// 奇异值分解 得到特征值 和 特征向量
#include <stdio.h>

namespace lsd_slam
{


NormalEquationsLeastSquares::~NormalEquationsLeastSquares() { }
// 类初始化 
void NormalEquationsLeastSquares::initialize(const size_t maxnum_constraints)
{
  A.setZero();// 置为0
  A_opt.setZero();
  b.setZero();
  solved = false;// 优化求解标志 
  error = 0;
  this->num_constraints = 0;
  this->maxnum_constraints = maxnum_constraints;
}

inline void NormalEquationsLeastSquares::update(const Vector6& J, const float& res, const float& weight)
{
// 调试 打印信息
//	printf("up: %f, %f, %f, %f, %f, %f; res: %f; w: %f\n",
//			J[0],J[1],J[2],J[3],J[4],J[5],res, weight);
  
//  使用指令集优化 提高运算速度
  A_opt.rankUpdate(J, weight);// J * J.transpose() * weight
  //MathSse<Sse::Enabled, float>::addOuterProduct(A, J, factor);
  //A += J * J.transpose() * factor;
  
  //MathSse<Sse::Enabled, float>::add(b, J, -res * factor); // not much difference :(
  b -= J * (res * weight);

  error += res * res * weight;
  num_constraints += 1;
}
// 结合 
void NormalEquationsLeastSquares::combine(const NormalEquationsLeastSquares& other)
{
  A_opt += other.A_opt;
  b += other.b;
  error += other.error;
  num_constraints += other.num_constraints;
}
// 结果类型转换 带有约束
void NormalEquationsLeastSquares::finish()
{
  A_opt.toEigen(A);
  A /= (float) num_constraints;
  b /= (float) num_constraints;
  error /= (float) num_constraints;
}
// 结果类型转换 
void NormalEquationsLeastSquares::finishNoDivide()
{
  A_opt.toEigen(A);
}
// 优化
void NormalEquationsLeastSquares::solve(Vector6& x)
{
  x = A.ldlt().solve(b);
  solved = true;
}


OptimizedSelfAdjointMatrix6x6f::OptimizedSelfAdjointMatrix6x6f()
{
}

void OptimizedSelfAdjointMatrix6x6f::setZero()
{
  for(size_t idx = 0; idx < Size; idx++)
    data[idx] = 0.0f;// 置0
}

#if !defined(ENABLE_SSE) && !defined(__SSE__)
//  X86框架下 	SSE 单指令多数据流式扩展 优化 
// 定义的 加法 乘法 函数
// 单指令多数据流式扩展（SSE，Streaming SIMD Extensions）技术能够有效增强CPU浮点运算的能力
	// TODO: Ugly temporary replacement for SSE instructions to make rankUpdate() work.
	// TODO: code faster version
// __m128（4个float）、__m128d（2个double）、__m128i(int、short、char)直接控制那些128-bit的寄存器。	
	struct __m128 {
		__m128(float a, float b, float c, float d) {
			data[0] = a;
			data[1] = b;
			data[2] = c;
			data[3] = d;
		}
		float& operator[](int idx) {
			return data[idx];
		}
		const float& operator[](int idx) const {
			return data[idx];
		}
		float data[4];
	};
	__m128 _mm_set1_ps(float v) {
		return __m128(v, v, v, v);
	}
	__m128 _mm_loadu_ps(const float* d) {
		return __m128(d[0], d[1], d[2], d[3]);
	}
	__m128 _mm_load_ps(const float* d) {
		return __m128(d[0], d[1], d[2], d[3]);
	}
	__m128 _mm_movelh_ps(const __m128& a, const __m128& b) {
		return __m128(a[0], a[1], b[0], b[1]);
	}
	__m128 _mm_movehl_ps(const __m128& a, const __m128& b) {
		return __m128(b[2], b[3], a[2], a[3]);
	}
	__m128 _mm_unpacklo_ps(const __m128& a, const __m128& b) {
		return __m128(a[0], b[0], a[1], b[1]);
	}
      // 乘法
	__m128 _mm_mul_ps(const __m128& a, const __m128& b) {
		return __m128(a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]);
	}
      // 加法
	__m128 _mm_add_ps(const __m128& a, const __m128& b) {
		return __m128(a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]);
	}
	void _mm_store_ps(float* p, const __m128& a) {
		p[0] = a[0];
		p[1] = a[1];
		p[2] = a[2];
		p[3] = a[3];
	}
	
#endif


// 使用SSE/NENO指令集 优化 矩阵运算
// u * u.transpose() * alpha
inline void OptimizedSelfAdjointMatrix6x6f::rankUpdate(const Eigen::Matrix<float, 6, 1>& u, const float alpha)
{
//  ARM平台 NEON 指令  优化
// 16×    128-bit 寄存器(Q0-Q15)；
// 或32× 64-bit  寄存器(D0-D31)
// 或上述寄存器的组合。

#if defined(ENABLE_NEON)
	
	const float* in_ptr = u.data();
	float* out_ptr = data;
	__asm__ __volatile__
	(
		// NOTE: may reduce count of used registers by calculating some value(s?) later.
		
		"vdup.32  q15, %[alpha]               \n\t" // alpha(q15)
		"vldmia   %[in_ptr], {q9-q10}         \n\t" // v1234(q9), v56xx(q10)
		"vldmia   %[out_ptr], {q0-q5}         \n\t"

		"vmov     d21, d20                    \n\t" // v5656(q10)
		"vmov     q11, q10                    \n\t"
		"vmov     q12, q10                    \n\t"
		"vzip.32  q11, q12                    \n\t" // v5566(q11)
		"vmul.f32 q11, q11, q15               \n\t" // alpha*v3344(q14)
	
		"vmov     q12, q9                     \n\t"
		"vmov     d19, d18                    \n\t" // v1212(q9)
		"vmov     d24, d25                    \n\t" // v3434(q12)
	
		"vmov     q13, q9                     \n\t"
		"vmov     q14, q9                     \n\t"
		"vzip.32  q13, q14                    \n\t" // v1122(q13)
		"vmul.f32 q13, q13, q15               \n\t" // alpha*v1122(q13)
	
		"vmov     q14, q12                    \n\t"
		"vmov     q8, q12                     \n\t"
		"vzip.32  q14, q8                     \n\t" // v3344(q14)
		"vmul.f32 q14, q14, q15               \n\t" // alpha*v3344(q14)
	
		"vmla.f32 q0, q13, q9                 \n\t"
		"vmla.f32 q1, q13, q12                \n\t"
		"vmla.f32 q2, q13, q10                \n\t"
		
		"vmla.f32 q3, q14, q12                \n\t"
		"vmla.f32 q4, q14, q10                \n\t"
		
		"vmla.f32 q5, q11, q10                \n\t"
		
		"vstmia %[out_ptr], {q0-q5}           \n\t"

	: /* outputs */ 
	: /* inputs  */ [alpha]"r"(alpha), [in_ptr]"r"(in_ptr), [out_ptr]"r"(out_ptr)
	: /* clobber */ "memory", "cc", // TODO: is cc necessary?
					"q0", "q1", "q2", "q3", "q4", "q5", "q8", "q9", "q10", "q11", "q12", "q13", "q14"
	);

//  X86框架下 	SSE 单指令多数据流式扩展 优化
#else
	
  __m128 s = _mm_set1_ps(alpha);
  __m128 v1234 = _mm_loadu_ps(u.data());
  __m128 v56xx = _mm_loadu_ps(u.data() + 4);

  __m128 v1212 = _mm_movelh_ps(v1234, v1234);
  __m128 v3434 = _mm_movehl_ps(v1234, v1234);
  __m128 v5656 = _mm_movelh_ps(v56xx, v56xx);

  __m128 v1122 = _mm_mul_ps(s, _mm_unpacklo_ps(v1212, v1212));// 乘法

  _mm_store_ps(data + 0, _mm_add_ps(_mm_load_ps(data + 0), _mm_mul_ps(v1122, v1212)));
  _mm_store_ps(data + 4, _mm_add_ps(_mm_load_ps(data + 4), _mm_mul_ps(v1122, v3434)));
  _mm_store_ps(data + 8, _mm_add_ps(_mm_load_ps(data + 8), _mm_mul_ps(v1122, v5656)));

  __m128 v3344 = _mm_mul_ps(s, _mm_unpacklo_ps(v3434, v3434));

  _mm_store_ps(data + 12, _mm_add_ps(_mm_load_ps(data + 12), _mm_mul_ps(v3344, v3434)));
  _mm_store_ps(data + 16, _mm_add_ps(_mm_load_ps(data + 16), _mm_mul_ps(v3344, v5656)));

  __m128 v5566 = _mm_mul_ps(s, _mm_unpacklo_ps(v5656, v5656));

  _mm_store_ps(data + 20, _mm_add_ps(_mm_load_ps(data + 20), _mm_mul_ps(v5566, v5656)));
  
#endif
}

inline void OptimizedSelfAdjointMatrix6x6f::operator +=(const OptimizedSelfAdjointMatrix6x6f& other)
{
//  X86框架下 	SSE 单指令多数据流式扩展 优化
#if defined(ENABLE_SSE)
  _mm_store_ps(data +  0, _mm_add_ps(_mm_load_ps(data +  0), _mm_load_ps(other.data +  0)));
  _mm_store_ps(data +  4, _mm_add_ps(_mm_load_ps(data +  4), _mm_load_ps(other.data +  4)));
  _mm_store_ps(data +  8, _mm_add_ps(_mm_load_ps(data +  8), _mm_load_ps(other.data +  8)));
  _mm_store_ps(data + 12, _mm_add_ps(_mm_load_ps(data + 12), _mm_load_ps(other.data + 12)));
  _mm_store_ps(data + 16, _mm_add_ps(_mm_load_ps(data + 16), _mm_load_ps(other.data + 16)));
  _mm_store_ps(data + 20, _mm_add_ps(_mm_load_ps(data + 20), _mm_load_ps(other.data + 20)));
  
#elif defined(ENABLE_NEON)
    const float* other_data_ptr = other.data;
  	__asm__ __volatile__
	(
		// NOTE: The way of loading the data was benchmarked and this was the
		// fastest variant (faster than loading everything in order, and faster
		// than loading both blocks at once).
		"vldmia   %[other_data]!, {q0-q2}     \n\t"
		"vldmia   %[data], {q9-q14}           \n\t"
		"vldmia   %[other_data], {q3-q5}      \n\t"
		
		"vadd.f32 q0, q0, q9                  \n\t"
		"vadd.f32 q1, q1, q10                 \n\t"
		"vadd.f32 q2, q2, q11                 \n\t"
		"vadd.f32 q3, q3, q12                 \n\t"
		"vadd.f32 q4, q4, q13                 \n\t"
		"vadd.f32 q5, q5, q14                 \n\t"
		
		"vstmia %[data], {q0-q5}              \n\t"

	: /* outputs */ [other_data]"+&r"(other_data_ptr)
	: /* inputs  */ [data]"r"(data)
	: /* clobber */ "memory",
					"q0", "q1", "q2", "q3", "q4", "q5", "q9", "q10", "q11", "q12", "q13", "q14"
	);
#else
  for(size_t idx = 0; idx < Size; idx++)
    data[idx] += other.data[idx];
#endif
}
// 结果 转换到 eigen 类型下
void OptimizedSelfAdjointMatrix6x6f::toEigen(Eigen::Matrix<float, 6, 6>& m) const
{
  Eigen::Matrix<float, 6, 6> tmp;
  size_t idx = 0;

  for(size_t i = 0; i < 6; i += 2)
  {
    for(size_t j = i; j < 6; j += 2)
    {
      tmp(i  , j  ) = data[idx++];
      tmp(i  , j+1) = data[idx++];
      tmp(i+1, j  ) = data[idx++];
      tmp(i+1, j+1) = data[idx++];
    }
  }

  tmp.selfadjointView<Eigen::Upper>().evalTo(m);
}



// 4维变量 最小二乘优化 
NormalEquationsLeastSquares4::~NormalEquationsLeastSquares4() { }

void NormalEquationsLeastSquares4::initialize(const size_t maxnum_constraints)
{
  A.setZero();// 置0
  b.setZero();
  solved = false;
  error = 0;
  this->num_constraints = 0;
  this->maxnum_constraints = maxnum_constraints;
}

inline void NormalEquationsLeastSquares4::update(const Vector4& J, const float& res, const float& weight)
{
// TODO: SSE optimization
// SSE 或者 NENO指令集优化 
  A.noalias() += J * J.transpose() * weight;// eigen 的 noalias()机制 避免中间结果 类的 构造
  b.noalias() -= J * (res * weight);
  error += res * res * weight;
  num_constraints += 1;
}
// 结合
void NormalEquationsLeastSquares4::combine(const NormalEquationsLeastSquares4& other)
{
  A += other.A;
  b += other.b;
  error += other.error;
  num_constraints += other.num_constraints;
}

void NormalEquationsLeastSquares4::finishNoDivide()
{
	// TODO: SSE optimization
}

NormalEquationsLeastSquares7::~NormalEquationsLeastSquares7() { }

void NormalEquationsLeastSquares7::initializeFrom(const NormalEquationsLeastSquares& ls6, const NormalEquationsLeastSquares4& ls4)
{
	// set zero
	A.setZero();
	b.setZero();

	// add ls6
	A.topLeftCorner<6,6>() = ls6.A;
	b.head<6>() = ls6.b;

	// add ls4
	int remap[4] = {2,3,4,6};
	for(int i=0;i<4;i++)
	{
		b[remap[i]] += ls4.b[i];
		for(int j=0;j<4;j++)
			A(remap[i], remap[j]) += ls4.A(i,j);
	}

	num_constraints = ls6.num_constraints + ls4.num_constraints;
}
void NormalEquationsLeastSquares7::combine(const NormalEquationsLeastSquares7& other)
{
	  A += other.A;
	  b += other.b;
	  error += other.error;
	  num_constraints += other.num_constraints;
}

}
