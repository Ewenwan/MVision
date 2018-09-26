/**************************************************/
//Created by allen hong 
//LICENSE : Do what u want
//Fork me on Github => https://github.com/xylcbd
// 修改：万有文
/**************************************************/
//std headers
#include <vector>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <cassert>
#include <random>

#define WIN32// WIN32平台

/*
//project headers
#ifdef WIN32
#include "NEONvsSSE.h"
#pragma message("x86_sse_platform") 
#elif defined(__ARM_NEON__)
#include "arm_neon.h"
#pragma message("arm_neon_platform") 
#else
#error "unknown platform!"
#endif
*/

#include <xmmintrin.h>     //SSE
#include <emmintrin.h>     //SSE2
#include <pmmintrin.h>     //SSE3
#include <tmmintrin.h>     //SSSE3


#include "helper_functions.h"// 时间记录函数
// 1. 在容器中填充随机数===========

// 生成器generator：能够产生离散的等可能分布数值
// 分布器distributions: 能够把generator产生的均匀分布值映射到其他常见分布，
//                     如均匀分布uniform，正态分布normal，二项分布binomial，泊松分布poisson
static void fill_random_value(std::vector<float>& vec_data)
{        
        // 浮点数均匀分布 分布器    uniform_int_distribution(整数均匀分布)
	std::uniform_real_distribution<float> distribution(
		std::numeric_limits<float>::min(),
		std::numeric_limits<float>::max());
        // 随机数 生成器
	std::default_random_engine generator;
        // std::default_random_engine generator(time(NULL));// 配合随机数种子
	
        // 为 vec_data 生成随机数，传入首尾迭代器和一个 lambda匿名函数
	// [变量截取](参数表){函数提体}； [&](){}, 中括号内的& 表示函数可使用函数外部的变量。
	std::generate(vec_data.begin(), vec_data.end(), [&]() { return distribution(generator); });
}
// 下面是各种变量截取的选项：
//   [] 不截取任何变量
//   [&} 截取外部作用域中所有变量，并作为引用在函数体中使用
//   [=] 截取外部作用域中所有变量，并拷贝一份在函数体中使用
//   [=, &foo]   截取外部作用域中所有变量，并拷贝一份在函数体中使用，但是对foo变量使用引用
//   [bar]       截取bar变量并且拷贝一份在函数体重使用，同时不截取其他变量
//   [this]      截取当前类中的this指针。如果已经使用了&或者=就默认添加此选项。


// 2. 判断两vector是否相等====================================
static bool is_equals_vector(const std::vector<float>& vec_a, const std::vector<float>& vec_b)
{
	// 首先判断 大小是否一致
	if (vec_a.size() != vec_b.size())
		return false;
        for (size_t i=0; i<vec_a.size(); i++)
	{
		if(vec_a[i] != vec_b[i]) // 浮点数可以这样 判断不相等？？
			return false;
	}
	// 每个元素均相等
 	return true;
}

// 3. 正常的vector相乘(需要关闭编译器的自动向量优化)
// gcc -O3会自动打开 -ftree-vectorize 选项。
// 关闭向量化的选项是 -fno-tree-vectorize 
static void normal_vector_mul(const std::vector<float>& vec_a, 
                              const std::vector<float>& vec_b, 
			      std::vector<float>& vec_result)
{
	// 检查 数组维度
	assert(vec_a.size() == vec_b.size());
	assert(vec_a.size() == vec_result.size());
	
	// 循环遍历相乘  编译器可能会自动 进行向量化优化  添加标志进行关闭 -ftree-vectorize
	for (size_t i=0; i<vec_result.size(); i++)
		vec_result[i] = vec_a[i] * vec_b[i];
}

#ifdef ARM
///*
// 4. neon优化的vector相乘
static void neon_vector_mul(const std::vector<float>& vec_a, 
                            const std::vector<float>& vec_b, 
			    std::vector<float>& vec_result)
{
	// 检查 数组维度
	assert(vec_a.size() == vec_b.size());
	assert(vec_a.size() == vec_result.size());
	
	// noon 寄存器操作
	int i = 0;
        for(; i<vec_result.size()-3; i+=4)// 128位的寄存器，一次会进行四次运算
	{
		const auto data_a = vld1q_f32(&vec_a[i]);// 放入寄存器
		const auto data_b = vld1q_f32(&vec_b[i]);// 放入寄存器
	
		float* dst_ptr = &vec_result[i]; // 结果矩阵 指针
	
		const auto data_res = vmulq_f32(data_a, data_b); // 32为寄存器 浮点数乘法
	
		vst1q_32(dst_ptr, data_res);// 将 寄存器乘法结果 复制到 结果数组中
	}
}
// 上面的代码使用了3条NEON指令：vld1q_f32，vmulq_f32，vst1q_f32
//*/


// 5. 测试函数============================
static int test_neon()
{
// a. 变量设置
	const int test_round = 1000;// 测试总次数
	const int data_len = 100000;// 10万个数据相乘
	std::vector<float> vec_a(data_len);
	std::vector<float> vec_b(data_len);
	std::vector<float> vec_result(data_len);
	std::vector<float> vec_result2(data_len);
// b. 产生随机数据 fill random value in vecA & vecB
	std::cout << "round of test is " << test_round << std::endl;
	std::cout << "size of input vector is "<< data_len << std::endl;
	std::cout << "filling random data to input vector..." << std::endl;
	fill_random_value(vec_a);
	fill_random_value(vec_b);
	std::cout << "fill random data to input vector done.\n" << std::endl;
// c. 确保两种计算方法的结果一致 check the result is same
	{
		normal_vector_mul(vec_a, vec_b, vec_result);
		neon_vector_mul(vec_a, vec_b, vec_result2);
		if (!is_equals_vector(vec_result,vec_result2))
		{
			std::cerr << "result vector is not equals!" << std::endl;
			return -1;
		}
	}
// d. 多次测试普通矩阵乘法 test normal_vector_mul
	{
		FuncCostTimeHelper time_helper("normal_vector_mul");
		for (int i = 0; i < test_round;i++)
		{
			normal_vector_mul(vec_a, vec_b, vec_result);
		}
	}
// e. 多次测试neon寄存器矩阵乘法 test neon_vector_mul
	{
		FuncCostTimeHelper time_helper("neon_vector_mul");
		for (int i = 0; i < test_round; i++)
		{
			neon_vector_mul(vec_a, vec_b, vec_result2);
		}
	}
	return 0;
}
#endif

#ifdef WIN32
static void sse_vector_mul(const std::vector<float>& vec_a, 
                           const std::vector<float>& vec_b, 
			   std::vector<float>& vec_result)
{
	// 检查 数组维度
	assert(vec_a.size() == vec_b.size());
	assert(vec_a.size() == vec_result.size());
	
	// sse 寄存器操作
        int i=0;
        for(; i<vec_result.size()-3; i+=4)
	{
		// 1. load 数据从内存载入暂存器
		__m128  a = _mm_loadu_ps(&vec_a[i]);
	        __m128  b = _mm_loadu_ps(&vec_b[i]);
		__m128  res;

	        float* dst_ptr = &vec_result[i]; // 结果矩阵 指针
                // 2. 进行计算
		res = _mm_mul_ps(a, b); // 32为寄存器 浮点数乘法
	        // 3. 将计算结果 从暂存器 保存到 内存  res ----> dst_ptr
		_mm_storeu_ps(dst_ptr, res);
	}
}
// 5. 测试函数============================
static int test_sse()
{
// a. 变量设置
	const int test_round = 1000;// 测试总次数
	const int data_len = 100000;// 10万个数据相乘
	std::vector<float> vec_a(data_len);
	std::vector<float> vec_b(data_len);
	std::vector<float> vec_result(data_len);
	std::vector<float> vec_result2(data_len);
// b. 产生随机数据 fill random value in vecA & vecB
	std::cout << "round of test is " << test_round << std::endl;
	std::cout << "size of input vector is "<< data_len << std::endl;
	std::cout << "filling random data to input vector..." << std::endl;
	fill_random_value(vec_a);
	fill_random_value(vec_b);
	std::cout << "fill random data to input vector done.\n" << std::endl;
// c. 确保两种计算方法的结果一致 check the result is same
	{
		normal_vector_mul(vec_a, vec_b, vec_result);
		sse_vector_mul(vec_a, vec_b, vec_result2);
		if (!is_equals_vector(vec_result,vec_result2))
		{
			std::cerr << "result vector is not equals!" << std::endl;
			return -1;
		}
	}
// d. 多次测试普通矩阵乘法 test normal_vector_mul
	{
		FuncCostTimeHelper time_helper("normal_vector_mul");
		for (int i = 0; i < test_round;i++)
		{
			normal_vector_mul(vec_a, vec_b, vec_result);
		}
	}
// e. 多次测试neon寄存器矩阵乘法 test neon_vector_mul
	{
		FuncCostTimeHelper time_helper("neon_vector_mul");
		for (int i = 0; i < test_round; i++)
		{
			sse_vector_mul(vec_a, vec_b, vec_result2);
		}
	}
	return 0;
}
#endif


int main(int, char*[])
{
	//return test_neon();
	return test_sse();
}
