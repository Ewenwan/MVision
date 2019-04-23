# ARM_NENO_CNN编程

[参考1 ARM NEON 编程系列](http://hongbomin.com/2016/05/13/arm_neon_introduction/)


ARM CPU最开始只有普通的寄存器，可以进行基本数据类型的基本运算。
自ARMv5开始引入了VFP（Vector Floating Point）指令，该指令用于向量化加速浮点运算。
自ARMv7开始正式引入NEON指令，NEON性能远超VFP，因此VFP指令被废弃。

SIMD即单指令多数据指令，目前在x86平台下有MMX/SSE/AVX系列指令，arm平台下有NEON指令。
一般SIMD指令通过intrinsics(内部库C函数接口的函数) 或者 汇编 实现。

类似于Intel CPU下的MMX/SSE/AVX/FMA指令，ARM CPU的NEON指令同样是通过向量化计算来进行速度优化，通常应用于图像处理、音视频处理等等需要大量计算的场景。

使用NEON主要有四种方法：
* 1. NEON优化库(Optimized libraries)
* 2. 向量化编译器(Vectorizing compilers)
* 3. NEON intrinsics
* 4. NEON assembly

根据优化程度需求不同，第4种最为底层，若熟练掌握效果最佳，一般也会配合第3种一起使用。

1. 优化库 Libraries：直接在程序中调用优化库

  OpenMax DL：支持加速视频编解码、信号处理、色彩空间转换等；
  
  Ne10：一个ARM的开源项目，提供数学运算、图像处理、FFT函数等。
  
2. 向量化编译 Vectorizing compilers：GCC编译器的向量优化选项

在GCC选项中加入向量化表示能有助于C代码生成NEON代码，如‐ftree‐vectorize。


3. NEON intrinsics：提供了一个连接NEON操作的C函数接口，编译器会自动生成相关的NEON指令，支持ARMv7或ARMv8平台。

[所有的intrinsics函数都在GNU官方说明文档 ](https://gcc.gnu.org/onlinedocs/gcc-4.7.4/gcc/ARM-NEON-Intrinsics.html#ARM-NEON-Intrinsics)

## 示例1：向量加法
```neon
// 假设 count 是4的倍数
#include<arm_neon.h>

// C version
void add_int_c(int* dst, int* src1, int* src2, int count)
{
	int i;
	for (i = 0; i < count; i++)
		{
		    dst[i] = src1[i] + src2[i];
		}
}

// NEON version
void add_float_neon1(int* dst, 
                     int* src1, 
		     int* src2, // 传入三个数据单元的指针（地址）
		     int count) // 数据量 假设为4的倍数
{
	int i;
	for (i = 0; i < count; i += 4) // 寄存器操作每次 进行4个数据的运输（单指令多数据SIMD）
	{
		int32x4_t in1, in2, out;
		
		// 1. 从内存 载入 数据 到寄存器
		in1 = vld1q_s32(src1);// intrinsics传入的为内存数据指针
		src1 += 4;// 数据 指针 递增+4 
		
		in2 = vld1q_s32(src2);
		src2 += 4;
		
		// 2. 在寄存器中进行数据运算 加法add
		out = vaddq_s32(in1, in2);
		
		// 3. 将寄存器中的结果 保存到 内存地址中
		vst1q_s32(dst, out);
		dst += 4;// 
	}
	// 实际情况，需要做最后不够4个的数的运输，使用普通c函数部分进行
	// 可参考下面的代码进行改进
}


``` 

代码中的 vld1q_s32 会被编译器转换成 vld1.32 {d0, d1}, [r0] 指令，

同理 vaddq_s32 被转换成 vadd.i32 q0, q0, q0，

 vst1q_s32 被转换成 vst1.32 {d0,d1}, [r0]。



## 示例2：向量乘法 

```neon
//NRON优化的vector相乘
static void neon_vector_mul(
  const std::vector<float>& vec_a, // 向量a 常量引用
  const std::vector<float>& vec_b, // 向量b 常量引用 
  std::vector<float>& vec_result)  // 结果向量 引用
{
	assert(vec_a.size() == vec_b.size());
	assert(vec_a.size() == vec_result.size());
	int i = 0;// 向量索引 从0开始
  
	//neon process
	for (; i < (int)vec_result.size() - 3 ; i+=4)// 每一步会并行执行四个数(单指令多数据simd) 注意每次增加4
	{// 不够 4的部分留在后面用 普通 c代码运算
               // 从内存载入数据到寄存器
		const auto data_a = vld1q_f32(&vec_a[i]);// 函数传入的是 地址（指针）
		const auto data_b = vld1q_f32(&vec_b[i]);
    
		float* dst_ptr = &vec_result[i];// 结果向量的地址(内存中)
    
                // 在寄存器中进行运算，乘法 mulp 运算
		const auto data_res = vmulq_f32(data_a, data_b);
    
                // 将处于寄存器中的结果 保存传输到 内存中国
		vst1q_f32(dst_ptr, data_res);
	}
  
	// normal process 普通C代码 数据相乘= 剩余不够4个数的部分===可能为 1,2,3个数
	for (; i < (int)vec_result.size(); i++)
	{
		vec_result[i] = vec_a[i] * vec_b[i];
	}
}

```

4. NEON assembly

NEON可以有两种写法：
* 1. Assembly文件： 纯汇编文件，后缀为”.S”或”.s”。注意对寄存器数据的保存。
* 2. inline assembly内联汇编

优点：在C代码中嵌入汇编，调用简单，无需手动存储寄存器；
缺点：有较为复杂的格式需要事先学习，不好移植到其他语言环境。


比如上述intrinsics代码产生的汇编代码为：
```c
// ARMv7‐A/AArch32
void add_float_neon2(int* dst, int* src1, int* src2, int count)
{
	asm volatile (
		"1: \n"
		"vld1.32 {q0}, [%[src1]]! \n"
		"vld1.32 {q1}, [%[src2]]! \n"
		"vadd.f32 q0, q0, q1 \n"
		"subs %[count], %[count], #4 \n"
		"vst1.32 {q0}, [%[dst]]! \n"
		"bgt 1b \n"
		: [dst] "+r" (dst)
		: [src1] "r" (src1), [src2] "r" (src2), [count] "r" (count)
		: "memory", "q0", "q1"
	);
}

```



