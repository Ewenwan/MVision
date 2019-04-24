# ARM_NENO_CNN编程

[参考1 ARM NEON 编程系列](http://hongbomin.com/2016/05/13/arm_neon_introduction/)

[arm官方数据手册](http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.subset.swdev.sdt/index.html)

[Cortex-A Series Programmer’s Guide Version: 4.0](http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.subset.swdev.sdt/index.html)

ARM CPU最开始只有普通的寄存器，可以进行基本数据类型的基本运算。
自ARMv5开始引入了VFP（Vector Floating Point）指令，该指令用于向量化加速浮点运算。
自ARMv7开始正式引入NEON指令，NEON性能远超VFP，因此VFP指令被废弃。

SIMD即单指令多数据指令，目前在x86平台下有MMX/SSE/AVX系列指令，arm平台下有NEON指令。
一般SIMD指令通过intrinsics(内部库C函数接口的函数) 或者 汇编 实现。

类似于Intel CPU下的MMX/SSE/AVX/FMA指令，ARM CPU的NEON指令同样是通过向量化计算来进行速度优化，通常应用于图像处理、音视频处理等等需要大量计算的场景。

> NEON支持的数据类型：

* 32bit  single precision floatingpoint  ， 32bit 单精度浮点数；
* 8, 16, 32 and 64bit unsigned and signed integers ，  8, 16, 32 and 64bit 无符号/有符号 整型；
* 8 and 16bit polynomials 8 and 16bit 多项式。

>NEON数据类型说明符：
* Unsigned integer  无符号整形 U8 U16 U32 U64
* Signed integer    有符号整形 S8 S16 S32 S64
* Integer of unspecified type  未指定类型的整数  I8 I16 I32 I64
Floating point number F16 F32  浮点数 16位浮点数(半精度) 32位浮点数(全精度)
Polynomial over {0,1} P8       多项式

注：F16不适用于数据处理运算，只用于数据转换，仅用于实现半精度体系结构扩展的系统。

多项式算术在实现某些加密、数据完整性算法中非常有用。


> NEON寄存器有几种形式：

* 16×128bit寄存器(Q0-Q15)；  16个128位的寄存器
* 或32×64bit寄存器(D0-D31)   32个64位的寄存器
* 或上述寄存器的组合。

![](https://github.com/Ewenwan/MVision/blob/master/CNN/HighPerformanceComputing/img/neon.PNG)

![](https://github.com/Ewenwan/MVision/blob/master/CNN/HighPerformanceComputing/img/neon-regest.PNG)

一个D寄存器64位是双字宽度，一个Q寄存器是128位是四字宽度。

注：每一个Q0-Q15寄存器映射到 一对D寄存器。

> 寄存器之间的映射关系：

* D<2n> 偶数 映射到 Q 的最低有效半部；
* D<2n+1> 奇数 映射到 Q 的最高有效半部；

> NEON 数据处理指令可分为：

* 1. 正常指令 Normal instructions 结果 同 操作数 同大小同类型。
* 2. 长指令   Long instructions   操作双字vectors，生成四倍长字vectors 结果的宽度一般比操作数加倍，同类型。

     在指令中加L
     
![](https://github.com/Ewenwan/MVision/blob/master/CNN/HighPerformanceComputing/img/long.PNG)
     
* 3. 宽指令   Wide instructions   操作 双字+四倍长字，生成四倍长字，结果和第一个操作数都是第二个操作数的两倍宽度。

     在指令中加W
![](https://github.com/Ewenwan/MVision/blob/master/CNN/HighPerformanceComputing/img/wide.PNG)
     
* 4. 窄指令   Narrow instructions 操作四倍长字，生成双字 结果宽度一般是操作数的一半
     
     在指令中加N
![](https://github.com/Ewenwan/MVision/blob/master/CNN/HighPerformanceComputing/img/narrow.PNG)
     
* 5. 饱和变量 Saturating variants

	对于有符号饱和运算，如果结果小于 –2^n，则返回的结果将为 –2^n；
	对于无符号饱和运算，如果整个结果将是负值，那么返回的结果是 0；如果结果大于 2^n–1，则返回的结果将为 2^n–1；
	NEON中的饱和算法：通过在V和指令助记符之间使用Q前缀可以指定饱和指令，原理与上述内容相同。
     
> **NEON指令集（重点）ARMv7/AArch32指令格式**

所有的支持NEON指令都有一个助记符V，下面以32位指令为例，说明指令的一般格式：

V{<mod模式>}<op操作>{<shape指令类型>}{<cond条件>}{.<dt数据类型>}{<dest目标地址>}, src1, src2
	
> <mod模式> 可选：

	Q: 饱和效果The instruction uses saturating arithmetic, so that the result is saturated within the range of the specified data type, such as VQABS, VQSHLetc.

	H: The instruction will halve the result. It does this by shifting right by one place (effectively a divide by two with truncation), such as VHADD,VHSUB.
	D: 双倍结果 The instruction doubles the result, such as VQDMULL, VQDMLAL, VQDMLSL and VQ{R}DMULH.
	R: 取整 The instruction will perform rounding on the result, equivalent to adding 0.5 to the result before truncating, such as VRHADD, VRSHR.
	
> <op操作>：  必须

the operation (for example, ADD加, SUB减, MUL乘).	

> <shape> shape指令类型 可选：
	
即前文中的Long (L), Wide (W), Narrow (N).

> <cond条件> Condition 可选,
	
	used with IT instruction.
> <.dt> Datatype 可选 数据类型  .数据类型  前面有点
 
	such as .s8, .u8, .f32 , .I16, .S16 etc.
> <dest> Destination. 可选  目标操作数地址

> <src1> Source operand 1. 必须 源操作数地址
> <src2> Source operand 2. 必须 源操作数地址


注: {} 表示可选的参数。

比如：

VADD.I16 D0, D1, D2   @ 16位整数 加法

VMLAL.S16 Q2, D8, D9  @ 有符号16位整数 乘加

> 使用NEON主要有四种方法：

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

[ARM GCC Inline Assembler Cookbook](http://www.ethernut.de/en/documents/arm-inline-asm.html)

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


## 建议的NEON调优步骤

* 1. 理清所需的寄存器、指令。 建议根据要实现的任务，画出数据变换流程，和每步所需的具体指令，尽可能找到最优的实现流程。这一步非常关键，如果思路出错或是不够优化，则会影响使用NEON的效果，并且对程序修改带来麻烦，一定要找到最优的实现算法哦~

* 2. 先实现intrinsics（可选）。 初学者先实现intrinsics是有好处的，字面理解性更强，且有助于理解NEON指令。建议随时打印关键步骤的数据，以检查程序的正误。

* 3. 写成汇编进一步优化。 将intrinsics生成的汇编代码进行优化调整。一般来说，有以下几点值得注意【干货】：

* 只要intrinsics运算指令足够精简，运算类的汇编指令就不用大修；
* 大部分的问题会出在存取、移动指令的滥用、混乱使用上；
* 优化时要尽量减少指令间的相关性，包括结构相关、数据相关控制相关，保证流水线执行效率更高；
* 大概估算所有程序指令取指、执行、写回的总理论时间，以此估算本程序可以优化的空间；
* 熟练对每条指令准备发射、写回时间有一定的认识，有助于对指令的优化排序；
* 一定要多测试不同指令的处理时间！！原因是你所想跟实际有出入，且不同的编译器优化的效果可能也有些不同；
* 一定要有一定的计算机体系结构基础，对存储结构、流水线有一定的体会！！

总结一下NEON优化就是：
* 第一优化算法实现流程；
* 第二优化程序存取；
* 第三优化程序执行；
* 第四哪儿能优化，就优化哪儿


## 内联汇编使用心得
[ARM GCC Inline Assembler Cookbook](http://www.ethernut.de/en/documents/arm-inline-asm.html)

inline assembly下面的三个冒号一定要注意
output/input registers的写法一定要写对，clobber list也一定要写完全，否则会造成令你头疼的问题 (TT)

这个问题在给出的cookbook中也有介绍，但是并不全面，有些问题只有自己碰到了再去解决。 笔者就曾经被虐了很久，从生成的汇编发现编译器将寄存器乱用，导致指针操作完全混乱，毫无头绪…


一般情况下建议的写法举例：
```asm
asm volatile (
	... /* assembly code */
	: "+r"(arg0) // %0
	  "+r"(arg1) // %1 // 输入寄存器 Output Registers
	: "r"(arg2)  // %2 // 输入寄存器 Input Registers
	: "cc", "memory", r0, r1  // 寄存器变化
);
```


