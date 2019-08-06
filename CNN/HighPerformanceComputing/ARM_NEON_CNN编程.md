# ARM_NEON_CNN编程

内联函数优化的越来越好了，甚至在ARMv8 平台下有优于汇编的性能，同时兼容性方面又比汇编好，因此使用内联函数是上上之选。
毕竟，NEON肯定会更新的，到时一更新你的底层汇编得全部跟着更新，但是使用内联函数的话就不要考虑这些了，反正编译器都帮我们做了嘛！
最后关于内联函数告诉后辈们几点人生经验：

使用的寄存器数量要考虑周全；
编译器注意好啊！
一定要看看产生的汇编代码啊！

[图像算法的工程优化技术: 算法流程优化 CPU多线程 SIMD GPU编程 专用芯片](https://blog.csdn.net/jxt1234and2010/article/details/50768263)

[AI 移动端框架常用指令·汇总 v7 v8 差异](https://www.jianshu.com/p/5f75fa02c5d0)

[什么？！NEON还要优化？](https://www.jianshu.com/p/16d60ac56249)

[神经网络arm neon加速实现](https://blog.csdn.net/fuwenyan/article/details/78793907)

[常用NEON 内置函数记录备用](https://blog.csdn.net/fuwenyan/article/details/78811034)

[ARM Cortex系列(A8/A9/A15/A7) NEON多媒体处理SIMD引擎优化](https://blog.csdn.net/yxnyxnyxnyxnyxn/article/details/18267955)

[aarch64 armv8 neon intrinsics 和内嵌汇编混用](https://github.com/Tencent/ncnn/wiki/aarch64-neon-intrinsics-%E5%92%8C%E5%86%85%E5%B5%8C%E6%B1%87%E7%BC%96%E6%B7%B7%E7%94%A8)

[32位 armv7 neon intrinsics 和内嵌汇编混用](https://github.com/Tencent/ncnn/wiki/armv7-neon-intrinsics-%E5%92%8C%E5%86%85%E5%B5%8C%E6%B1%87%E7%BC%96%E6%B7%B7%E7%94%A8)

[ARM NEON 社区](https://community.arm.com/cn/f/tags/NEON)

[ARM平台NEON指令的编译和优化  编译选项](https://blog.csdn.net/heli200482128/article/details/79303286)

[程序优化方法经验大全——神文](https://blog.csdn.net/STN_LCD/article/details/77606256)

> 术语： 

System-on-Chip(SOC) 片上系统：核心、内存控制器、片上内存、外围设备、总线互连和其他逻辑（可能包括模拟或射频组件），以便产生系统。 SOC通常指集成度较高的设备，包括单个设备中系统的许多部分，可能包括模拟、混合信号或射频电路。

专用集成电路Application Specific Integrated Circuit(ASIC) :包含ARM内核、内存和其他组件。显然，ASIC和SOC之间有很大的重叠。

嵌入式系统 Embedded systems，
内存消耗 Memory Footprint(memory usage),
SIMD(Single Instruction, Multiple Data) 单指令多数据流，
MMU(Memory Management Unit) 内存管理单元，
MPE(Media Processing Engine) 媒体处理引擎。
VFP(Vector Floating Point) 向量浮点

[参考1 ARM NEON 编程系列](http://hongbomin.com/2016/05/13/arm_neon_introduction/)

[arm官方数据手册](http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.subset.swdev.sdt/index.html)

[Cortex-A Series Programmer’s Guide Version: 4.0](http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.subset.swdev.sdt/index.html)

ARM CPU最开始只有普通的寄存器，可以进行基本数据类型的基本运算。
自ARMv5开始引入了VFP（Vector Floating Point）指令，该指令用于向量化加速浮点运算。
自ARMv7开始正式引入NEON指令，NEON性能远超VFP，因此VFP指令被废弃。

SIMD即单指令多数据指令，目前在x86平台下有MMX/SSE/AVX系列指令，arm平台下有NEON指令。
一般SIMD指令通过intrinsics(内部库C函数接口的函数) 或者 汇编 实现。

Intrinsics(内联函数)是使用C语言的方式对NEON寄存器进行操作，因为相比于传统的使用纯汇编语言，具有可读性强，开发速度快等优势。如果需要在代码中调用NEON Intrinsics函数，需要加入头文件"arm_neon.h"。

NEON C内联函数（intrinsics）是由ARM定义的一组全新的数据类型和内联函数，便于使用C语言直接访问NEON单元。在C/C++程序中，内联函数就同普通函数一样，但在编译时，这些内联函数会直接映射为NEON提供的向量指令。当前GCC编译器和ARM编译器都支持相同的NEON内联语法，只需在程序中添加“arm_neon.h”头文件，就可以使用NEON内联函数。

[ARM NEON常用 intrinsics 函数总结 !!!!](https://blog.csdn.net/may0324/article/details/72847800)

**优势**：使用内联函数进行优化，开发人员无需关注寄存器分配和互锁等问题，这些都交由编译器处理，而且编写程序比较容易，优化后的性能相对较高。

**不足**：目前内联函数所提供的功能和灵活性仍远远比不上汇编指令，并且经过编译器编译后，会反复加载／存取寄存器数据，导致系统时钟的浪费。 


采用汇编语言进行NEON(**NEON 汇编（assembly）**)的最底层优化，可以使优化性能最大化，但汇编语言比较灵活，手写汇编程序对开发人员来说具有较大挑战，如果使用不恰当，反而会影响优化性能。


![](https://github.com/Ewenwan/MVision/blob/master/CNN/HighPerformanceComputing/img/simd.PNG)

![](https://github.com/Ewenwan/MVision/blob/master/CNN/HighPerformanceComputing/img/simd_add-op.PNG)

![](https://upload-images.jianshu.io/upload_images/3270173-633789004154255f.gif?imageMogr2/auto-orient/strip%7CimageView2/2/w/634/format/webp)

在这里，一条SIMD加法指令可以同时得到8个加法结果。就计算步骤本身而言，比单独使用8条加法指令能够获得8倍的加速比。从该示例也可以看出，随着寄存器长度的变长，单指令能够处理的数据量也越来越大，从而获得更高的加速性能。
在Intel最新的AVX2指令集中，寄存器最大长度已经达到512位。

类似于Intel CPU下的MMX/SSE/AVX/FMA指令，ARM CPU的NEON指令同样是通过向量化计算来进行速度优化，通常应用于图像处理、音视频处理等等需要大量计算的场景。

> **SISD(Single Instruction Single Data)单指令单数据**
```asm
add r0, r5  # 单条指令执行一个运算
add r1, r6
add r2, r7
add r3, r8
```
> **SIMD(Single Instruction Multiple Data (vector mode向量模式))单指令多数据**
```c
VADD.F32 S24, S8, S16 
// four operations occur 单条指令并行执行四个运算
// S24 = S8 +S16
// S25 = S9 +S17
// S26 = S10 +S18
// S27 = S11 +S20

```

> **SIMD(Single Instruction Multiple Data (packed data mode)包数据模式)**
```c
VADD.I16 Q10, Q8, Q9
// One operation adds two 64-bit registers, 128位寄存器
// but each of the four 16-bit lanes in the register is added separately.
// 单个数据为16位，所以有8个数据并行计算加法运算
```

> NEON支持的数据类型：

* 32bit  single precision floatingpoint  ， 32bit 单精度浮点数；
* 8, 16, 32 and 64bit unsigned and signed integers ，  8, 16, 32 and 64bit 无符号/有符号 整型；
* 8 and 16bit polynomials 8 and 16bit 多项式。


	B字节Byte：      8 bits.
	H半字Halfword：  16 bits.   半精度浮点16位
	S字Word：        32 bits.   单精度浮点32位
	D双字Doubleword：64 bits.   双精度浮点64位
	Q四字Quadword：  128 bits.

> 浮点数取整:

向负无穷取整(向左取整) Round towards Minus Infinity (RM) roundTowardsNegative

向正无穷取整(向右取整) Round towards Plus Infinity (RP) roundTowardsPositive

向零取整(向中间取整)Round towards Zero (RZ) roundTowardZero

就近取整 Round to Nearest (RN) roundTiesToEven

随机取整

>NEON数据类型说明符：

* Unsigned integer  无符号整形 U8 U16 U32 U64
* Signed integer    有符号整形 S8 S16 S32 S64
* Integer of unspecified type  未指定类型的整数  I8 I16 I32 I64
Floating point number F16 F32  浮点数 16位浮点数(半精度) 32位浮点数(全精度)
Polynomial over {0,1} P8       多项式

注：F16不适用于数据处理运算，只用于数据转换，仅用于实现半精度体系结构扩展的系统。

多项式算术在实现某些加密、数据完整性算法中非常有用。

寄存器 ARMV7架构包含：

16个通用寄存器（32bit），R0-R15 register

16个NEON寄存器（128bit），Q0-Q15 quad四字寄存器（同时也可以被视为32个64bit的寄存器，D0-D31 double双字寄存器）

16个VFP寄存器（32bit），S0-S15，single 单字寄存器

NEON和VFP的区别在于VFP是加速浮点计算的硬件不具备数据并行能力，同时VFP更尽兴双精度浮点数（double）的计算，NEON只有单精度浮点计算能力。

16个通用寄存器

![](https://github.com/Ewenwan/MVision/blob/master/CNN/HighPerformanceComputing/img/register.PNG)

寄存器 r0 到 r7 称为低位寄存器。 寄存器 r8 到r15 称为高位寄存器。

下列寄存器名称是预先声明的：

* r0-r15 和 R0-R15
* a1-a4（自变量、结果或暂存寄存器，r0 到 r3 的同义词）
* v1-v8（变量寄存器，r4 到 r11）
* sb 和 SB（静态基址，r9）
* ip 和 IP（内部程序调用暂存寄存器，r12）
* sp 和 SP（堆栈指针，r13）
* lr 和 LR（链接寄存器，r14）
* pc 和 PC（程序计数器，r15）。

> NEON寄存器有几种形式：

* 16×128bit寄存器(Q0-Q15)；  16个128位的寄存器
* 或32×64bit寄存器(D0-D31)   32个64位的寄存器
* 或上述寄存器的组合。

以下扩展寄存器名称是预先声明的：

* q0-q15 和 Q0-Q15（NEON™ 四字寄存器）
* d0-d31 和 D0-D31（NEON 双字寄存器，VFP 双精度寄存器）
* s0-s31 和 S0-S31（VFP 单精度寄存器）。

![](https://github.com/Ewenwan/MVision/blob/master/CNN/HighPerformanceComputing/img/neon.PNG)

![](https://github.com/Ewenwan/MVision/blob/master/CNN/HighPerformanceComputing/img/neon-regest.PNG)

一个D寄存器64位是双字宽度，一个Q寄存器是128位是四字宽度。

注：每一个Q0-Q15寄存器映射到 一对D寄存器。

> 寄存器之间的映射关系：

* D<2n> 偶数 映射到 Q 的最低有效半部；
* D<2n+1> 奇数 映射到 Q 的最高有效半部；
* S<2n> 映射到 D<n> 的最低有效半部
* S<2n+1> 映射到 D<n> 的最高有效半部
	
例如，通过引用 D12 可以访问 Q6 中向量元素的最低有效半部，通过引用 D13 可以访问这些元素的最高有效半部。
	
![](https://github.com/Ewenwan/MVision/blob/master/CNN/HighPerformanceComputing/img/extern-regest.PNG)	
	
## 指令集概述
所有 ARM 指令的长度都是 32 位。 这些指令是按字对齐方式存储的，因此在ARM 状态下，指令地址的两个最低有效位始终为零。

> **跳转指令**，此类指令用于：

* 1.向后跳转以构成循环
* 2.在条件结构中向前跳转
* 3.跳转到子例程
* 4.在 ARM 状态和 Thumb 状态之间转换处理器状态。

> **寄存器加载和存储指令**

此类指令用于从内存加载单个寄存器的值，或者在内存中存储单个寄存器的值。它们可加载或存储 32 位字、16 位半字或 8 位无符号字节。 可以用符号或零扩展字节和半字加载以填充 32 位寄存器。此外，还定义了几个可将 64 位双字值加载或存储到两个 32 位寄存器的指令。

> **数据处理指令**

此类指令用于对通用寄存器执行运算。 它们可对两个寄存器的内容执行加法、减法或按位逻辑等运算，并将结果存放到第三个寄存器中。 此外，它们还可以对单个寄存器中的值执行运算，或者对寄存器中的值与指令中提供的常数（立即值）执行运算。

> NEON 数据处理指令可分为：

* 1. 正常指令 Normal instructions 结果 同 操作数 同大小同类型。

     生成大小相同且类型通常与操作数向量相同到结果向量。
     
     正常指令可对上述任意向量类型执行运算，并生成大小相同且类型通常与操作数向量相同的结果向量。

     **通过将 Q 附加到指令助记符，可以指定正常指令的操作数和结果必须全部为四字。** 

     这样指定后，如果操作数或结果不是四字，则汇编程序会生成错误。


* 2. 长指令   Long instructions   操作双字vectors，生成四倍长字vectors 结果的宽度一般比操作数加倍，同类型。

     在指令中加L
     
     长指令对双字向量操作数执行运算，并生成四字向量结果。 所生成的元素通常是操作数元素宽度的两倍，并属于同一类型。通过将 L 追加到指令助记符来指定长指令。
     
     对双字向量操作数执行运算，生成四字向量到结果。所生成的元素一般是操作数元素宽度到两倍，并属于同一类型。L标记，如VMOVL。
     
![](https://github.com/Ewenwan/MVision/blob/master/CNN/HighPerformanceComputing/img/long.PNG)
     
* 3. 宽指令   Wide instructions   操作 双字+四倍长字，生成四倍长字，结果和第一个操作数都是第二个操作数的两倍宽度。

     在指令中加W
     
     一个双字向量操作数和一个四字向量操作数执行运算，生成四字向量结果。W标记，如VADDW。
     
     宽指令对一个双字向量操作数和一个四字向量操作数执行运算。 此类指令生成四字向量结果。 所生成的元素和第一个操作数的元素是第二个操作数元素宽度的两倍。
     
     通过将 W 追加到指令助记符来指定宽指令。

![](https://github.com/Ewenwan/MVision/blob/master/CNN/HighPerformanceComputing/img/wide.PNG)
     
* 4. 窄指令   Narrow instructions 操作四倍长字，生成双字 结果宽度一般是操作数的一半
     
     在指令中加N
     
     四字向量操作数执行运算，并生成双字向量结果，所生成的元素一般是操作数元素宽度的一半。N标记，如VMOVN。
     
     窄指令对四字向量操作数执行运算，并生成双字向量结果。 所生成的元素通常是操作数元素宽度的一半。
     
     通过将 N 追加到指令助记符来指定窄指令。
     
![](https://github.com/Ewenwan/MVision/blob/master/CNN/HighPerformanceComputing/img/narrow.PNG)
     
* 5. 饱和指令 Saturating variants

        通过在 V 和指令助记符之间使用 Q 前缀来指定饱和指令。
	
	对于有符号饱和运算，如果结果小于 –2^n，则返回的结果将为 –2^n；
	 
	对于无符号饱和运算，如果整个结果将是负值，那么返回的结果是 0；如果结果大于 2^n–1，则返回的结果将为 2^n–1；
	
	NEON中的饱和算法：通过在V和指令助记符之间使用Q前缀可以指定饱和指令，原理与上述内容相同。
        
	饱和指令：当超过数据类型指定到范围则自动限制在该范围内。Q标记，如VQSHRUN

![](https://github.com/Ewenwan/MVision/blob/master/CNN/HighPerformanceComputing/img/data-range.PNG)

数据类型 x 的饱和范围 (s 就是signed，有符号的意思，u就是unsigned，无符号的意思） 

	s8   –2^7  <= x < 2^7 
	s16  –2^15 <= x < 2^15 
	s32  –2^31 <= x < 2^31 
	s64  –2^63 <= x < 2^63 
	u8   0     <= x < 2^8 
	u16  0     <= x < 2^16 
	u32  0     <= x < 2^32 
	u64  0     <= x < 2^64


> **NEON指令集（重点）ARMv7/AArch32指令格式**

所有的支持NEON指令都有一个助记符V，下面以32位指令为例，说明指令的一般格式：

V{<mod模式>}<op操作>{<shape指令类型>}{<cond条件>}{.<dt数据类型>}{<dest目标地址>}, src1, src2

> <mod模式> 可选：

	Q: Staturating饱和结果，The instruction uses saturating arithmetic, so that the result is saturated within the range of the specified data type, such as VQABS, VQSHLetc.
        
	VQADD.S16 D0, D2, D3
	
	H: Halving，半结果，结果右移动移位，相当于得到结构后在除以2 The instruction will halve the result. It does this by shifting right by one place (effectively a divide by two with truncation), such as VHADD,VHSUB.
	
	VHADD.S16 Q0, Q1, Q4
	
	D: Doubling，双倍结果 The instruction doubles the result, such as VQDMULL, VQDMLAL, VQDMLSL and VQ{R}DMULH.
	
	VQDMULL.S16 Q0, D1, D3   双倍+饱和+长指令
	
	
	R: Rounding，取整 The instruction will perform rounding on the result, equivalent to adding 0.5 to the result before truncating, such as VRHADD, VRSHR.
	
	VRSUBHN.I16 D0, Q1, Q3
	
	
> <op操作>：  必须

the operation (for example, ADD加, SUB减, MUL乘).	

NEON指令按照作用可以分为：加载数据、存储数据、加减乘除运算、逻辑AND/OR/XOR运算、比较大小运算

> <shape> shape指令类型 可选：
	
即前文中的Long (L长指令，结果数据位扩大), Wide (W), Narrow (N结果数据位变窄).

> <cond条件> Condition 可选,
	
	used with IT instruction.
> <.dt> Datatype 可选 数据类型  .数据类型  前面有点
 
	such as .s8, .u8, .f32 , .I16, .S16 etc.
	
![](https://github.com/Ewenwan/MVision/blob/master/CNN/HighPerformanceComputing/img/dtypr.PNG)	
	
	
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

## 3. NEON Instrinsic函数

NEON Instrinsic是编译器支持的一种buildin类型和函数的集合，基本涵盖NEON的所有指令，通常这些Instrinsic包含在arm_neon.h头文件中。

[ARM-NEON-Intrinsics](https://gcc.gnu.org/onlinedocs/gcc-4.6.1/gcc/ARM-NEON-Intrinsics.html)

[使用ARM NEON Intrinsics加速Video Codec 参考](https://www.jianshu.com/p/70601b36540f)

### 数据类型

NEON 向量数据类型是根据以下模式命名的：<type类型><size大小宽度>x<number of lanes通道数量>_t

例如，int16x4_t 是一个包含四条向量线的向量，每条向量线包含一个有符号 16位整数。

NEON Intrinsics内置的整数数据类型主要包括以下几种:

* (u)int8x8_t;
* (u)int8x16_t;
* (u)int16x4_t;
* (u)int16x8_t;
* (u)int32x2_t;
* (u)int32x4_t;
* (u)int64x1_t;

其中，第一个数字代表的是数据类型宽度为8/16/32/64位，第二个数字代表的是一个寄存器中该类型数据的数量。如int16x8_t代表16位有符号数，寄存器中共有8个数据。

某些内在函数使用以下格式的向量类型数组：

<type><size>x<number of lanes>x<length of array>_t
	
这些类型被视为包含名为 val 的单个元素的普通 C 结构。

以下是一个结构定义示例：
```c
struct int16x4x2_t
{
int16x4_t val[2];
};
```

标号和具体类型转换：

	标记  双字64位D寄存器    四字128位寄存器
	s8    int8x8_t           int8x16_t        有符号整数
	s16   int16x4_t          int16x8_t
	s32   int32x2_t          int32x4_t
	s64   int64x1_t          int64x2_t 
	u8    uint8x8_t          uint8x16_t       无符号整数
	u16   uint16x4_t         uint16x8_t
	u32   uint32x2_t         uint32x4_t
	u64   uint64x1_t         uint64x2_t
	f16   float16x4_t        float16x8_t      浮点数
	f32   float32x2_t        float32x4_t
	p8    poly8x8_t          poly8x16_t       多项式数
	p16   poly16x4_t         poly16x8_t

vcombine_type()  连接组合函数 结果类型长度加倍

vget_high_type() 获取高位     结果类型长度减半

vget_low_type()  获取低位     结果类型长度减半

长指令类型 结果类型长度加倍

窄指令类型 结果类型长度减半



### 内在函数 inline function
每个内在函数的格式如下：

<opname><flags>_<type>
	
另外提供 q 标记来指定内在函数对 128 位向量进行运算。

例如：

* vmul_s16，表示两个有符号 16 位值的向量相乘multiply。
这编译为 VMUL.I16 d2, d0, d1。

* vaddl_u8，l为long长指令标识，是指两个包含无符号 8 位值的 64 位向量按长型相加，结果为无符号 16 位值的 128 位向量。
这编译为 VADDL.U8 q1, d0, d1。

* int8_t vget_lane_s8 (int8x8_t __a, const int __b); 

v是向量操作，可以认为就是neon函数，shr是右移位，lane表示操作向量中的某个元素，s8表示结果是s8类型（向量） 

* int8x8_t vget_high_s8 (int8x16_t __a); //ri = a(i+4); 

v是向量操作，可以认为就是neon函数，get是取值，high表示取高64位，s8表示结果是s8类型（向量） 

* int8x8_t vget_low_s8 (int8x16_t __a); //ri = ai; 

v是向量操作，可以认为就是neon函数，get是取值，low表示取低64为，s8表示结果是s8类型（向量）

     v<noen函数前缀>q<饱和操作>ops<具体操作>tyep<指令类型  q,l,w,n>_flag<标识  n,lane,high or low>_dtype<返回值类型或参数类型>
     
	add 加法 
	mul 乘法 
	sub 减法 
	mla 乘加 
	mls 乘减 
	ceq 比较，类似与 == 
	cge 比较，类似与 >= 
	cle 比较，类似与 <= 
	cgt 比较，类似与 > 
	clt 比较，类似与 < 
	tst 做与运算后，判断是否等于0 ,ri = (ai & bi != 0) ? 1…1:0…0; 
	abd 两个向量相减后的绝对值，vabd -> ri = |ai - bi|; 
	max 求最大值，ri = ai >= bi ? ai : bi; 
	min 求最小值，ri = ai >= bi ? bi : ai; 
	shl 左移位， ri = ai << b; 
	shr 右移位， ri = ai >> b; 
	abs 求绝对值，ri = |ai|; 
	neg 取反，ri = -ai; 
	mvn 按位取反，ri = ~ai; 
	and 与运算，ri = ai & bi; 
	orr 或运算，ri = ai | bi; 
	eor 异或运算，ri = ai ^ bi; 
	cls 计算连续相同的位数 
	get 取值，从向量中取出一个值，所谓的向量可以认为是一个数组，给数组中的某个元素赋值 
	set 赋值，给向量中赋值 
	dup 构造一个向量，并赋上初始值，ri = a; 
	combine 合并操作，把两个向量合并 
	mov 改变数据类型，数据范围，比如把u8 变成u16，或者u16变成u8 
	zip 压缩操作 
	uzp 解压操作 
	ld1 加载数据，给定的buffer 指针中拷贝数据，注意是ld后面的是数字1，而不是字母l 
	st1 拷贝数据，将neon数据类型拷贝到指定buffer中


〉**示例函数指令分析**
```c
int16x8_t vqaddq_s16 (int16x8_t, int16x8_t)
int16x4_t vqadd_s16 (int16x4_t, int16x4_t)
```

* 第一个字母'v'指明是vector向量指令，也就是NEON指令；
* 第二个字母'q'指明是饱和指令，即后续的加法结果会自动饱和；
* 第三个字段'add'指明是加法指令；
* 第四个字段'q'指明操作寄存器宽度，为'q'时操作QWORD, 为128位；未指明时操作寄存器为DWORD，为64位；
* 第五个字段's16'指明操作的基本单元为有符号16位整数，其最大表示范围为-32768 ~ 32767；
* 第六个字段为空，普通指令，形参和返回值类型约定与C语言一致。

其它可能用到的助记符包括:

* l 长指令，数据扩展，双字运算得到四字结果
* w 宽指令，数据对齐，双字和四字运算得到四字结果
* n 窄指令, 数据压缩，四字运算得到双字结果

> 示例2
```c
uint8x8_t vld1_u8 (const uint8_t *)
```
* 第一个字母'v'指明是vector向量指令，也就是NEON指令；
* 第二个字段'ld'表示加载指令 load
* 第三个字段'1'(注意是1，不是l)表示顺次加载。如果需要处理图像的RGB分量，可能会用到vld3间隔3个单元加载。


NEON指令按照作用可以分为：加载数据、存储数据、加减乘除运算、逻辑AND/OR/XOR运算、比较大小运算

> **初始化寄存器**
```c
// 寄存器的每个lane（通道）都赋值为一个值N
Result_t vcreate_type(Scalar_t N)   // type需要换成具体类型 s8, u8, f32, I16, S16
Result_t vdup_type(Scalar_t N)      // vcreate_s8  vdup_s8   vmov_s8
Result_t vmov_type(Scalar_t N)
```
> **加载load 内存数据进寄存器**
```c
// 间隔为x，加载数据进NEON寄存器, 间隔：交叉存取，是ARM NEON特有的指令
Result_t vld[x]_type(Scalar_t* N)  // 
Result_t vld[x]q_type(Scalar_t* N) // vld1q_s32 间隔1 即连续内存访问， 

// **通过将 Q 附加到指令助记符，可以指定正常指令的操作数和结果必须全部为四字。** 

float32x4x3_t = vld3q_f32(float32_t* ptr)
// 此处间隔为3，即交叉读取12个float32进3个NEON寄存器中。
// 3个寄存器的值分别为：
// {ptr[0],ptr[3],ptr[6],ptr[9]}，   // 128为Q寄存器
// {ptr[1],ptr[4],ptr[7],ptr[10]}，
// {ptr[2],ptr[5],ptr[8],ptr[11]}。
```

* 1. VLD1是最简单的形式，从内存加载1~4个寄存器的数据，没有deinterleave，即线性加载；

* 2. VLD2加载2或者4个寄存器的数据，解交织奇偶元素到各自的寄存器，这样很容易的把交织的立体声音频数据分解为左右声道的数据；

* 3. VLD3加载3个寄存器的数据，很方便的把RGB的数据分为R、G、B通道；

* 4. VLD4加载4个寄存器的数据，解交织，用于分解ARGB图像数据；


> **存储set 寄存器数据到内存   间隔为x，存储NEON寄存器的数据到内存中**
```cpp
void vst[x]_type(Scalar_t* N)
void vst[x]q_type(Scalar_t* N)
```

> **算数运算指令**

[普通指令]  普通加法运算 res = M+N
```c
Result_t vadd_type(Vector_t M,Vector_t N)
Result_t vaddq_type(Vector_t M,Vector_t N)

```
[长指令 long] 变长加法运算 res = M+N

为了防止溢出，一种做法是使用如下指令，加法结果存储到长度x2的寄存器中，

如：
```c

Result_t vaddl_type(Vector_t M,Vector_t N)

vuint16x8_t res = vaddl_u8(uint8x8_t M,uint8x8_t N)
```

[宽指令] 加法运算 res = M+N，第一个参数M宽度大于第二个参数N。
```c
Result_t vaddw_type(Vector_t M,Vector_t N)
```

[普通指令] 减法运算 res = M-N
```c
Result_t vsub_type(Vector_t M,Vector_t N)
```

[普通指令] 乘法运算 res = M*N
```c
Result_t vmul_type(Vector_t M,Vector_t N)
Result_t vmulq_type(Vector_t M,Vector_t N)
```

[普通指令] 乘&加法运算 res = M + N*P
```c
Result_t vmla_type(Vector_t M,Vector_t N,Vector_t P)
Result_t vmlaq_type(Vector_t M,Vector_t N,Vector_t P)
```

乘&减法运算 res = M-N*P
```c
Result_t vmls_type(Vector_t M,Vector_t N,Vector_t P)
Result_t vmlsq_type(Vector_t M,Vector_t N,Vector_t P)
```

> **数据处理指令**

[普通指令] 计算绝对值 res=abs(M)
```c
Result_t vabs_type(Vector_t M)
```
[普通指令] 计算负值 res=-M   negative
```c
Result_t vneg_type(Vector_t M)
```
[普通指令] 计算最大值 res=max(M,N)   maxmum
```c
Result_t vmax_type(Vector_t M,Vector_t N)
```
[普通指令] 计算最小值 res=min(M,N)
```c
Result_t vmin_type(Vector_t M,Vector_t N)
```

> **比较指令**

[普通指令] 比较是否相等 res=mask(M == N)  compare equal
```c
Result_t vceg_type(Vector_t M,Vector_t N)
```
[普通指令] 比较是否大于或等于 res=mask(M >= N)  compare greate and  equal
```c
Result_t vcge_type(Vector_t M,Vector_t N)
```
[普通指令] 比较是否大于 res=mask(M > N)
```c
Result_t vcgt_type(Vector_t M,Vector_t N)
```
[普通指令] 比较是否小于或等于 res=mask(M <= N)  compare little  and equal
```c
Result_t vcle_type(Vector_t M,Vector_t N)
```
[普通指令] 比较是否小于 res=mask(M < N)        compare little 
```c
Result_t vclt_type(Vector_t M,Vector_t N)
```
####  向量加法：

> **正常向量加法 vadd -> Vr[i]:=Va[i]+Vb[i]**
Vr、Va、Vb 具有相等的向量线大小。
```c
//64位==
int8x8_t vadd_s8(int8x8_t a, int8x8_t b); // VADD.I8 d0,d0,d0
int16x4_t vadd_s16(int16x4_t a, int16x4_t b); // VADD.I16 d0,d0,d0
int32x2_t vadd_s32(int32x2_t a, int32x2_t b); // VADD.I32 d0,d0,d0
int64x1_t vadd_s64(int64x1_t a, int64x1_t b); // VADD.I64 d0,d0,d0
float32x2_t vadd_f32(float32x2_t a, float32x2_t b); // VADD.F32 d0,d0,d0
uint8x8_t vadd_u8(uint8x8_t a, uint8x8_t b); // VADD.I8 d0,d0,d0
uint16x4_t vadd_u16(uint16x4_t a, uint16x4_t b); // VADD.I16 d0,d0,d0
uint32x2_t vadd_u32(uint32x2_t a, uint32x2_t b); // VADD.I32 d0,d0,d0
uint64x1_t vadd_u64(uint64x1_t a, uint64x1_t b); // VADD.I64 d0,d0,d0
//128位==
int8x16_t vaddq_s8(int8x16_t a, int8x16_t b); // VADD.I8 q0,q0,q0
int16x8_t vaddq_s16(int16x8_t a, int16x8_t b); // VADD.I16 q0,q0,q0
int32x4_t vaddq_s32(int32x4_t a, int32x4_t b); // VADD.I32 q0,q0,q0
int64x2_t vaddq_s64(int64x2_t a, int64x2_t b); // VADD.I64 q0,q0,q0
float32x4_t vaddq_f32(float32x4_t a, float32x4_t b); // VADD.F32 q0,q0,q0
uint8x16_t vaddq_u8(uint8x16_t a, uint8x16_t b); // VADD.I8 q0,q0,q0
uint16x8_t vaddq_u16(uint16x8_t a, uint16x8_t b); // VADD.I16 q0,q0,q0
uint32x4_t vaddq_u32(uint32x4_t a, uint32x4_t b); // VADD.I32 q0,q0,q0
uint64x2_t vaddq_u64(uint64x2_t a, uint64x2_t b); // VADD.I64 q0,q0,q0
```

> **向量长型加法：vaddl -> Vr[i]:=Va[i]+Vb[i]**

Va、Vb 具有相等的向量线大小，结果为向量线宽度变成两倍的 128 位向量。
```c
int16x8_t vaddl_s8(int8x8_t a, int8x8_t b); // VADDL.S8 q0,d0,d0
int32x4_t vaddl_s16(int16x4_t a, int16x4_t b); // VADDL.S16 q0,d0,d0
int64x2_t vaddl_s32(int32x2_t a, int32x2_t b); // VADDL.S32 q0,d0,d0
uint16x8_t vaddl_u8(uint8x8_t a, uint8x8_t b); // VADDL.U8 q0,d0,d0
uint32x4_t vaddl_u16(uint16x4_t a, uint16x4_t b); // VADDL.U16 q0,d0,d0
uint64x2_t vaddl_u32(uint32x2_t a, uint32x2_t b); // VADDL.U32 q0,d0,d0

```
> **向量宽型加法：vaddw -> Vr[i]:=Va[i]+Vb[i] 64位与128位运算得到128位**
```c
int16x8_t vaddw_s8(int16x8_t a, int8x8_t b); // VADDW.S8 q0,q0,d0
int32x4_t vaddw_s16(int32x4_t a, int16x4_t b); // VADDW.S16 q0,q0,d0
int64x2_t vaddw_s32(int64x2_t a, int32x2_t b); // VADDW.S32 q0,q0,d0
uint16x8_t vaddw_u8(uint16x8_t a, uint8x8_t b); // VADDW.U8 q0,q0,d0
uint32x4_t vaddw_u16(uint32x4_t a, uint16x4_t b); // VADDW.U16 q0,q0,d0
uint64x2_t vaddw_u32(uint64x2_t a, uint32x2_t b); // VADDW.U32 q0,q0,d0
```
> **向量半加：vhadd -> Vr[i]:=(Va[i]+Vb[i])>>1 求和后除以2**
```c
//64位
int8x8_t vhadd_s8(int8x8_t a, int8x8_t b); // VHADD.S8 d0,d0,d0
int16x4_t vhadd_s16(int16x4_t a, int16x4_t b); // VHADD.S16 d0,d0,d0
int32x2_t vhadd_s32(int32x2_t a, int32x2_t b); // VHADD.S32 d0,d0,d0
uint8x8_t vhadd_u8(uint8x8_t a, uint8x8_t b); // VHADD.U8 d0,d0,d0
uint16x4_t vhadd_u16(uint16x4_t a, uint16x4_t b); // VHADD.U16 d0,d0,d0
uint32x2_t vhadd_u32(uint32x2_t a, uint32x2_t b); // VHADD.U32 d0,d0,d0
// 128位
int8x16_t vhaddq_s8(int8x16_t a, int8x16_t b); // VHADD.S8 q0,q0,q0
int16x8_t vhaddq_s16(int16x8_t a, int16x8_t b); // VHADD.S16 q0,q0,q0
int32x4_t vhaddq_s32(int32x4_t a, int32x4_t b); // VHADD.S32 q0,q0,q0
uint8x16_t vhaddq_u8(uint8x16_t a, uint8x16_t b); // VHADD.U8 q0,q0,q0
uint16x8_t vhaddq_u16(uint16x8_t a, uint16x8_t b); // VHADD.U16 q0,q0,q0
uint32x4_t vhaddq_u32(uint32x4_t a, uint32x4_t b); // VHADD.U32 q0,q0,q0
```

> **向量舍入半加：vrhadd -> Vr[i]:=(Va[i]+Vb[i]+1)>>1 求和再加1后除以2**

```c
//64位
int8x8_t vrhadd_s8(int8x8_t a, int8x8_t b); // VRHADD.S8 d0,d0,d0
int16x4_t vrhadd_s16(int16x4_t a, int16x4_t b); // VRHADD.S16 d0,d0,d0
int32x2_t vrhadd_s32(int32x2_t a, int32x2_t b); // VRHADD.S32 d0,d0,d0
uint8x8_t vrhadd_u8(uint8x8_t a, uint8x8_t b); // VRHADD.U8 d0,d0,d0
uint16x4_t vrhadd_u16(uint16x4_t a, uint16x4_t b); // VRHADD.U16 d0,d0,d0
uint32x2_t vrhadd_u32(uint32x2_t a, uint32x2_t b); // VRHADD.U32 d0,d0,d0
//128位
int8x16_t vrhaddq_s8(int8x16_t a, int8x16_t b); // VRHADD.S8 q0,q0,q0
int16x8_t vrhaddq_s16(int16x8_t a, int16x8_t b); // VRHADD.S16 q0,q0,q0
int32x4_t vrhaddq_s32(int32x4_t a, int32x4_t b); // VRHADD.S32 q0,q0,q0
uint8x16_t vrhaddq_u8(uint8x16_t a, uint8x16_t b); // VRHADD.U8 q0,q0,q0
uint16x8_t vrhaddq_u16(uint16x8_t a, uint16x8_t b); // VRHADD.U16 q0,q0,q0
uint32x4_t vrhaddq_u32(uint32x4_t a, uint32x4_t b); // VRHADD.U32 q0,q0,q0
```
> **向量饱和加法：vqadd -> Vr[i]:=sat<size>(Va[i]+Vb[i])**

```c
//64位	
int8x8_t vqadd_s8(int8x8_t a, int8x8_t b); // VQADD.S8 d0,d0,d0
int16x4_t vqadd_s16(int16x4_t a, int16x4_t b); // VQADD.S16 d0,d0,d0
int32x2_t vqadd_s32(int32x2_t a, int32x2_t b); // VQADD.S32 d0,d0,d0
int64x1_t vqadd_s64(int64x1_t a, int64x1_t b); // VQADD.S64 d0,d0,d0
uint8x8_t vqadd_u8(uint8x8_t a, uint8x8_t b); // VQADD.U8 d0,d0,d0
uint16x4_t vqadd_u16(uint16x4_t a, uint16x4_t b); // VQADD.U16 d0,d0,d0
uint32x2_t vqadd_u32(uint32x2_t a, uint32x2_t b); // VQADD.U32 d0,d0,d0
uint64x1_t vqadd_u64(uint64x1_t a, uint64x1_t b); // VQADD.U64 d0,d0,d0
//128位  前面的q表示饱和运算，后面的q表示q寄存器，128位寄存器操作数
int8x16_t vqaddq_s8(int8x16_t a, int8x16_t b); // VQADD.S8 q0,q0,q0
int16x8_t vqaddq_s16(int16x8_t a, int16x8_t b); // VQADD.S16 q0,q0,q0
int32x4_t vqaddq_s32(int32x4_t a, int32x4_t b); // VQADD.S32 q0,q0,q0
int64x2_t vqaddq_s64(int64x2_t a, int64x2_t b); // VQADD.S64 q0,q0,q0
uint8x16_t vqaddq_u8(uint8x16_t a, uint8x16_t b); // VQADD.U8 q0,q0,q0
uint16x8_t vqaddq_u16(uint16x8_t a, uint16x8_t b); // VQADD.U16 q0,q0,q0
uint32x4_t vqaddq_u32(uint32x4_t a, uint32x4_t b); // VQADD.U32 q0,q0,q0
uint64x2_t vqaddq_u64(uint64x2_t a, uint64x2_t b); // VQADD.U64 q0,q0,q0
```
> **高位半部分向量加法：- > Vr[i]:=Va[i]+Vb[i]**
```c
int8x8_t vaddhn_s16(int16x8_t a, int16x8_t b); // VADDHN.I16 d0,q0,q0
int16x4_t vaddhn_s32(int32x4_t a, int32x4_t b); // VADDHN.I32 d0,q0,q0
int32x2_t vaddhn_s64(int64x2_t a, int64x2_t b); // VADDHN.I64 d0,q0,q0
uint8x8_t vaddhn_u16(uint16x8_t a, uint16x8_t b); // VADDHN.I16 d0,q0,q0
uint16x4_t vaddhn_u32(uint32x4_t a, uint32x4_t b); // VADDHN.I32 d0,q0,q0
uint32x2_t vaddhn_u64(uint64x2_t a, uint64x2_t b); // VADDHN.I64 d0,q0,q0
```
> **高位半部分向量舍入加法**
```c
int8x8_t vraddhn_s16(int16x8_t a, int16x8_t b); // VRADDHN.I16 d0,q0,q0
int16x4_t vraddhn_s32(int32x4_t a, int32x4_t b); // VRADDHN.I32 d0,q0,q0
int32x2_t vraddhn_s64(int64x2_t a, int64x2_t b); // VRADDHN.I64 d0,q0,q0
uint8x8_t vraddhn_u16(uint16x8_t a, uint16x8_t b); // VRADDHN.I16 d0,q0,q0
uint16x4_t vraddhn_u32(uint32x4_t a, uint32x4_t b); // VRADDHN.I32 d0,q0,q0
uint32x2_t vraddhn_u64(uint64x2_t a, uint64x2_t b); // VRADDHN.I64 d0,q0,q0
```

#### 向量减法

>**正常向量减法 vsub -> Vr[i]:=Va[i]-Vb[i]**
```c
//64bits
int8x8_t vsub_s8(int8x8_t a, int8x8_t b); // VSUB.I8 d0,d0,d0
int16x4_t vsub_s16(int16x4_t a, int16x4_t b); // VSUB.I16 d0,d0,d0
int32x2_t vsub_s32(int32x2_t a, int32x2_t b); // VSUB.I32 d0,d0,d0
int64x1_t vsub_s64(int64x1_t a, int64x1_t b); // VSUB.I64 d0,d0,d0
float32x2_t vsub_f32(float32x2_t a, float32x2_t b); // VSUB.F32 d0,d0,d0
uint8x8_t vsub_u8(uint8x8_t a, uint8x8_t b);        // VSUB.I8 d0,d0,d0
uint16x4_t vsub_u16(uint16x4_t a, uint16x4_t b);    // VSUB.I16 d0,d0,d0
uint32x2_t vsub_u32(uint32x2_t a, uint32x2_t b);    // VSUB.I32 d0,d0,d0
uint64x1_t vsub_u64(uint64x1_t a, uint64x1_t b);    // VSUB.I64 d0,d0,d0
//128bits
int8x16_t vsubq_s8(int8x16_t a, int8x16_t b); // VSUB.I8 q0,q0,q0
int16x8_t vsubq_s16(int16x8_t a, int16x8_t b); // VSUB.I16 q0,q0,q0
int32x4_t vsubq_s32(int32x4_t a, int32x4_t b); // VSUB.I32 q0,q0,q0
int64x2_t vsubq_s64(int64x2_t a, int64x2_t b); // VSUB.I64 q0,q0,q0
float32x4_t vsubq_f32(float32x4_t a, float32x4_t b); // VSUB.F32 q0,q0,q0
uint8x16_t vsubq_u8(uint8x16_t a, uint8x16_t b); // VSUB.I8 q0,q0,q0
uint16x8_t vsubq_u16(uint16x8_t a, uint16x8_t b); // VSUB.I16 q0,q0,q0
uint32x4_t vsubq_u32(uint32x4_t a, uint32x4_t b); // VSUB.I32 q0,q0,q0
uint64x2_t vsubq_u64(uint64x2_t a, uint64x2_t b); // VSUB.I64 q0,q0,q0
```


>**向量长型减法：vsubl -> Vr[i]:=Va[i]-Vb[i]**
```c
int16x8_t vsubl_s8(int8x8_t a, int8x8_t b); // VSUBL.S8 q0,d0,d0
int32x4_t vsubl_s16(int16x4_t a, int16x4_t b); // VSUBL.S16 q0,d0,d0
int64x2_t vsubl_s32(int32x2_t a, int32x2_t b); // VSUBL.S32 q0,d0,d0
uint16x8_t vsubl_u8(uint8x8_t a, uint8x8_t b); // VSUBL.U8 q0,d0,d0
uint32x4_t vsubl_u16(uint16x4_t a, uint16x4_t b); // VSUBL.U16 q0,d0,d0
uint64x2_t vsubl_u32(uint32x2_t a, uint32x2_t b); // VSUBL.U32 q0,d0,d0
```

>**向量宽型减法：vsubw -> Vr[i]:=Va[i]+Vb[i]**
```c
int16x8_t vsubw_s8(int16x8_t a, int8x8_t b); // VSUBW.S8 q0,q0,d0
int32x4_t vsubw_s16(int32x4_t a, int16x4_t b); // VSUBW.S16 q0,q0,d0
int64x2_t vsubw_s32(int64x2_t a, int32x2_t b); // VSUBW.S32 q0,q0,d0
uint16x8_t vsubw_u8(uint16x8_t a, uint8x8_t b); // VSUBW.U8 q0,q0,d0
uint32x4_t vsubw_u16(uint32x4_t a, uint16x4_t b); // VSUBW.U16 q0,q0,d0
uint64x2_t vsubw_u32(uint64x2_t a, uint32x2_t b); // VSUBW.U32 q0,q0,d0
```

>**向量饱和减法 vqsub-> Vr[i]:=sat<size>(Va[i]-Vb[i])**
	
```c
//64bits
int8x8_t vqsub_s8(int8x8_t a, int8x8_t b); // VQSUB.S8 d0,d0,d0
int16x4_t vqsub_s16(int16x4_t a, int16x4_t b); // VQSUB.S16 d0,d0,d0
int32x2_t vqsub_s32(int32x2_t a, int32x2_t b); // VQSUB.S32 d0,d0,d0
int64x1_t vqsub_s64(int64x1_t a, int64x1_t b); // VQSUB.S64 d0,d0,d0
uint8x8_t vqsub_u8(uint8x8_t a, uint8x8_t b); // VQSUB.U8 d0,d0,d0
uint16x4_t vqsub_u16(uint16x4_t a, uint16x4_t b); // VQSUB.U16 d0,d0,d0
uint32x2_t vqsub_u32(uint32x2_t a, uint32x2_t b); // VQSUB.U32 d0,d0,d0
uint64x1_t vqsub_u64(uint64x1_t a, uint64x1_t b); // VQSUB.U64 d0,d0,d0
//128bits
int8x16_t vqsubq_s8(int8x16_t a, int8x16_t b); // VQSUB.S8 q0,q0,q0
int16x8_t vqsubq_s16(int16x8_t a, int16x8_t b); // VQSUB.S16 q0,q0,q0
int32x4_t vqsubq_s32(int32x4_t a, int32x4_t b); // VQSUB.S32 q0,q0,q0
int64x2_t vqsubq_s64(int64x2_t a, int64x2_t b); // VQSUB.S64 q0,q0,q0
uint8x16_t vqsubq_u8(uint8x16_t a, uint8x16_t b); // VQSUB.U8 q0,q0,q0
uint16x8_t vqsubq_u16(uint16x8_t a, uint16x8_t b); // VQSUB.U16 q0,q0,q0
uint32x4_t vqsubq_u32(uint32x4_t a, uint32x4_t b); // VQSUB.U32 q0,q0,q0
uint64x2_t vqsubq_u64(uint64x2_t a, uint64x2_t b); // VQSUB.U64 q0,q0,q0
```

>**向量半减Vr[i]:=(Va[i]-Vb[i])>>1**
```c
int8x8_t vhsub_s8(int8x8_t a, int8x8_t b); // VHSUB.S8 d0,d0,d0
int16x4_t vhsub_s16(int16x4_t a, int16x4_t b); // VHSUB.S16 d0,d0,d0
int32x2_t vhsub_s32(int32x2_t a, int32x2_t b); // VHSUB.S32 d0,d0,d0
uint8x8_t vhsub_u8(uint8x8_t a, uint8x8_t b); // VHSUB.U8 d0,d0,d0
uint16x4_t vhsub_u16(uint16x4_t a, uint16x4_t b); // VHSUB.U16 d0,d0,d0
uint32x2_t vhsub_u32(uint32x2_t a, uint32x2_t b); // VHSUB.U32 d0,d0,d0
int8x16_t vhsubq_s8(int8x16_t a, int8x16_t b); // VHSUB.S8 q0,q0,q0
int16x8_t vhsubq_s16(int16x8_t a, int16x8_t b); // VHSUB.S16 q0,q0,q0
int32x4_t vhsubq_s32(int32x4_t a, int32x4_t b); // VHSUB.S32 q0,q0,q0
uint8x16_t vhsubq_u8(uint8x16_t a, uint8x16_t b); // VHSUB.U8 q0,q0,q0
uint16x8_t vhsubq_u16(uint16x8_t a, uint16x8_t b); // VHSUB.U16 q0,q0,q0
uint32x4_t vhsubq_u32(uint32x4_t a, uint32x4_t b); // VHSUB.U32 q0,q0,q0
```

#### 乘法

>**向量乘法：vmul -> Vr[i] := Va[i] * Vb[i]**
```c
//64bits===
int8x8_t vmul_s8(int8x8_t a, int8x8_t b); // VMUL.I8 d0,d0,d0
int16x4_t vmul_s16(int16x4_t a, int16x4_t b); // VMUL.I16 d0,d0,d0
int32x2_t vmul_s32(int32x2_t a, int32x2_t b); // VMUL.I32 d0,d0,d0
float32x2_t vmul_f32(float32x2_t a, float32x2_t b); // VMUL.F32 d0,d0,d0
uint8x8_t vmul_u8(uint8x8_t a, uint8x8_t b); // VMUL.I8 d0,d0,d0
uint16x4_t vmul_u16(uint16x4_t a, uint16x4_t b); // VMUL.I16 d0,d0,d0
uint32x2_t vmul_u32(uint32x2_t a, uint32x2_t b); // VMUL.I32 d0,d0,d0
poly8x8_t vmul_p8(poly8x8_t a, poly8x8_t b); // VMUL.P8 d0,d0,d0
//128bits==
int8x16_t vmulq_s8(int8x16_t a, int8x16_t b); // VMUL.I8 q0,q0,q0
int16x8_t vmulq_s16(int16x8_t a, int16x8_t b); // VMUL.I16 q0,q0,q0
int32x4_t vmulq_s32(int32x4_t a, int32x4_t b); // VMUL.I32 q0,q0,q0
float32x4_t vmulq_f32(float32x4_t a, float32x4_t b); // VMUL.F32 q0,q0,q0
uint8x16_t vmulq_u8(uint8x16_t a, uint8x16_t b); // VMUL.I8 q0,q0,q0
uint16x8_t vmulq_u16(uint16x8_t a, uint16x8_t b); // VMUL.I16 q0,q0,q0
uint32x4_t vmulq_u32(uint32x4_t a, uint32x4_t b); // VMUL.I32 q0,q0,q0
poly8x16_t vmulq_p8(poly8x16_t a, poly8x16_t b); // VMUL.P8 q0,q0,q0

```
>**向量长型乘法：vmull -> Vr[i] := Va[i] * Vb[i]**
```c
int16x8_t vmull_s8(int8x8_t a, int8x8_t b); // VMULL.S8 q0,d0,d0
int32x4_t vmull_s16(int16x4_t a, int16x4_t b); // VMULL.S16 q0,d0,d0
int64x2_t vmull_s32(int32x2_t a, int32x2_t b); // VMULL.S32 q0,d0,d0
uint16x8_t vmull_u8(uint8x8_t a, uint8x8_t b); // VMULL.U8 q0,d0,d0
```

>**向量乘加：vmla -> Vr[i] := Va[i] + Vb[i] * Vc[i]**
```c
//64bits===
int8x8_t vmla_s8(int8x8_t a, int8x8_t b, int8x8_t c); // VMLA.I8 d0,d0,d0
int16x4_t vmla_s16(int16x4_t a, int16x4_t b, int16x4_t c); // VMLA.I16 d0,d0,d0
int32x2_t vmla_s32(int32x2_t a, int32x2_t b, int32x2_t c); // VMLA.I32 d0,d0,d0
float32x2_t vmla_f32(float32x2_t a, float32x2_t b, float32x2_t c); // VMLA.F32 d0,d0,d0
uint8x8_t vmla_u8(uint8x8_t a, uint8x8_t b, uint8x8_t c); // VMLA.I8 d0,d0,d0
uint16x4_t vmla_u16(uint16x4_t a, uint16x4_t b, uint16x4_t c); // VMLA.I16 d0,d0,d0
uint32x2_t vmla_u32(uint32x2_t a, uint32x2_t b, uint32x2_t c); // VMLA.I32 d0,d0,d0
//128bits==
int8x16_t vmlaq_s8(int8x16_t a, int8x16_t b, int8x16_t c); // VMLA.I8 q0,q0,q0
int16x8_t vmlaq_s16(int16x8_t a, int16x8_t b, int16x8_t c); // VMLA.I16 q0,q0,q0
int32x4_t vmlaq_s32(int32x4_t a, int32x4_t b, int32x4_t c); // VMLA.I32 q0,q0,q0
float32x4_t vmlaq_f32(float32x4_t a, float32x4_t b, float32x4_t c); // VMLA.F32 q0,q0,q0
uint8x16_t vmlaq_u8(uint8x16_t a, uint8x16_t b, uint8x16_t c); // VMLA.I8 q0,q0,q0
uint16x8_t vmlaq_u16(uint16x8_t a, uint16x8_t b, uint16x8_t c); // VMLA.I16 q0,q0,q0
uint32x4_t vmlaq_u32(uint32x4_t a, uint32x4_t b, uint32x4_t c); // VMLA.I32 q0,q0,q0
```


>**向量长型乘加：vmlal -> Vr[i] := Va[i] + Vb[i] * Vc[i]**
```c
int16x8_t vmlal_s8(int16x8_t a, int8x8_t b, int8x8_t c); // VMLAL.S8 q0,d0,d0
int32x4_t vmlal_s16(int32x4_t a, int16x4_t b, int16x4_t c); // VMLAL.S16 q0,d0,d0
int64x2_t vmlal_s32(int64x2_t a, int32x2_t b, int32x2_t c); // VMLAL.S32 q0,d0,d0
uint16x8_t vmlal_u8(uint16x8_t a, uint8x8_t b, uint8x8_t c); // VMLAL.U8 q0,d0,d0
uint32x4_t vmlal_u16(uint32x4_t a, uint16x4_t b, uint16x4_t c); // VMLAL.U16 q0,d0,d0
uint64x2_t vmlal_u32(uint64x2_t a, uint32x2_t b, uint32x2_t c); // VMLAL.U32 q0,d0,d0
```

>**向量乘减：vmls -> Vr[i] := Va[i] - Vb[i] * Vc[i]**
```c
//64bits==
int8x8_t vmls_s8(int8x8_t a, int8x8_t b, int8x8_t c); // VMLS.I8 d0,d0,d0
int16x4_t vmls_s16(int16x4_t a, int16x4_t b, int16x4_t c); // VMLS.I16 d0,d0,d0
int32x2_t vmls_s32(int32x2_t a, int32x2_t b, int32x2_t c); // VMLS.I32 d0,d0,d0
float32x2_t vmls_f32(float32x2_t a, float32x2_t b, float32x2_t c); // VMLS.F32 d0,d0,d0
uint8x8_t vmls_u8(uint8x8_t a, uint8x8_t b, uint8x8_t c); // VMLS.I8 d0,d0,d0
uint16x4_t vmls_u16(uint16x4_t a, uint16x4_t b, uint16x4_t c); // VMLS.I16 d0,d0,d0
uint32x2_t vmls_u32(uint32x2_t a, uint32x2_t b, uint32x2_t c); // VMLS.I32 d0,d0,d0
//128bits==
int8x16_t vmlsq_s8(int8x16_t a, int8x16_t b, int8x16_t c); // VMLS.I8 q0,q0,q0
int16x8_t vmlsq_s16(int16x8_t a, int16x8_t b, int16x8_t c); // VMLS.I16 q0,q0,q0
int32x4_t vmlsq_s32(int32x4_t a, int32x4_t b, int32x4_t c); // VMLS.I32 q0,q0,q0
float32x4_t vmlsq_f32(float32x4_t a, float32x4_t b, float32x4_t c); // VMLS.F32 q0,q0,q0
uint8x16_t vmlsq_u8(uint8x16_t a, uint8x16_t b, uint8x16_t c); // VMLS.I8 q0,q0,q0
uint16x8_t vmlsq_u16(uint16x8_t a, uint16x8_t b, uint16x8_t c); // VMLS.I16 q0,q0,q0
uint32x4_t vmlsq_u32(uint32x4_t a, uint32x4_t b, uint32x4_t c); // VMLS.I32 q0,q0,q0
```

>**向量长型乘减 vmlsl -> Vr[i] := Va[i] - Vb[i] * Vc[i]**
```c
int16x8_t vmlsl_s8(int16x8_t a, int8x8_t b, int8x8_t c); // VMLSL.S8 q0,d0,d0
int32x4_t vmlsl_s16(int32x4_t a, int16x4_t b, int16x4_t c); // VMLSL.S16 q0,d0,d0
int64x2_t vmlsl_s32(int64x2_t a, int32x2_t b, int32x2_t c); // VMLSL.S32 q0,d0,d0
uint16x8_t vmlsl_u8(uint16x8_t a, uint8x8_t b, uint8x8_t c); // VMLSL.U8 q0,d0,d0
uint32x4_t vmlsl_u16(uint32x4_t a, uint16x4_t b, uint16x4_t c); // VMLSL.U16 q0,d0,d0
uint64x2_t vmlsl_u32(uint64x2_t a, uint32x2_t b, uint32x2_t c); // VMLSL.U32 q0,d0,d0
```

#### 比较compare
提供一系列比较内在函数。如果对于一条向量线比较结果为 true，则该向量线的结果为将所有位设置为一。如果对于一条向量线比较结果为 false，则将所有位设置为零。返回类型是无符号整数类型。这意味着可以将比较结果用作 vbsl内在函数的第一个参数。


>**向量比较 等于否 vceq_type vceqq_type  compare equal**
```c
// 64位
uint8x8_t vceq_s8(int8x8_t a, int8x8_t b); // VCEQ.I8 d0, d0, d0
uint16x4_t vceq_s16(int16x4_t a, int16x4_t b); // VCEQ.I16 d0, d0, d0
uint32x2_t vceq_s32(int32x2_t a, int32x2_t b); // VCEQ.I32 d0, d0, d0
uint32x2_t vceq_f32(float32x2_t a, float32x2_t b); // VCEQ.F32 d0, d0, d0
uint8x8_t vceq_u8(uint8x8_t a, uint8x8_t b); // VCEQ.I8 d0, d0, d0
uint16x4_t vceq_u16(uint16x4_t a, uint16x4_t b); // VCEQ.I16 d0, d0, d0
uint32x2_t vceq_u32(uint32x2_t a, uint32x2_t b); // VCEQ.I32 d0, d0, d0
uint8x8_t vceq_p8(poly8x8_t a, poly8x8_t b); // VCEQ.I8 d0, d0, d0
// 128位
uint8x16_t vceqq_s8(int8x16_t a, int8x16_t b); // VCEQ.I8 q0, q0, q0
uint16x8_t vceqq_s16(int16x8_t a, int16x8_t b); // VCEQ.I16 q0, q0, q0
uint32x4_t vceqq_s32(int32x4_t a, int32x4_t b); // VCEQ.I32 q0, q0, q0
uint32x4_t vceqq_f32(float32x4_t a, float32x4_t b); // VCEQ.F32 q0, q0, q0
uint8x16_t vceqq_u8(uint8x16_t a, uint8x16_t b); // VCEQ.I8 q0, q0, q0
uint16x8_t vceqq_u16(uint16x8_t a, uint16x8_t b); // VCEQ.I16 q0, q0, q0
uint32x4_t vceqq_u32(uint32x4_t a, uint32x4_t b); // VCEQ.I32 q0, q0, q0
uint8x16_t vceqq_p8(poly8x16_t a, poly8x16_t b); // VCEQ.I8 q0, q0, q0
```

>**向量比较大于或等于 vcge vcgeq : compare greate or equal**
```c
// 64位
uint8x8_t vcge_s8(int8x8_t a, int8x8_t b); // VCGE.S8 d0, d0, d0
uint16x4_t vcge_s16(int16x4_t a, int16x4_t b); // VCGE.S16 d0, d0, d0
uint32x2_t vcge_s32(int32x2_t a, int32x2_t b); // VCGE.S32 d0, d0, d0
uint32x2_t vcge_f32(float32x2_t a, float32x2_t b); // VCGE.F32 d0, d0, d0
uint8x8_t vcge_u8(uint8x8_t a, uint8x8_t b); // VCGE.U8 d0, d0, d0
uint16x4_t vcge_u16(uint16x4_t a, uint16x4_t b); // VCGE.U16 d0, d0, d0
uint32x2_t vcge_u32(uint32x2_t a, uint32x2_t b); // VCGE.U32 d0, d0, d0

// 128位
uint8x16_t vcgeq_s8(int8x16_t a, int8x16_t b); // VCGE.S8 q0, q0, q0
uint16x8_t vcgeq_s16(int16x8_t a, int16x8_t b); // VCGE.S16 q0, q0, q0
uint32x4_t vcgeq_s32(int32x4_t a, int32x4_t b); // VCGE.S32 q0, q0, q0
uint32x4_t vcgeq_f32(float32x4_t a, float32x4_t b); // VCGE.F32 q0, q0, q0
uint8x16_t vcgeq_u8(uint8x16_t a, uint8x16_t b); // VCGE.U8 q0, q0, q0
uint16x8_t vcgeq_u16(uint16x8_t a, uint16x8_t b); // VCGE.U16 q0, q0, q0
uint32x4_t vcgeq_u32(uint32x4_t a, uint32x4_t b); // VCGE.U32 q0, q0, q0

```

>**向量比较小于或等于 vcle vcleq : compare little or equal**
```c
//64bits
uint8x8_t vcle_s8(int8x8_t a, int8x8_t b); // VCGE.S8 d0, d0, d0
uint16x4_t vcle_s16(int16x4_t a, int16x4_t b); // VCGE.S16 d0, d0, d0
uint32x2_t vcle_s32(int32x2_t a, int32x2_t b); // VCGE.S32 d0, d0, d0
uint32x2_t vcle_f32(float32x2_t a, float32x2_t b); // VCGE.F32 d0, d0, d0
uint8x8_t vcle_u8(uint8x8_t a, uint8x8_t b); // VCGE.U8 d0, d0, d0
uint16x4_t vcle_u16(uint16x4_t a, uint16x4_t b); // VCGE.U16 d0, d0, d0
uint32x2_t vcle_u32(uint32x2_t a, uint32x2_t b); // VCGE.U32 d0, d0, d0
// 128bits
uint8x16_t vcleq_s8(int8x16_t a, int8x16_t b); // VCGE.S8 q0, q0, q0
uint16x8_t vcleq_s16(int16x8_t a, int16x8_t b); // VCGE.S16 q0, q0, q0
uint32x4_t vcleq_s32(int32x4_t a, int32x4_t b); // VCGE.S32 q0, q0, q0
uint32x4_t vcleq_f32(float32x4_t a, float32x4_t b); // VCGE.F32 q0, q0, q0
uint8x16_t vcleq_u8(uint8x16_t a, uint8x16_t b); // VCGE.U8 q0, q0, q0
uint16x8_t vcleq_u16(uint16x8_t a, uint16x8_t b); // VCGE.U16 q0, q0, q0
uint32x4_t vcleq_u32(uint32x4_t a, uint32x4_t b); // VCGE.U32 q0, q0, q0
```

>**向量比较大于 vcgt vcgtq compare great **
```c
// 64bits
uint8x8_t vcgt_s8(int8x8_t a, int8x8_t b); // VCGT.S8 d0, d0, d0
uint16x4_t vcgt_s16(int16x4_t a, int16x4_t b); // VCGT.S16 d0, d0, d0
uint32x2_t vcgt_s32(int32x2_t a, int32x2_t b); // VCGT.S32 d0, d0, d0
uint32x2_t vcgt_f32(float32x2_t a, float32x2_t b); // VCGT.F32 d0, d0, d0
uint8x8_t vcgt_u8(uint8x8_t a, uint8x8_t b); // VCGT.U8 d0, d0, d0
uint16x4_t vcgt_u16(uint16x4_t a, uint16x4_t b); // VCGT.U16 d0, d0, d0
uint32x2_t vcgt_u32(uint32x2_t a, uint32x2_t b); // VCGT.U32 d0, d0, d0
// 128bits
uint8x16_t vcgtq_s8(int8x16_t a, int8x16_t b); // VCGT.S8 q0, q0, q0
uint16x8_t vcgtq_s16(int16x8_t a, int16x8_t b); // VCGT.S16 q0, q0, q0
uint32x4_t vcgtq_s32(int32x4_t a, int32x4_t b); // VCGT.S32 q0, q0, q0
uint32x4_t vcgtq_f32(float32x4_t a, float32x4_t b); // VCGT.F32 q0, q0, q0
uint8x16_t vcgtq_u8(uint8x16_t a, uint8x16_t b); // VCGT.U8 q0, q0, q0
uint16x8_t vcgtq_u16(uint16x8_t a, uint16x8_t b); // VCGT.U16 q0, q0, q0
uint32x4_t vcgtq_u32(uint32x4_t a, uint32x4_t b); // VCGT.U32 q0, q0, q0
```

>**向量比较小于 vclt vcltq : compare little **
```c
//64bits==
uint8x8_t vclt_s8(int8x8_t a, int8x8_t b); // VCGT.S8 d0, d0, d0
uint16x4_t vclt_s16(int16x4_t a, int16x4_t b); // VCGT.S16 d0, d0, d0
uint32x2_t vclt_s32(int32x2_t a, int32x2_t b); // VCGT.S32 d0, d0, d0
uint32x2_t vclt_f32(float32x2_t a, float32x2_t b); // VCGT.F32 d0, d0, d0
uint8x8_t vclt_u8(uint8x8_t a, uint8x8_t b); // VCGT.U8 d0, d0, d0
uint16x4_t vclt_u16(uint16x4_t a, uint16x4_t b); // VCGT.U16 d0, d0, d0
uint32x2_t vclt_u32(uint32x2_t a, uint32x2_t b); // VCGT.U32 d0, d0, d0
// 128bits===
uint8x16_t vcltq_s8(int8x16_t a, int8x16_t b); // VCGT.S8 q0, q0, q0
uint16x8_t vcltq_s16(int16x8_t a, int16x8_t b); // VCGT.S16 q0, q0, q0
uint32x4_t vcltq_s32(int32x4_t a, int32x4_t b); // VCGT.S32 q0, q0, q0
uint32x4_t vcltq_f32(float32x4_t a, float32x4_t b); // VCGT.F32 q0, q0, q0
uint8x16_t vcltq_u8(uint8x16_t a, uint8x16_t b); // VCGT.U8 q0, q0, q0
uint16x8_t vcltq_u16(uint16x8_t a, uint16x8_t b); // VCGT.U16 q0, q0, q0
uint32x4_t vcltq_u32(uint32x4_t a, uint32x4_t b); // VCGT.U32 q0, q0, q0
```

>**向量绝对值比较大于或等于 vcage vcageq: compare abs great equal**
```c

uint32x2_t vcage_f32(float32x2_t a, float32x2_t b); // VACGE.F32 d0, d0, d0
uint32x4_t vcageq_f32(float32x4_t a, float32x4_t b); // VACGE.F32 q0, q0, q0
```

>**向量绝对值比较小于或等于 vcale vcaleq: compare abs little equal **
```c
uint32x2_t vcale_f32(float32x2_t a, float32x2_t b); // VACGE.F32 d0, d0, d0
uint32x4_t vcaleq_f32(float32x4_t a, float32x4_t b); // VACGE.F32 q0, q0, q0
```

>**向量绝对值比较大于 vcagt vcagtq: compare abs great**
```c
uint32x2_t vcagt_f32(float32x2_t a, float32x2_t b); // VACGT.F32 d0, d0, d0
uint32x4_t vcagtq_f32(float32x4_t a, float32x4_t b); // VACGT.F32 q0, q0, q0
```

>**向量绝对值比较小于 vcalt vcaltq:compare abs little**
```c
uint32x2_t vcalt_f32(float32x2_t a, float32x2_t b); // VACGT.F32 d0, d0, d0
uint32x4_t vcaltq_f32(float32x4_t a, float32x4_t b); // VACGT.F32 q0, q0, q0
```

>**向量测试位 test**
```c
uint8x8_t vtst_s8(int8x8_t a, int8x8_t b); // VTST.8 d0, d0, d0
uint16x4_t vtst_s16(int16x4_t a, int16x4_t b); // VTST.16 d0, d0, d0
uint32x2_t vtst_s32(int32x2_t a, int32x2_t b); // VTST.32 d0, d0, d0
uint8x8_t vtst_u8(uint8x8_t a, uint8x8_t b); // VTST.8 d0, d0, d0
uint16x4_t vtst_u16(uint16x4_t a, uint16x4_t b); // VTST.16 d0, d0, d0
uint32x2_t vtst_u32(uint32x2_t a, uint32x2_t b); // VTST.32 d0, d0, d0
uint8x8_t vtst_p8(poly8x8_t a, poly8x8_t b); // VTST.8 d0, d0, d0

uint8x16_t vtstq_s8(int8x16_t a, int8x16_t b); // VTST.8 q0, q0, q0
uint16x8_t vtstq_s16(int16x8_t a, int16x8_t b); // VTST.16 q0, q0, q0
uint32x4_t vtstq_s32(int32x4_t a, int32x4_t b); // VTST.32 q0, q0, q0
uint8x16_t vtstq_u8(uint8x16_t a, uint8x16_t b); // VTST.8 q0, q0, q0
uint16x8_t vtstq_u16(uint16x8_t a, uint16x8_t b); // VTST.16 q0, q0, q0
uint32x4_t vtstq_u32(uint32x4_t a, uint32x4_t b); // VTST.32 q0, q0, q0
uint8x16_t vtstq_p8(poly8x16_t a, poly8x16_t b); // VTST.8 q0, q0, q0

```
#### 差值绝对值
>**参数间的差值绝对值：Vr[i] = | Va[i] - Vb[i] |  vabd: abs difference**
```c
int8x8_t vabd_s8(int8x8_t a, int8x8_t b); // VABD.S8 d0,d0,d0
int16x4_t vabd_s16(int16x4_t a, int16x4_t b); // VABD.S16 d0,d0,d0
int32x2_t vabd_s32(int32x2_t a, int32x2_t b); // VABD.S32 d0,d0,d0
uint8x8_t vabd_u8(uint8x8_t a, uint8x8_t b); // VABD.U8 d0,d0,d0
uint16x4_t vabd_u16(uint16x4_t a, uint16x4_t b); // VABD.U16 d0,d0,d0
uint32x2_t vabd_u32(uint32x2_t a, uint32x2_t b); // VABD.U32 d0,d0,d0
float32x2_t vabd_f32(float32x2_t a, float32x2_t b); // VABD.F32 d0,d0,d0
// 128bits
int8x16_t vabdq_s8(int8x16_t a, int8x16_t b); // VABD.S8 q0,q0,q0
int16x8_t vabdq_s16(int16x8_t a, int16x8_t b); // VABD.S16 q0,q0,q0
int32x4_t vabdq_s32(int32x4_t a, int32x4_t b); // VABD.S32 q0,q0,q0
uint8x16_t vabdq_u8(uint8x16_t a, uint8x16_t b); // VABD.U8 q0,q0,q0
uint16x8_t vabdq_u16(uint16x8_t a, uint16x8_t b); // VABD.U16 q0,q0,q0
uint32x4_t vabdq_u32(uint32x4_t a, uint32x4_t b); // VABD.U32 q0,q0,q0
float32x4_t vabdq_f32(float32x4_t a, float32x4_t b); // VABD.F32 q0,q0,q0
```

>**差值绝对值 - 长型 **
```c
int16x8_t vabdl_s8(int8x8_t a, int8x8_t b); // VABDL.S8 q0,d0,d0
int32x4_t vabdl_s16(int16x4_t a, int16x4_t b); // VABDL.S16 q0,d0,d0
int64x2_t vabdl_s32(int32x2_t a, int32x2_t b); // VABDL.S32 q0,d0,d0
uint16x8_t vabdl_u8(uint8x8_t a, uint8x8_t b); // VABDL.U8 q0,d0,d0
uint32x4_t vabdl_u16(uint16x4_t a, uint16x4_t b); // VABDL.U16 q0,d0,d0
uint64x2_t vabdl_u32(uint32x2_t a, uint32x2_t b); // VABDL.U32 q0,d0,d0
```
#### 加载存储指令
>**加载并存储单个向量 加载并存储某类型的单个向量。vld1q_type**
```c

```
### 实例0：数组元素求和
```c
// c版本=======================
#include <iostream>
using namespace std;

float sum_array(float *arr, int len)
{
    if(NULL == arr || len < 1)
    {
        cout<<"input error\n";
        return 0;
    }
    float sum(0.0);
    for(int i=0; i<len; ++i)
    {
        sum += *arr++;
    }
    return sum;
}


// arm intrinsics==============
#include <iostream>
#include <arm_neon.h> //需包含的头文件
using namespace std;

float sum_array(float *arr, int len)
{
    if(NULL == arr || len < 1)
    {
        cout<<"input error\n";
        return 0;
    }

    int dim4 = len >> 2; // 数组长度除4整数
    int left4 = len & 3; // 数组长度除4余数,不够4的剩下的
    
    float32x4_t sum_vec = vdupq_n_f32(0.0);//定义用于暂存累加结果的寄存器且初始化为0
    for (; dim4>0; dim4--, arr+=4) //每次同时访问4个数组元素
    {
        float32x4_t data_vec = vld1q_f32(arr); //依次取4个元素存入寄存器vec
        sum_vec = vaddq_f32(sum_vec, data_vec);//ri = ai + bi 计算两组寄存器对应元素之和并存放到相应结果
    }
    float sum = vgetq_lane_f32(sum_vec, 0)+vgetq_lane_f32(sum_vec, 1)+vgetq_lane_f32(sum_vec, 2)+vgetq_lane_f32(sum_vec, 3);//将累加结果寄存器中的所有元素相加得到最终累加值
    for (; left4>0; left4--, arr++)
        sum += (*arr) ;   //对于剩下的少于4的数字，依次计算累加即可
    return sum;
}
```

上述算法的时间复杂度时O(N/4) 
从上面的例子看出，使用NEON函数很简单，只需要将依次处理，变为批处理（如上面的每次处理4个）。

上面用到的函数有： 
float32x4_t vdupq_n_f32 (float32_t value) 
将value复制4分存到返回的寄存器中

float32x4_t vld1q_f32 (float32_t const * ptr) 
从数组中依次Load4个元素存到寄存器中

相应的 有void vst1q_f32 (float32_t * ptr, float32x4_t val) 
将寄存器中的值写入数组中

float32x4_t vaddq_f32 (float32x4_t a, float32x4_t b) 
返回两个寄存器对应元素之和 r = a+b

相应的 有float32x4_t vsubq_f32 (float32x4_t a, float32x4_t b) 
返回两个寄存器对应元素之差 r = a-b

float32_t vgetq_lane_f32 (float32x4_t v, const int lane) 
返回寄存器某一lane的值

其他常用的函数还有：

float32x4_t vmulq_f32 (float32x4_t a, float32x4_t b) 
返回两个寄存器对应元素之积 r = a*b

float32x4_t vmlaq_f32 (float32x4_t a, float32x4_t b, float32x4_t c) 
乘加 r = a +b*c

float32x4_t vmlsq_f32 (float32x4_t a, float32x4_t b, float32x4_t c) 
乘减 r = a - b*c

float32x4_t vextq_f32 (float32x4_t a, float32x4_t b, const int n) 
拼接两个寄存器并返回从第n位开始的大小为4的寄存器 0<=n<=3 
例如 

	a: 1 2 3 4 
	b: 5 6 7 8 
	vextq_f32(a,b,1) -> r: 2 3 4 5 
	vextq_f32(a,b,2) -> r: 3 4 5 6 
	vextq_f32(a,b,3) -> r: 4 5 6 7
	
```c
float32x4_t sum = vdupq_n_f32(0); // sum四个通道全部赋值为0，sum={0,0,0,0}
float _a[] = {1,2,3,4}, _b[] = {5,6,7,8} ;
float32x4_t a = vld1q_f32(_a), b = vld1q_f32(_b)  ;// 载入两个数组元素到 两个寄存器

//a的元素乘以b的第几个通道元素，然后后面的累加
float32x4_t sum1 = vfmaq_laneq_f32(sum, a, b, 0);  // sum1={5,10,15,20}
float32x4_t sum2 = vfmaq_laneq_f32(sum1, a, b, 1); 
// sum2={5,10,15,20}+{6,12,18,24} = {11,22,33,44}

float32x4_t sum3 = vfmaq_laneq_f32(sum2, a, b, 2);
// sum3={11,22,33,44}+{7,14,21,28} = {18,36,54,72}
```

[官方文档 其他常用函数](https://developer.arm.com/architectures/instruction-sets/simd-isas/neon/intrinsics)

### 示例1：向量加法**
```c
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
		                      // v 表示neon函数
				      // ld表示加载load
				      // q表示使用128位寄存器
				      // s32,有符号32位整数，单个数据32，共有4个数据并行超声
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



### 示例2：向量乘法 

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

> 处理剩余的元素
[参考](https://blog.csdn.net/hw5226349/article/details/45111237)

* 1. Larger Arrays 扩展成更大的数组

如果改变你要处理的数组大小，比如增加数组大小到向量大小的整数倍，这样就能在最后一次数据处理时也按照向量大小处理而不会把临近的数据损坏。如上面的例子里，把数组大小增加到24个元素，这样就能用NEON用3次迭代完成所有的数据处理而不会损坏周边数据。

填补数组到向量的整数个大小：
![](https://github.com/Ewenwan/MVision/blob/master/CNN/HighPerformanceComputing/img/Large-Arrays.jpg)

一些情况下，可能没法初始化填充的数据，无论填充什么都会影响计算的结果；

* 2. Overlapping重叠计算

如果进行数据处理的操作合适的话，可以考虑把剩余部分的元素通过重叠计算的方式处理，这就会把某些重叠部分的元素计算两次。如下面的例子里，第一次迭代计算元素0到7，第一次计算5到12，第三次计算13到20。从而第一次计算和第二次计算重叠的元素5到7就被计算了两次。

重叠向量，在橙色区域的数据计算两次:
![](https://github.com/Ewenwan/MVision/blob/master/CNN/HighPerformanceComputing/img/Overlapping.jpg))

重叠处理只适用于需要处理的数组长度不会随着每次迭代而改变的情况，但不适用于每次迭代结果改变的情况，如累加计算，这样重叠部分的数据会被计算两次；


* 3. 单个元素的计算过程Single Elements

NEON提供了能处理向量里的单一元素的加载和存储指令，用这些指令，你能加载包含一个元素的部分向量，处理它然后把结果保存到内存。如下面的例子，前两次的迭代处理跟前面类似，处理元素0到7以及8到15，剩下的5个元素可以在第三次迭代处理，加载处理并存储单一的元素。

处理单一的元素实例：

![](https://github.com/Ewenwan/MVision/blob/master/CNN/HighPerformanceComputing/img/Single-Elements.jpg))

这种方法比前面的两种方法速度要慢，每个元素的处理都需要单独进行；

这种的剩余元素处理方法需要两个迭代循环，第一个处理向量的循环，还有处理剩余元素的循环，这会增加代码大小；

NEON的单一元素加载只改变目标元素的值，而保留其他的元素不变，如果你向量计算的指令会在一个向量间反复计算，如VPADD，这些寄存器需要在第一个元素加载时初始化。


* 4. 或者剩余的单个元素直接使用C语言进行计算


### 示例3：从内存变量 加载数据 到 寄存器向量
```c
#include <stdio.h>
#include <arm_neon.h>
unsigned short int A[] = {1,2,3,4}; 
    // 含有四个无符号短整型整数的数组 array with 4 elements
int main(void)
{
	uint16x4_t v;     // 4通道16位的向量declare a vector of four 16-bit lanes
	v = vld1_u16(A);  // 从数组加载到向量load the array from memory into a vector
	v = vadd_u16(v,v);// 每个元素加上自身，扩大一倍double each element in the vector
	vst1_u16(A, v);   // 存储结果回数组A store the vector back to memory
	return 0;
}
```


### 示例4：直接从数据创建vcreate_u8()寄存器变量
```c
#include <arm_neon.h>
int main (void)
{
	uint8x8_t v;        // 定义一个8通道个8位数据的向量
	unsigned char A[8]; // 分配内存存储一个含有8个无符号字符数据的数组
	v = vcreate_u8(0x0102030405060708); // 创建一个8X8位向量，存储 1,2,3,4,5,6,7,8
	vst1_u8(A, v);      // 将向量数据 存储到内存
	return 0;
}

```

### 示例5：加载多个向量数据
```c
#include <arm_neon.h>
int main (void)
{
	uint8x8x3_t v; // 定义一个包含3个向量的向量数组，每个向量为8通道8位无符号整形
	unsigned char A[24]; // 定义一个包含24个无符号字节数据的数组，表示24个像素
	v = vld3_u8(A);      // 从A处加载数据(多向量间隔加载)
	// v.val[0] 是第一个向量={A[0],A[3],A[6],A[9],A[12],A[15],A[18],A[21]},RGB红色通道
	// v.val[1] 是第二个向量={A[1],A[4],A[7],A[10],A[13],A[16],A[19],A[22]},RGB绿色通道
	// v.val[2] 是第三个向量={A[2],A[5],A[8],A[11],A[14],A[17],A[20],A[23]},RGB蓝色通道
	v.val[0] = vadd_u8(v.val[0],v.val[0]);// 红色通道数值加倍
	vst3_u8(A, v); // 在把使用向量处理后的数据，存回内存数组A中
	return 0;
}

```
vld3_u8：

![](https://github.com/Ewenwan/MVision/blob/master/CNN/HighPerformanceComputing/img/rgb-3.PNG)	

vswp_u8: 交换R和B通道

![](https://github.com/Ewenwan/MVision/blob/master/CNN/HighPerformanceComputing/img/rgb-bgr.jpg)	


vld1_u8：

![](https://github.com/Ewenwan/MVision/blob/master/CNN/HighPerformanceComputing/img/rgb.PNG)	

加载和保存：

![](https://github.com/Ewenwan/MVision/blob/master/CNN/HighPerformanceComputing/img/rgb-store.PNG)	

### 示例6：数组矩阵相乘

    列主导4*4矩阵相乘：

![](https://github.com/Ewenwan/MVision/blob/master/CNN/HighPerformanceComputing/img/matrixMul.PNG)

细节-结果矩阵的产生：

![](https://github.com/Ewenwan/MVision/blob/master/CNN/HighPerformanceComputing/img/matrixMul_COL.PNG)

结果矩阵的第一列：

A矩阵第一列和B矩阵第一列的第一个元素相乘 +
A矩阵第二列和B矩阵第一列的第二个元素相乘 +
A矩阵第三列和B矩阵第一列的第三个元素相乘 +
A矩阵第四列和B矩阵第一列的第四个元素相乘 

```c
void altneonmult(const float *matrixA, const float *matrixB, float *matrixR)
// matrixA \ matrixB \ matrixR均为 4*4 浮点数矩阵，列优先存储??
// 计算过程为 matrixR = matrixA * matrixB
{
	float32x4_t a, b0, b1, b2, b3, r;// 4通道32位浮点数  行row 列column
	a0 = vld1q_f32(matrixA);     /* A矩阵第一列 从内存地址加载数据，连续加载，4个32位共128位数据*/
	a1 = vld1q_f32(matrixA + 4); /* A矩阵第二列*/
	a2 = vld1q_f32(matrixA + 8); /* A矩阵第三列*/
	a3 = vld1q_f32(matrixA + 12); /* A矩阵第四列 */
	
// 结果矩阵的第一列
	b = vld1q_f32(matrixB); /* B矩阵第一列 */
	r = vmulq_lane_f32(a0, vget_low_f32(b), 0);     // A矩阵第一列 乘 B矩阵第一列的第一个元素
	r = vmlaq_lane_f32(r, a1, vget_low_f32(b), 1);  // 乘加
	r = vmlaq_lane_f32(r, a2, vget_high_f32(b), 0);
	r = vmlaq_lane_f32(r, a3, vget_high_f32(b), 1);
	vst1q_f32(matrixR, r); /* store col 0 of result */
// 结果矩阵的第二列
	b = vld1q_f32(matrixB + 4); /* B矩阵第二列 */
	r = vmulq_lane_f32(a0, vget_low_f32(b), 0);
	r = vmlaq_lane_f32(r, a1, vget_low_f32(b), 1);
	r = vmlaq_lane_f32(r, a2, vget_high_f32(b), 0);
	r = vmlaq_lane_f32(r, a3, vget_high_f32(b), 1);
	vst1q_f32(matrixR + 4, r); /* store col 1 of result */
// 结果矩阵的第三列
	b = vld1q_f32(matrixB + 8); /* B矩阵第三列 */
	r = vmulq_lane_f32(a0, vget_low_f32(b), 0);
	r = vmlaq_lane_f32(r, a1, vget_low_f32(b), 1);
	r = vmlaq_lane_f32(r, a2, vget_high_f32(b), 0);
	r = vmlaq_lane_f32(r, a3, vget_high_f32(b), 1);
	vst1q_f32(matrixR + 8, r); /* store col 2 of result */
// 结果矩阵的第四列
	b = vld1q_f32(matrixB + 12); /* B矩阵第四列 */
	r = vmulq_lane_f32(a0, vget_low_f32(b), 0);
	r = vmlaq_lane_f32(r, a1, vget_low_f32(b), 1);
	r = vmlaq_lane_f32(r, a2, vget_high_f32(b), 0);
	r = vmlaq_lane_f32(r, a3, vget_high_f32(b), 1);
	vst1q_f32(matrixR + 12, r); /* store col 3 of result */
}

// 先提取 再计算 最后存取
void neonmult(const float *matrixA, const float *matrixB, float *matrixR)
{
// 0. 定义变量
	float32x4_t a0, a1, a2, a3, b0, b1, b2, b3, r0, r1, r2, r3;
	
// 1. 先提取每个矩阵的每一列
	a0 = vld1q_f32(matrixA); /* col 0 of matrixA */
	a1 = vld1q_f32(matrixA + 4); /* col 1 of matrixA */
	a2 = vld1q_f32(matrixA + 8); /* col 2 of matrixA */
	a3 = vld1q_f32(matrixA + 12); /* col 3 of matrixA */
	
	b0 = vld1q_f32(matrixB); /* col 0 of matrixB */
	b1 = vld1q_f32(matrixB + 4); /* col 1 of matrixB */
	b2 = vld1q_f32(matrixB + 8); /* col 2 of matrixB */
	b3 = vld1q_f32(matrixB + 12); /* col 3 of matrixB */
	
// 2. 计算结果矩阵的每一列
	/* compute all the cols in the order specified by compiler */
        // 第一列
	r0 = vmulq_lane_f32(a0, vget_low_f32(b0), 0);     // 乘 
	r0 = vmlaq_lane_f32(r0, a1, vget_low_f32(b0), 1); // 乘加
	r0 = vmlaq_lane_f32(r0, a2, vget_high_f32(b0), 0);// 乘加
	r0 = vmlaq_lane_f32(r0, a3, vget_high_f32(b0), 1);// 乘加
	//第二列
	r1 = vmulq_lane_f32(a0, vget_low_f32(b1), 0);
	r1 = vmlaq_lane_f32(r1, a1, vget_low_f32(b1), 1);
	r1 = vmlaq_lane_f32(r1, a2, vget_high_f32(b1), 0);
	r1 = vmlaq_lane_f32(r1, a3, vget_high_f32(b1), 1);
	//第三列
	r2 = vmulq_lane_f32(a0, vget_low_f32(b2), 0);
	r2 = vmlaq_lane_f32(r2, a1, vget_low_f32(b2), 1);
	r2 = vmlaq_lane_f32(r2, a2, vget_high_f32(b2), 0);
	r2 = vmlaq_lane_f32(r2, a3, vget_high_f32(b2), 1);
	//第四列
	r3 = vmulq_lane_f32(a0, vget_low_f32(b3), 0);
	r3 = vmlaq_lane_f32(r3, a1, vget_low_f32(b3), 1);
	r3 = vmlaq_lane_f32(r3, a2, vget_high_f32(b3), 0);
	r3 = vmlaq_lane_f32(r3, a3, vget_high_f32(b3), 1);
	
// 3. 存储设置结果矩阵
	vst1q_f32(matrixR, r0);    // 第一列
	vst1q_f32(matrixR + 4, r1);//第二列
	vst1q_f32(matrixR + 8, r2);//第三列
	vst1q_f32(matrixR + 12, r3);//第四列
}
```
### 示例7： 向量叉乘 Cross product

a = [ai, aj, ak]

b = [bi, bj, bk]

> ** r = a 叉乘 b = [aj*bk-ak*bj, ak*bi-ai*bk, ai*bj-aj*bi]**

```c
// Single cross product===== 单叉积?
void cross_product_s(float32_t *r, float32_t* a, float32_t* b)
{
	// 向量存储 ai bi在低地址，ak bk在高地址
	// 寄存器内存 register for example:
	// [element3, element2, element1, element0]  element0低地址  element3高地址
	float32x2_t vec_a_1 = vld1_f32(a + 1); //D register = [ak, aj]  aj低地址
	float32x2_t vec_a_2 = vld1_f32(a);     //D register = [aj, ai]  ai低地址
	
	float32x2_t vec_b_1 = vld1_f32(b + 1); //D register = [bk, bj]  bj低地址
	float32x2_t vec_b_2 = vld1_f32(b);     //D register = [bj, bi]  bi低地址
	
	// 寄存器合并 combine
	float32x4_t vec_a = vcombine_f32(vec_a_1, vec_a_2); //Q register = [aj, ai, ak, aj]
	float32x4_t vec_b = vcombine_f32(vec_b_1, vec_b_2); //Q register = [bj, bi, bk, bj]
        // 寄存器移通道 低位通道数据到最高位通道，其他数据依次往低位通道移动
	float32x4_t vec_a_rot = vextq_f32(vec_a, vec_a, 1); //Q register = [ aj, aj, ai, ak ] 
	float32x4_t vec_b_rot = vextq_f32(vec_b, vec_b, 1); //Q register = [ bj, bj, bi, bk ]
	
	// vec_a = [ aj, ai, ak, aj ]
	// vec_b_rot = [ bj, bj, bi, bk ]
	// vec_a_rot = [ aj, aj, ai, ak ]
	// vec_b = [ bj, bi, bk, bj ]
	
	float32x4_t prod = vmulq_f32(vec_a, vec_b_rot); // 乘
	// prod = [ ajbj, aibj, akbi, ajbk ]
	
        // vec_a_rot*vec_b = [aj*bj, aj*bi, ai*bk, ak*bj]
	prod = vmlsq_f32(prod, vec_a_rot, vec_b);// 乘  再 减  prod - vec_a_rot * vec_b
	// prod = [ ajbj-ajbj, aibj-ajbi, akbi-aibk, ajbk-akbj ]
	
	vst1_f32(r, vget_low_f32(prod)); // 先存储低位两个通道  [XXX, akbi-aibk, ajbk-akbj]
	vst1_lane_f32(r + 2, vget_high_f32(prod), 0); // 再存储第三个通道 [aibj-ajbi, akbi-aibk, ajbk-akbj]
}


// Four cross products
void cross_product_q(float32_t* r, float32_t* a, float32_t* b)
{
	float32x4x3_t vec_a = vld3q_f32(a); // [,,,ai]  0
	                                    // [,,,aj]  1
					    // [,,,ak]  2
					    
	float32x4x3_t vec_b = vld3q_f32(b); // [,,,bi]  0
	                                    // [,,,bj]  1
					    // [,,,bk]  2
	float32x4x3_t result;
	
	result.val[0] = vmulq_f32(vec_a.val[1], vec_b.val[2]); // 乘 aj*bk
	result.val[0] = vmlsq_f32(result.val[0], vec_a.val[2], vec_b.val[1]); // 乘减 aj*bk - ak*bj
	
	result.val[1] = vmulq_f32(vec_a.val[2], vec_b.val[0]); // 乘 ak*bi
	result.val[1] = vmlsq_f32(result.val[1], vec_a.val[0], vec_b.val[2]); // 乘减 ak*bi - ai*bk
	
	result.val[2] = vmulq_f32(vec_a.val[0], vec_b.val[1]); // 乘 ai*bj
	result.val[2] = vmlsq_f32(result.val[2], vec_a.val[1], vec_b.val[0]); // 乘减 ai*bj - aj*bi
	
	vst3q_f32(r, result);
}
```

### 示例7： 向量的点积 Dot product
A = (a1,a2,a3,...,an)

B = (b1,b2,b3,...,bn)

A * B = a1b1 + a2b2 + a3b3 + ... + anbn

向量的每一维相乘然后相加，相乘之间具有良好的并行性，所以可以通过ARM NEON intrinsic指令进行加速。下面是代码实现：

```c
// 浮点数 
float dot(float* A,float* B,int K)
{
    float sum=0;
    float32x4_t sum_vec=vdupq_n_f32(0); // 和向量，从立即数创建数据
    float32x4_t left_vec,right_vec;     // 向量A 和 向量 B
    for(int k=0; k<K; k+=4) // 这里默认K为4倍数，未考虑剩余数据
    {
        left_vec  = vld1q_f32(A + k); // 先将两个数组每次4个存入ARM NEON intrinsic下的128位变量中
        right_vec = vld1q_f32(B + k);
        sum_vec   = vmlaq_f32(sum_vec,left_vec,right_vec);// 乘加,利用一个乘加指令计算4个乘积的累加和。
    }
    
    // 最后将4个sum再相加就得到最终的结果。
    float32x2_t r = vadd_f32(vget_high_f32(sum_vec),vget_low_f32(sum_vec));// 两两相加
    sum += vget_lane_f32(vpadd_f32(r,r),0);

    return sum;
}

// 相比于串行代码，上面的代码有接近4倍的加速比。当数据类型是short或者char时，可以取得更高的加速比，下面以char举例：

int dot(char* A,char* B,int K)
{
    int sum=0;
    int16x8_t sum_vec=vdupq_n_s16(0);// 128位 和向量，从立即数创建数据
    int8x8_t left_vec, right_vec;    // 64位  向量A 和 向量B
    int32x4_t part_sum4; // 4个32位 128位寄存器 和
    int32x2_t part_sum2; // 2个32位 64位寄存器  和

    //有溢出的风险
    for(k=0; k<K; k+=8)
    {
        left_vec  = vld1_s8(A + A_pos + k);
        right_vec = vld1_s8(B + B_pos + k);
        sum_vec   = vmlal_s8(sum_vec,left_vec,right_vec);
    }

    part_sum4=vaddl_s16(vget_high_s16(sum_vec),vget_low_s16(sum_vec));   
    part_sum2=vadd_s32(vget_high_s32(part_sum4),vget_low_s32(part_sum4));
    sum+=vget_lane_s32(vpadd_s32(part_sum2,part_sum2),0);

    return sum;
}
```
### 示例8：3x3  pool 池化代码 最大值/均值池化

```c
// 先分别读取三列
constexpr const int pool_size = 3;
const float32x4_t top_data    = vld1q_f32(reinterpret_cast<const float *>(input_top_ptr + input.offset()));
const float32x4_t middle_data = vld1q_f32(reinterpret_cast<const float *>(input_middle_ptr + input.offset()));
const float32x4_t bottom_data = vld1q_f32(reinterpret_cast<const float *>(input_bottom_ptr + input.offset()));

float32x2_t       res         = {};
if(pooling_type == PoolingType::AVG)
{// 均值池化=============
   // Calculate scale
   float scale = calculate_avg_scale(id, pool_size, upper_bound_w, upper_bound_h, pool_pad_x, pool_pad_y, pool_stride_x, pool_stride_y);
   const float32x2_t scale_v = vdup_n_f32(scale);// 寄存器 初始化为 scale 2个32位

   // Perform pooling
   const float32x4_t sum_data = vaddq_f32(vaddq_f32(top_data, bottom_data), middle_data);
   res = vpadd_f32(vget_high_f32(vsetq_lane_f32(0.f, sum_data, 3)), vget_low_f32(sum_data));
   res  = vmul_f32(vpadd_f32(res, res), scale_v);// 得到4个最大的float 
}
else
{// 最大值池化
   const float32x4_t max_data = vmaxq_f32(vmaxq_f32(top_data, bottom_data), middle_data);
   res = vpmax_f32(vget_high_f32(vsetq_lane_f32(-std::numeric_limits<float>::max(), max_data, 3)), vget_low_f32(max_data));
   res = vpmax_f32(res, res);
}

*(reinterpret_cast<float *>(output.ptr())) = vget_lane_f32(res, 0);

```
## 4. NEON assembly

采用汇编语言进行NEON(**NEON 汇编（assembly）**)的最底层优化，可以使优化性能最大化，但汇编语言比较灵活，手写汇编程序对开发人员来说具有较大挑战，如果使用不恰当，反而会影响优化性能。

NEON可以有两种写法：
* 1. Assembly文件： 纯汇编文件，后缀为”.S”或”.s”。注意对寄存器数据的保存。
* 2. inline assembly内联汇编

在C/C++程序中编写汇编代码主要有两种形式：汇编函数或内联汇编。汇编函数中，需要声明代码段、操作堆栈等，过于复杂。而编写内联汇编，在C代码中需要以“asm”关键字标识，并在asm（）编写汇编语句。这种方法只需要在待优化部分局部采用汇编语言实现，相对简单。


### 数据加载保存移动

> **扩展 寄存器 加载和存储 指令**

语法：
```asm
VLDR{cond}{.size} Fd, [Rn{, #offset}]   # load加载，从内存中加载一个扩展寄存器。
VSTR{cond}{.size} Fd, [Rn{, #offset}]   # set保存，将一个扩展寄存器的内容保存到内存中。
VLDR{cond}{.size} Fd, label
VSTR{cond}{.size} Fd, label
```
cond: 是一个可选的条件代码，EQ等于\NE不等于\HI无符号大于\LS无符号小于等于\GE有符号大于等于\LT有符号小于\GT有符号大于\LE有符号小于等于

size：是一个可选的数据大小说明符。 如果 Fd 是单精度 VFP 寄存器，则必须为 32，传送一个字；否则必须为 64，传送两个字。

Fd：是要加载或保存的扩展寄存器。 对于 NEON 指令，它必须为 Dd。 对于 VFP 指令，它可以为 Dd 或 Sd。

Rn：是存放要传送的基址的 ARM 寄存器。

offset：是一个可选的数值表达式。 在汇编时，该表达式的值必须为一个数字常数。 该值必须是 4 的倍数，并在 -1020 到 +1020 的范围内。 该值被加到基址上以构成用于传送的地址。

label：是一个程序相对的表达式。必须位于当前指令的 ±1KB 范围之内。

> **扩展寄存器加载多个、存储多个、从堆栈弹出、推入堆栈**

语法:
```asm
VLDMmode{cond} Rn,{!} Registers # 加载多个
VSTMmode{cond} Rn,{!} Registers # 存储多个
VPOP{cond} Registers            # 从堆栈弹出 VPOP Registers 等效于 VLDM sp!,Registers
VPUSH{cond} Registers           # 推入堆栈   VPUSH Registers 等效于 VSTMDB sp!,Registers
```
mode 必须是下列值之一：

	IA 表示在每次传送后递增地址。IA 是缺省值，可以省略。 increase
	DB 表示在每次传送前递减地址。 decrease
	EA 表示空的升序堆栈操作。 对于加载操作，该值与 DB 相同；对于保存操作，该值与 IA 相同。
	FD 表示满的降序堆栈操作。 对于加载操作，该值与 IA 相同；对于保存操作，该值与 DB 相同。

! 是可选的。! 指定必须将更新后的基址写回到 Rn 中。 如果未指定!，则 mode 必须为 IA。

Registers 是一个用大括号 { 和 } 括起的连续扩展寄存器的列表。 该列表可用逗号分隔，也可以采用范围格式。 列表中必须至少有一个寄存器。可指定 S、D 或 Q 寄存器，但一定不能混用这些寄存器。 D 寄存器的数目不得超过 16 个，Q 寄存器的数目不得超过 8 个。 如果指定 Q 寄存器，则在反汇编时它们将显示为 D 寄存器。

> **VMOV（在两个 ARM 寄存器和一个扩展寄存器之间传送内容）**

在两个 ARM 寄存器与一个 64 位扩展寄存器或两个连续的 32 位 VFP 寄存器之间传送内容。

语法:
```asm
VMOV{cond} Dm, Rd, Rn # 将 Rd 的内容传送到 Dm 的低半部分，并将 Rn 的内容传送到 Dm 的高半部分
VMOV{cond} Rd, Rn, Dm # 将 Dm 的低半部分的内容传送到 Rd，并将 Dm 的高半部分的内容传送到 Rn
VMOV{cond} {Sm, Sm1}, Rd, Rn # 将 Sm 的内容传送到 Rd，并将 Sm1 的内容传送到
VMOV{cond} Rd, Rn, {Sm, Sm1} # 将 Rd 的内容传送到 Sm，并将 Rn 的内容传送到 Sm1
```

	Dm 是一个 64 位扩展寄存器。
	Sm 是一个 VFP 32 位寄存器。
	Sm1 是 Sm 之后的下一个 VFP 32 位寄存器。
	Rd、Rn 是 ARM 寄存器。 不要使用 r15。
> **VMOV（在一个 ARM 寄存器R 和一个 NEON 标量之间）**

在一个 ARM 寄存器和一个 NEON 标量之间传送内容。

语法
VMOV{cond}{.size} Dn[x], Rd     # 将 Rd 的最低有效字节、半字或字的内容传送到 Sn。
VMOV{cond}{.datatype} Rd, Dn[x] # 将 Dn[x] 的内容传送到 Rd 的最低有效字节、半字或字。

size 是数据大小。 可以为 8、16 或 32。 如果省略，则 size 为 32。

datatype 是数据类型。 可以为 U8、S8、U16、S16 或 32。 如果省略，则 datatype为 32。

Dn[x] 是 NEON 标量,16 位标量限定为寄存器 D0-D7，其中 x 位于范围 0-3 内,32 位标量限定为寄存器 D0-D15，其中 x 为 0 或 1。

Rd 是 ARM 寄存器。Rd 不得为 R15。

#### NEON 逻辑运算和比较运算
> **VAND、VBIC、VEOR、VORN 和 VORR（寄存器）**

VAND（按位与）、VBIC（位清除）、VEOR（按位异或）、VORN（按位或非）和 VORR（按位或）指令在两个寄存器之间执行按位逻辑运算，并将结果存放到目标寄存器中。

语法:
```asm
Vop{cond}.{datatype} {Qd}, Qn, Qm
Vop{cond}.{datatype} {Dd}, Dn, Dm
```

op 必须是下列值之一：
AND 逻辑“与”\ORR 逻辑“或”\EOR 逻辑异或\BIC 逻辑“与”求补\ORN 逻辑“或”求补。

Qd、Qn、Qm 为四字运算指定目标寄存器、第一个操作数寄存器和第二个操作数寄存器。

Dd、Dn、Dm 为双字运算指定目标寄存器、第一个操作数寄存器和第二个操作数寄存器。

> **VBIC 和 VORR（立即数）**

VBIC（位清除（立即数））获取目标向量的每个元素，对其与一个立即数执行按位与求补运算，并将结果返回到目标向量。

VORR（按位或（立即数））获取目标向量的每个元素，对其与一个立即数执行按位或运算，并将结果返回到目标向量。


语法:
```asm
Vop{cond}.datatype Qd, #imm
Vop{cond}.datatype Dd, #imm
```
op 必须为 BIC 或 ORR。

datatype 必须为 I16 或 I32。

Qd 或 Dd 是用于存放源和结果的 NEON 寄存器。

imm 是立即数。

立即数 

如果 datatype 为 I16，则立即数必须采用下列格式之一：
• 0x00XY
• 0xXY00。

如果 datatype 为 I32，则立即数必须采用下列格式之一：
• 0x000000XY
• 0x0000XY00
• 0x00XY0000
• 0xXY000000。

〉**VBIF、VBIT 和 VBSL**

VBIT（为 True 时按位插入）：如果第二个操作数的对应位为 1，则该指令将第一个操作数中的每一位插入目标中；否则将目标位保持不变。

VBIF（为 False 时按位插入）：如果第二个操作数的对应位为 0，则该指令将第一个操作数中的每一位插入目标中；否则将目标位保持不变。

VBSL（按位选择）：如果目标的对应位为 1，则该指令从第一个操作数中选择目标的每一位；如果目标的对应位为 0，则从第二个操作数中选择目标的每一位。

语法：
```asm
Vop{cond}{.datatype} {Qd}, Qn, Qm
Vop{cond}{.datatype} {Dd}, Dn, Dm

```

> **VMOV、VMVN（寄存器）**

VMOV向量移动（寄存器）将源寄存器中的值复制到目标寄存器中。

VMVN向量求反移动（寄存器）对源寄存器中每一位的值执行求反运算，并将结果存放到目标寄存器中。


语法:
```asm
VMOV{cond}{.datatype} Qd, Qm
VMOV{cond}{.datatype} Dd, Qm
VMVN{cond}{.datatype} Qd, Qm
VMVN{cond}{.datatype} Dd, Qm
```

### NEON 乘法指令

VMUL（向量乘法））将两个向量中的相应元素相乘，并将结果存放到目标向量中。
VMLA（向量乘加）将两个向量中的相应元素相乘，并将结果累加到目标向量的元素中。
VMLS（向量乘减）将两个向量中的相应元素相乘，从目标向量的相应元素中减去相乘的结果，并将最终结果放入目标向量中。
语法:
```asm
Vop{cond}.datatype {Qd}, Qn, Qm
Vop{cond}.datatype {Dd}, Dn, Dm
VopL{cond}.datatype Qd, Dn, Dm
```



### 内联汇编 inline assembly
[ARM GCC Inline Assembler Cookbook](http://www.ethernut.de/en/documents/arm-inline-asm.html)

[博客参考](https://blog.csdn.net/dahailantian1/article/details/78584920)

优点：在C代码中嵌入汇编，调用简单，无需手动存储寄存器；
缺点：有较为复杂的格式需要事先学习，不好移植到其他语言环境。

[汇编语言笔记](https://github.com/Ewenwan/ShiYanLou/blob/master/OS/%E6%B1%87%E7%BC%96%E8%AF%AD%E8%A8%80.md)

[内联汇编参考](https://github.com/Ewenwan/ShiYanLou/tree/master/OS/Linux#c内联汇编)

比如上述intrinsics代码产生的汇编代码为：
```c
// ARMv7‐A/AArch32
void add_float_neon2(int* dst, int* src1, int* src2, int count)
{
	asm volatile (
		"1: \n"                        // 用于构成循环的标记号
		"vld1.32 {q0}, [%[src1]]! \n"  // 从src地址处载入4个32位的浮点数 地址递增
		"vld1.32 {q1}, [%[src2]]! \n"
		"vadd.f32 q0, q0, q1 \n"       // q0 = q0 +q1
		"subs %[count], %[count], #4 \n"// 循环计数count = count-4
		"vst1.32 {q0}, [%[dst]]! \n"   // 将运算结果存储到目标地址，目标地址递增
		"bgt 1b \n"                    // 如果count>0,跳转到标记号1处继续执行
		: [dst] "+r" (dst)             // 可写
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

〉 **总结一下NEON优化就是：**

* 第一优化算法实现流程；
* 第二优化程序存取；
* 第三优化程序执行；
* 第四哪儿能优化，就优化哪儿

〉 **需要注意的地方**

   1. load数据的时候，第一次load会把数据放在cache里面，只要不超过cache的大小，下一次load同样数据的时候，则会比第一次load要快很多，会直接从cache中load数据，这样在汇编程序设计的时候是非常需要考虑的问题。

   如：求取一个图像的均值，8*8的窗口，先行求和，然后列求和出来均值，这时候会有两个函数，数据会加载两遍，如果按照这样去优化的话则优化不了多少。如果换成上面这种思路，先做行16行，然后再做列，这样数据都在cache里面，做列的时候load数据会很快。

   在做neon乘法指令的时候会有大约2个clock的阻塞时间，如果你要立即使用乘法的结果，则就会阻塞在这里，在写neon指令的时候需要特别注意。乘法的结果不能立即使用，可以将一些其他的操作插入到乘法后面而不会有时间的消耗。

如：vmul.u16 q1, d3, d4 

         vadd.u32 q1, q2, q3

此时直接使用乘法的结果q1则会阻塞，执行vadd需要再等待2个clock的时间

使用饱和指令的时候，如乘法饱和的时候，在做乘法后会再去做一次饱和，所以时间要比直接做乘法要慢。

如：  vmul.u16 q1, d3, d4

          vqmul.u32 q1, q2, q3

后一个的时间要比第一个的时间要久。

在对16位数据进行load或者store操作的时候，需要注意的是字节移位。比如是16位数据，则load 8个16位数据，如果指定寄存器进行偏移，此时需要特别注意。

例如：vld1.64 {d0}, [r0], r1


## 内联汇编使用心得
[ARM GCC Inline Assembler Cookbook](http://www.ethernut.de/en/documents/arm-inline-asm.html)

inline assembly下面的三个冒号一定要注意
output/input registers的写法一定要写对，clobber list也一定要写完全，否则会造成令你头疼的问题 (TT)

这个问题在给出的cookbook中也有介绍，但是并不全面，有些问题只有自己碰到了再去解决。 笔者就曾经被虐了很久，从生成的汇编发现编译器将寄存器乱用，导致指针操作完全混乱，毫无头绪…


一般情况下建议的写法举例：
```asm
asm volatile (
	... /* assembly code 汇编代码 */
	// 所有的汇编代码必须用双引号括起来。
        // 如果有多行汇编代码的话，每一条语句都要用双引号括起来，并且在代码后面要加上换行符（“\n”或者“\n\t”）。
	
	// "[modifier修改符 可选]constraint限定符" (C expression C语言表达式)
	// 修改符和限定符要用双引号括起来，而C表达式要用括号括起来。
	: "+r"(arg0) // %0
	  "+r"(arg1) // %1 // 输入寄存器 Output Registers
	: "r"(arg2)  // %2 // 输入寄存器 Input Registers
	: "cc", "memory", r0, r1  // 寄存器变化
);
```

> **限定符**

	限定符   在ARM指令集下              在Thumb指令集下
	f         浮点寄存器f0...f7              N/A
	h         N/A                           寄存器r8...r15
	G         浮点常量立即数                 N/A
	H         和G作用相同                    N/A
	I         数据处理指令中用到的立即数      范围为0...255的常量
	J         范围为-4095...4095的索引常量    范围为-255...-1的常量
	K         和I作用相同                    和I作用相同
	L         和I作用相同                    范围为-7...7的常量
	l         和r作用相同                    寄存器r0...r7
	M         范围为0.32或者是2的幂次方的常量  范围为0...1020的4的倍数的常量
	m         内存地址memory                 内存地址
	N         N/A                           范围为0...31的常量
	O         N/A                           范围为 -508...508 的4的倍数的常量
	r         通用寄存器r0...r15             N/A
	w         向量浮点寄存器s0...s31         N/A
	X         任何类型的操作数               任何类型的操作数
        
	数字 0，1，2，3，... 指代前面定义的操作数
	
是常用的也就是r，f和m等几个。

> **修改符**

修改符是加在限定符之前的，并且是可选的，如果没有修改符的话，则表明这个操作数是只读的。

这个对输入操作数没有问题，但是对输出操作数来说，肯定是需要被修改的，那怎么办呢？

答案就是使用修改符，修改这个操作数的属性。目前，GCC中定义了三个修改符，分别是：

	修改符    含义
	=        只写 操作数，通常用于输出操作数中
	+        可读 且 可写 操作数，必须要列在输出操作数中
	&        寄存器只能用于输出(不能作为输入寄存器)
	
所以，作为输出操作数，只需要在限定符前加上“=”就可以了。

如果想让一个C变量既作为输入操作数，也作为输出操作数的话，可以使用“+”限定符，并且这个操作数只需要在输出操作数列表中列出就行了。例如:

```asm
__asm__(
        "mov %0, %0, ror #1"   
        : "+r" (y)  
        );  
```
是将变量y中的值右移1位。因为输入和输出操作数是一个，所以该操作数要既可读也可写，因此添加了“+”修改符。

其实，在限定符中，也可以使用数字，其作用是指代前面定义的操作数，0代表第一个，1代表第二个，以此类推。


```asm
__asm__(
        "mov %0, %0, ror #1"   
        : "=r" (y)  
        : "0" (y)  
        );  
```
// 这个例子的效果和前面的例子是相同的。本例不同的是，先定义了一个可写的输出变量，同时在输入变量列表中，明确用数字0指出了前面定义的第一个操作数同时也要用来作为输入操作数。


使用“&”修改符，明确告诉编译器，代表输出操作数的寄存器一定不能使用输入操作数已经使用过的寄存器。下面举个例子：

如果汇编代码中有输入寄存器还没有使用完毕，就对输出操作数进行修改的情况，则特别需要用“&”修改符，保证不复用。

```asm
__asm__ __volatile__(
                 "ldr %0, [%1]\n\t"  
                 "str %2, [%1, #4]"  
                 : "=&r" (rdv)  
                 : "r" (&table), "r" (wdv)  
                 : "memory");  

```
本例中，将操作一个table数组，读出它的第一个数存放到rdv中，然后修改第二个数为wdv中存放的值。乍看一下没什么问题，但是如果编译器用同一个寄存器来表示输入操作数&table（%1）和输出操作数rdv（%0）怎么办呢？执行完第一条语句之后，table数组的地址就被修改掉了。所以，可以在输出操作数中加上一个“&”修改符，强制保证输出操作数不能和输入操作数复用同一个寄存器，这个问题就解决了

> **修改寄存器列表**

在汇编指令中，有可能会用到一些指定的寄存器，但是在执行你定义的汇编程序时，那个指定的寄存器有可能另有别的用途，存放了非常重要的数据。等你的程序执行完成后，那个寄存器的值已经被你修改过了，肯定会造成执行错误。因此，在执行你的程序之前必须要做必要的备份和恢复的动作。但是，编译器并不会分析你的汇编代码，找出这种被你修改过，需要恢复的寄存器，因此你必须显式的告诉编译器，被你修改过的寄存器有哪些。这就是修改寄存器列表所起到的作用。

对于嵌入内联ARM汇编来说，此列表中的值有下面三种类型：

	类型           作用
	r0...r15     告诉编译器汇编代码中 修改了通用寄存器r0...r15
	cc           告诉编译器汇编代码 会 导致 CPU状态位 的 改变	memory       告诉编译器汇编代码 会 读取或修 改内存中某个地址 存放的值

对于“memory”来说，它并不是表示寄存器被读取或修改了，而是表示内存中的值被修改了。出于优化的目的，在执行你的汇编代码之前，编译器将某些变量的值还保存在寄存器中，并没有被写到实际的内存中。但是，如果你的汇编代码会读取内存中的值，则很有可能新的值还在寄存器中，而内存中存放的还是老的值，这样就会造成错误。添加了“memory”之后，编译器会在执行你的代码之前，保证将保存在寄存器中，没有更新到内存中的值全部都写入到内存中。

此列表中的每一项都要用双引号（""）括起来，每项之间要用逗号（“,”）分割。 



### 浮点向量加法 NEON instruction 内联函数 Inline assembly内联汇编  NEON assembly 纯汇编 对比
[Neon 寄存器 指令集 ARMv7/v8 对比](https://blog.csdn.net/zsc09_leaf/article/details/45825015)

// c 与内联函数对比
```c
#include<arm_neon.h>
 
void add_float_c(float* dst, float* src1, float* src2, int count)
{
     int i;
     for (i = 0; i < count; i++)
         dst[i] = src1[i] + src2[i];
}
 
void add_float_neon1(float* dst, float* src1, float* src2, int count)
{
     int i = 0;
     for (; i < count - 3; i += 4)
     {
         float32x4_t in1, in2, out;
         in1 = vld1q_f32(src1);
         src1 += 4;
         in2 = vld1q_f32(src2);
         src2 += 4;
	 // v8
	 #if __aarch64__
             out = vaddvq_f32(in1, in2);
	 #else
             out = vaddq_f32(in1, in2);
	 #endif
         vst1q_f32(dst, out);
         dst += 4;
     }
     // 剩余 1~3个数 使用普通c
     for(;i < count; i++)
     {
         dst[i] = src1[i] + src2[i]
     }
}



```

// 内联函数 V7 V8 对比
```c
// ARMv7-A/AArch32
void add_float_neon3(float* dst, float* src1, float* src2, int count)
{
    int nn = count >> 4;
    int remain = count - (nn << 2);
/*
    asm volatile (
               "1:                                           \n" // 用于循环跳转，标记号
               "vld1.32         {q0}, [%4]!                  \n"
               "vld1.32         {q1}, [%5]!                  \n"
               "vadd.f32        q0, q0, q1                   \n"
               "subs            %1, #1                       \n"
               "vst1.32         {q0}, [%0]!                  \n"
               "bgt             1b                           \n"
               : "+r"(dst),     // %0 输出参数列表
	         "+r"(nn)       // %1
	       : "0"(dst)     
	         "1"(nn)
                 "r"(src1),     // %4 输入参数列表
	         "r"(src2)      // %5
               : "memory", "q0", "q1"
          );
*/
    asm volatile (
               "1:                                           \n" // 用于循环跳转，标记号
               "vld1.32         {q0}, [%[src1]]!             \n"
               "vld1.32         {q1}, [%[src2]]!             \n"
               "vadd.f32       q0, q0, q1                    \n"
               "subs            %[nn], %[nn], #4       \n"
               "vst1.32         {q0}, [%[dst]]!              \n"
               "bgt             1b                           \n"
               : [dst] "+r" (dst)
               : [src1] "r" (src1), [src2] "r" (src2), [nn] "r" (nn)
               : "memory", "q0", "q1"
          );
    // 剩余数处理  
    for( ; remain > 0; remain--)
    {
        *dst = *src1 + *src2;
    }
}


// AArch64
void add_float_neon3(float* dst, float* src1, float* src2, int count)
{
    asm volatile (
               "1:                                           \n" // 用于循环跳转，标记号
               "ld1             {v0.4s}, [%[src1]], #16      \n"
               "ld1             {v1.4s}, [%[src2]], #16      \n"
               "fadd            v0.4s, v0.4s, v1.4s          \n"
               "subs            %[count],  %[count], #4      \n"
               "st1             {v0.4s}, [%[dst]], #16       \n"
               "bgt             1b                           \n"
               : [dst] "+r" (dst)   //输出参数
               : [src1] "r" (src1), [src2] "r" (src2), [count] "r" (count)
               : "memory", "v0", "v1"
          );
 
}
```

> 纯汇编 V7 V8 对比

// 函数声明头文件
```c
//header
void add_float_neon2(float* dst, float* src1, float* src2, int count);
```

// v7

```asm
    .text                               // .text表示代码正文部分
    .syntax unified
 
    .align 4                            // .align根据不同的汇编器会有不同的行为，像这里的.align4可能表示4字节对齐，也可能表示16字节对齐。
    .global add_float_neon2             // 函数名 可以用.global或.globl来标注全局函数。在Apple的Assembler中仅支持.globl。函数名前要加下划线。
    .type add_float_neon2, %function    // 函数名
    .thumb    // .arm表示后面的函数中的指令都是arm指令。
              // 而.thumb表示后面函数中的指令都是thumb或thumb-2指令。
	      // 其中，如果一个函数是用thumb写的，那么必须用 .thumb_func 修饰，否则连接器在连接符号时会有问题。
.thumb_func
 
add_float_neon2:
.L_loop:
    vld1.32  {q0}, [r1]!                // 函数第一个参数为 r0 第二个为 r1 第三个位r2 第四个为 r3
    vld1.32  {q1}, [r2]!
    vadd.f32 q0, q0, q1
    subs r3, r3, #4
    vst1.32  {q0}, [r0]!
    bgt .L_loop
 
    bx lr


```

// v8

```asm
   .text
 
    .align 4
    .global add_float_neon2            # 函数名
    .type add_float_neon2, %function   # 函数名
 
add_float_neon2:
 
.L_loop:
    ld1  {v0.4s}, [x1], #16      # 函数第一个参数为 x0 第二个为 x1 第三个为 x2 第四个为 x3
    ld1  {v1.4s}, [x2], #16
    fadd v0.4s, v0.4s, v1.4s
    subs x3, x3, #4
    st1  {v0.4s}, [x0], #16
    bgt .L_loop
 
    ret

```




## ARM NEON CNN卷积网络优化 深度学习优化 实例
[参考NCNN](https://github.com/Ewenwan/MVision/blob/master/CNN/HighPerformanceComputing/example/ncnn_%E6%BA%90%E7%A0%81%E5%88%86%E6%9E%90.md)

### 1.绝对值 AbsVal arm_neon_v7 neon_v8 优化
```c
//  arm 内联汇编
// asm(
// 代码列表
// : 输出运算符列表        "r" 表示同用寄存器  "m" 表示内存地址 "I" 立即数 
// : 输入运算符列表        "=r" 修饰符 = 表示只写，无修饰符表示只读，+修饰符表示可读可写，&修饰符表示只作为输出
// : 被更改资源列表
// );
// __asm__　__volatile__(); 

// 关键字“__asm__”，其实也可以写成“asm”。但是“asm”并不是所有版本的GCC编译器都支持的，
// 而且有可能和程序中别的地方定义的变量或函数名冲突，所以用“__asm__”的话，兼容性会好一点。

// __volatile__或volatile 是可选的，假如用了它，则是向GCC 声明不答应对该内联汇编优化，
// 否则当 使用了优化选项(-O)进行编译时，GCC 将会根据自己的判定决定是否将这个内联汇编表达式中的指令优化掉。

// 作用是禁止编译器对后面编写的汇编指令再进行优化。一般情况下，自己写的汇编代码肯定是自己进行设计优化过了的，
// 如果编译器再进行优化的话，很有可能效果还不如不优化，而且也有可能会出现奇怪的错误，所以通常都会带上这个关键字。
// 同样，“__volatile__”也可以写成“volatile”，但可能兼容性会没那么好。

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

// 换行符和制表符的使用可以使得指令列表看起来变得美观。
int AbsVal_arm::forward_inplace(Mat& bottom_top_blob) const
{
    int w = bottom_top_blob.w;// 输入特征图宽度
    int h = bottom_top_blob.h;// 输入特征图高度
    int channels = bottom_top_blob.c;// 输入特征图通道数
    int size = w * h;// 一个通道的元素数量

    #pragma omp parallel for // omp并行
    // #pragma omp parallel for num_threads(opt.num_threads)
    for (int q=0; q<channels; q++)//遍历每一个特征通道
    {
        float* ptr = bottom_top_blob.channel(q);// 当前特征通道数据的起始地址指针

// 如果支持ARM_NEON 则使用NEOB进行优化
#if __ARM_NEON
        int nn = size >> 2;// 128位的寄存器，一次可以操作 4个float32位,剩余不够4个的，最后面直接c语言执行
                           // 右移两位相当于除以4
        int remain = size - (nn << 2);// 4*32 =128字节对其后 剩余的 float32个数, 剩余不够4个的数量
        
#else
        int remain = size; // 若不支持优化，则全部使用不同C语言版本进行计算
#endif // __ARM_NEON
        
/*
从内存中载入:
v7:
   带了前缀v的就是v7 32bit指令的标志；
   ld1表示是顺序读取，还可以取ld2就是跳一个读取，ld3、ld4就是跳3、4个位置读取，这在RGB分解的时候贼方便；
   后缀是f32表示单精度浮点，还可以是s32、s16表示有符号的32、16位整型值。
   这里Q寄存器是用q表示，q5对应d10、d11可以分开单独访问（注：v8就没这么方便了。）
   大括号里面最多只有两个Q寄存器。

     "vld1.f32   {q10}, [%3]!        \n"
     "vld1.s16 {q0, q1}, [%2]!       \n" 


v8:
  ARMV8（64位cpu） NEON寄存器 用 v来表示 v1.8b v2.8h  v3.4s v4.2d
  后缀为8b/16b/4h/8h/2s/4s/2d）
  大括号内最多支持4个V寄存器；

  "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%2], #64 \n"   // 4s表示float32 后面的64 是 64个字节 16个字节为128位 读取4*128位数据
  "ld1    {v0.8h, v1.8h}, [%2], #32     \n"
  "ld1    {v0.4h, v1.4h}, [%2], #32     \n"             // 4h 表示int16 读取32*8 = 2*128位数据 2个v寄存器


所有的汇编代码必须用双引号括起来。如果有多行汇编代码的话，每一条语句都要用双引号括起来，并且在代码后面要加上换行符（“\n”或者“\n\t”）。

这样做是因为GCC会将汇编代码部分作为字符串形式直接传给汇编器，加上换行符后，汇编器就能准确知道哪些字符串表示的是一条汇编语句。同时，为了增加可读性，每条汇编语句都可以换行。

*/
        
// 优化过程
#if __ARM_NEON
// arm_v8================================
#if __aarch64__ // ARMv8-A 是首款64 位架构的ARM 处理器，是移动手机端使用的CPU
        if (nn > 0)// 这里的循环次数已经是 除以4之后的了
        {
        asm volatile(
            "0:                               \n" // 0: 作为标志，局部标签
            "prfm       pldl1keep, [%1, #128] \n" // %1处为ptr标识为1标识,即数据地址，预取 128个字节 4*32 = 128
            "ld1        {v0.4s}, [%1]         \n" // 载入 ptr 指针对应的值，连续4个float 12位
            "fabs       v0.4s, v0.4s          \n" // ptr 指针对应的值 连续4个，使用fabs函数 进行绝对值操作 4s表示浮点数
            "subs       %w0, %w0, #1          \n" // %0 引用 参数 nn 操作次数每次 -1  #1表示1
	                                          // w表示啥?
            "st1        {v0.4s}, [%1], #16    \n" // %1 引用 参数 ptr 指针 向前移动 4*4=16字节 = 16*8 =128位
	                                          // store 1, {v0.4s} 计算绝对值后 再存入 [%1]?
            "bne        0b                    \n" // 如果非0，则向后跳转到 0标志处执行
	    
            // BNE指令会去查看状态寄存器,当Z!=0的时候就跳转到指定位置.
            // BEQ功能与BNE刚好相反,Z==0的时候才跳转到指定位置.

            // 每个操作数的寄存器行为 “=”，表示此操作数类型是只写，即输出寄存器。
	    // "[modifier修改符可选]constraint限定符" (C expression C语言表达式) 
            : "=r"(nn),     // %0 操作次数 nn  循环变量
              "=r"(ptr)     // %1 引用参数 ptr 数据内存地址指针
            
             // 数据 标签标识 nn 标识为0  ptr标识为1
	     // 使用百分号（“%”）后面接一个数字，0表示定义的第一个操作数，1表示定义的第二个操作数，依次类推。
            : "0"(nn),  
              "1"(ptr)
            // 寄存器变化表　list of clobbered registers  
            : "cc", "memory", "v0" // v0寄存器，内存memory， cc CPU状态位 可能会变化
        );
        }
#else
        
// arm_v7===========================
        if (nn > 0)
        {
        asm volatile(
            "0:                             \n" // 0: 作为标志，局部标签
            "vld1.f32   {d0-d1}, [%1]       \n" // %1处为ptr标识为1标识,即数据地址
	                                        // IA 表示在每次传送后递增地址。IA 是缺省值，可以省略。？？
            "vabs.f32   q0, q0              \n" // q0寄存器 = [d1 d0]，128位寄存器，取出四个 float 单精度浮点数 进行绝对值计算 后 写入
            "subs       %0, #1              \n" // %0为 循环变量nn标识，标识循环次数-1  #1表示1
            "vst1.f32   {d0-d1}, [%1]!      \n" // 存储 store1 经过绝对值运算后的寄存器的值 存入原内存中
	                                        // !感叹号作用? 指针 [%1] 前移16字节??
						// ! 指定必须将更新后的基址([%1]递增16)写回到 [%1] 中
						
            "bne        0b                  \n" // 如果非0，则向后跳转到 0标志处执行
            // 每个操作数的寄存器行为 “=”，表示此操作数类型是只写，即输出寄存器。
            : "=r"(nn),     // %0
              "=r"(ptr)     // %1
            // 数据 标签标识 nn 标识为0  ptr标识为1
            : "0"(nn),
              "1"(ptr)
            // 寄存器变化表　list of clobbered registers  
            : "cc", "memory", "q0"// q0寄存器，内存memory， cc CPU状态位 可能会变化 
        );
        }
#endif // __aarch64__
#endif // __ARM_NEON
        
        // 剩余不够4个的直接c语言执行=====
        for (; remain>0; remain--)// 循环次数-1
        {
            *ptr = *ptr > 0 ? *ptr : - *ptr;
            ptr++;// 指针+1
        }
    }

    return 0;
}

```

### 2. BN层 通道数据归一化 BatchNorm

```c
// load_model() 函数预处理===============

    // 去均值 归一化 合在一起=============
    // 各个通道均值 mean_data = sum(xi)/m
    // 各个通道方差 var_data     = sum((xi - mean_data)^2)/m
    // xi‘ = ( xi - mean_data )/(sqrt(var_data + eps))  // 去均值，除以方差，归一化
    
    // yi = slope_data * xi'  + bias_data  //  缩放 + 平移=====
    
    // 写成一起=====================
    // yi = slope_data / (sqrt(var_data + eps)) * xi  + bias_data - slope_data*mean_data/(sqrt(var_data + eps)) 
    // b = slope_data / (sqrt(var_data + eps)) = slope_data /sqrt_var;
    // a = bias_data - slope_data*mean_data/(sqrt(var_data + eps)) = bias_data - slope_data*mean_data/sqrt_var;
    
    // yi = b * xi + a
    
// 在layer/batchnorm.cpp  的 BatchNorm::load_model 函数中处理

int BatchNorm::load_model(const ModelBin& mb)

{
    slope_data = mb.load(channels, 1);  // 缩放系数
    if (slope_data.empty())
        return -100;
	
    mean_data = mb.load(channels, 1);   // 均值
    if (mean_data.empty())
        return -100;

    var_data = mb.load(channels, 1);    // 方差
    if (var_data.empty())
        return -100;
	
    bias_data = mb.load(channels, 1);   // 标准差
    if (bias_data.empty())
        return -100;
	
    a_data.create(channels);            // 去均值减方差 缩放和平移合在一起 >>> 新偏移量
    if (a_data.empty()) 
        return -100;

    b_data.create(channels);            // 新 缩放系数
    if (b_data.empty())
        return -100;

    for (int i=0; i<channels; i++)
    {
        float sqrt_var = sqrt(var_data[i] + eps);                           // 标准差
        a_data[i] = bias_data[i] - slope_data[i] * mean_data[i] / sqrt_var; // 新偏移量
        b_data[i] = slope_data[i] / sqrt_var;                               // 新 缩放系数
    }

    return 0;
} 
    
    
int BatchNorm_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;
    
    if (dims != 3) // 只有三通道的特征图才使用 neon加速
        return BatchNorm::forward_inplace(bottom_top_blob, opt);

    // a = bias - slope * mean / sqrt(var)
    // b = slope / sqrt(var)
    // value = b * value + a

    int w = bottom_top_blob.w;// 特征图宽度
    int h = bottom_top_blob.h;// 特征图高度
    int size = w * h;// 一张特征图尺寸

// 整合后的变化系数  yi = b * xi + a
    const float* a_data_ptr = a_data; // batchnorm.h 中公开的 Mat矩阵数据，数组首地址
    const float* b_data_ptr = b_data;
    
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q=0; q<channels; q++)// 遍历每个通道
    {
        float* ptr = bottom_top_blob.channel(q);// 每一个通道的 特征图 数据 首地址
 // 每通道 的 变化系数==都一样=
        float a = a_data_ptr[q];
        float b = b_data_ptr[q];

#if __ARM_NEON
        int nn = size >> 2; // 128位寄存器一个可以操作 4个 32位浮点数，所以总数除以4得到 寄存器操作次数
	                    // 右移动2位，相当于除以4，例如 10，右移两位相当于乘除4，得到2
        int remain = size - (nn << 2);// 10-2*4=2 剩余2个 不够4，使用普通c语言版本
#else
        int remain = size; // 如果不支持neon，则全部使用 普通c语言计算呢
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
        if (nn > 0)
        {
        asm volatile(
            "dup        v1.4s, %w4             \n" // 每通道的 变化系数a,b都一样只需载入一次，传入的为立即数使用dup
            "dup        v2.4s, %w5             \n" // v1存储a，v2存储b，v0存储特征数据，v3存储变化的数据地址以及a
            "0:                                \n" // 构成循环的标记号
            "prfm       pldl1keep, [%1, #128]  \n" // 从%1 ptr 处预读取 128字节 4*32 4个浮点数
            "ld1        {v0.4s}, [%1]          \n" // 载入 ptr 指针对应的值到 v0，连续4个float
            "orr        v3.16b, v1.16b, v1.16b \n" // v1 --> v3,  v3 =a
            "fmla       v3.4s, v0.4s, v2.4s    \n" // 特征数据v0*缩放v2 + 偏置v3 最后赋值给 v3 += v0×b
            "subs       %w0, %w0, #1           \n" // %0 为nn 执行次数 -1   #1   为1
            "st1        {v3.4s}, [%1], #16     \n" // 结果v3 store存储到 原数据地址处，原数据地址递增16字节
            "bne        0b                     \n" // subs结果不为零的话跳转回去，继续循环
            : "=r"(nn),     // %0
              "=r"(ptr)     // %1
            : "0"(nn),      // 2 ???=====
              "1"(ptr),     // 3 ???=====
              "r"(a),       // %4 存入寄存器 只读, 不变, 参数 偏置a
              "r"(b)        // %5 存入寄存器 只读, 不变，参数 缩放归一化系数
            : "cc", "memory", "v0", "v1", "v2", "v3"
	    //  cc CPU状态位，内存memory，v，v1，v2，v3寄存器 可能会变化
        );
        }
#else
        if (nn > 0)
        {
        asm volatile(
            "vdup.f32   q1, %4              \n"// 每通道的 变化系数a,b都一样只需载入一次，传入的为立即数使用dup
            "vdup.f32   q2, %5              \n"// q1存储变量 a,q2存储变量b，q0存储特征值
	                                       // q3存储中间变量，先存储a和b以及q0执行乘加后，存储最终的结果
					       // 最后把 在q3中的结果 存储回原 特征数据地址处
					       
            "0:                             \n"// 构成循环的标记号
            "pld        [%1, #128]          \n"// 从%1 ptr 处预读取 128字节 4*32 4个浮点数
            "vld1.f32   {d0-d1}, [%1 :128]  \n"// 从%1 ptr 处载入 4个浮点数到q0，传入的为指针，使用ld
            "vorr.32    q3, q1, q1          \n"// q3 = q1 或 q1 = 变量a
            "vmla.f32   q3, q0, q2          \n"// q3 += q0(特征值)*q2(变量b), 乘加运算
            "subs       %0, #1              \n"// 循环次数 nn -1
            "vst1.f32   {d6-d7}, [%1 :128]! \n"// q3->{d6-d7} 结果值 顺序store到 原特征值地址处[%1]
	                                       // !感叹号，强制[%1]向后跳转128位 
            "bne        0b                  \n"// 不为零跳回去，继续循环 
	    
            : "=r"(nn),     // %0 循环次数(按寄存器一次并行运算4个浮点数数) nn 
              "=r"(ptr)     // %1 特征值数据地址
            : "0"(nn),      // 2 ???===
              "1"(ptr),     // 3 ???===
              "r"(a),       // %4
              "r"(b)        // %5
            : "cc", "memory", "q0", "q1", "q2", "q3"
	    //  cc CPU状态位，内存memory，q0，q1，q2，q3寄存器 可能会变化
        );
        }
#endif // __aarch64__
#endif // __ARM_NEON
        for (; remain>0; remain--)
        {
            *ptr = b * *ptr + a;// 剩余不够 4个的 直接c语言执行

            ptr++;// 数据地址增加 1
        }
    }
    return 0;
}    
    

```

### 3.添加偏置类 bias

```c
// 进行运算： y = x + bias ---> x

int Bias_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;// 特征图宽度
    int h = bottom_top_blob.h;// 特征图高度
    int channels = bottom_top_blob.c;// 通道数量（特征 图 厚度，汉堡包层数）
    int size = w * h;// 单通道特征尺寸

    const float* bias_ptr = bias_data; // 偏置数据 指针 在 bias.h 中定义的 public公开数据
    
    #pragma omp parallel for num_threads(opt.num_threads)// omp并行执行
    
    for (int q=0; q<channels; q++)// 遍历每个通道
    {
        float* ptr = bottom_top_blob.channel(q);// 每个通道数据起始指针 (原有特征数据)

        float bias = bias_ptr[q];// 每通道偏置参数一样

#if __ARM_NEON
        int nn = size >> 2; // 128位寄存器一个可以操作 4个 32位浮点数，所以总数除以4得到 寄存器操作次数
	                    // 右移动2位，相当于除以4，例如 10，右移两位相当于乘除4，得到2
        int remain = size - (nn << 2);// 剩余不够4个的数量 1～3
#else
        int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
/*
// 这里 直接使用了 neon Instrinsic 内在函数，不过优化程度不如 汇编代码

        float32x4_t _bias = vdupq_n_f32(bias);// 偏置数据 dup载入到 寄存器 4个32位的浮点数
	                                      // 传入的为 立即数
        for (; nn>0; nn--)
        {
            float32x4_t _p = vld1q_f32(ptr);// 载入 特征值 传入的为 数据的地址 
            float32x4_t _outp = vaddq_f32(_p, _bias);// 加上偏置_bias
            vst1q_f32(ptr, _outp);                   // 从寄存器数据 设置内存数据 store1存储结果数据到ptr

            ptr += 4;// 特征指针 移动四个单位
        }
*/	
// 可以试写 neon内联汇编代码，区分v8 、v7===============
	
#if __aarch64__
        if (nn > 0)
        {
        asm volatile(
            "dup        v1.4s, %w4             \n" // 每通道的 变化系数a,b都一样只需载入一次，传入的为立即数使用dup
            "0:                                \n" // 构成循环的标记号
            "prfm       pldl1keep, [%1, #128]  \n" // 从%1 ptr 处预读取 128字节 4*32 4个浮点数
            "ld1        {v0.4s}, [%1]          \n" // 载入 ptr 指针对应的值到 v0，连续4个float
            "fadd       v0.4s, v0.4s, v1.4s    \n" // v0 = v0 + v1
            "subs       %w0, %w0, #1           \n" // %0 为nn 执行次数 -1   #1   为1
            "st1        {v0.4s}, [%1], #16     \n" // 结果v0 store存储到 原数据地址处，原数据地址递增16字节
            "bne        0b                     \n" // subs结果不为零的话跳转回去，继续循环
            : "=r"(nn),     // %0
              "=r"(ptr)     // %1
            : "0"(nn),      // 2 ???=====
              "1"(ptr),     // 3 ???=====
              "r"(bias)     // %4 存入寄存器 只读, 不变, 参数 偏置a
            : "cc", "memory", "v0", "v1"
	    //  cc CPU状态位，内存memory，v，v1，v2，v3寄存器 可能会变化
        );
        }
#else
        if (nn > 0)
        {
        asm volatile(
            "vdup.f32   q1, %4              \n"// 每通道的 变化系数a,b都一样只需载入一次，传入的为立即数使用dup		       
            "0:                             \n"// 构成循环的标记号
            "pld        [%1, #128]          \n"// 从%1 ptr 处预读取 128字节 4*32 4个浮点数
            "vld1.f32   {d0-d1}, [%1 :128]  \n"// 从%1 ptr 处载入 4个浮点数到q0，传入的为指针，使用ld
            "vadd.f32   q0, q0, q1          \n"// q0 = q0(特征值) + q1(变量bias)
            "subs       %0, #1              \n"// 循环次数 nn -1
            "vst1.f32   {d0-d1}, [%1 :128]! \n"// q0->{d0-d1} 结果值 顺序store到 原特征值地址处[%1]  !感叹号，强制[%1]向后跳转128位 
            "bne        0b                  \n"// 不为零跳回去，继续循环 
	    
            : "=r"(nn),     // %0 循环次数(按寄存器一次并行运算4个浮点数数) nn 
              "=r"(ptr)     // %1 特征值数据地址
            : "0"(nn),      // 2 ???===
              "1"(ptr),     // 3 ???===
              "r"(bias)     // %4
            : "cc", "memory", "q0", "q1"
	    //  cc CPU状态位，内存memory，q0，q1，q2，q3寄存器 可能会变化
        );
        }
#endif // __aarch64__	
	
	
#endif // __ARM_NEON

        for (; remain>0; remain--)
        {
            *ptr = *ptr + bias; // 普通c 版本 加上偏置

            ptr++;
        }
    }

    return 0;
}


```

### 4.修剪 clip 上下阈值处理


```c
int Clip_arm::forward_inplace(Mat &bottom_top_blob, const Option &opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;
    int elempack = bottom_top_blob.elempack;

#if __ARM_NEON
    if (elempack == 4)  // 如果数据数量是4的整数倍 直接使用instric指令计算 也不用考虑剩余数的处理
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            float32x4_t _max = vdupq_n_f32(max); // 最小值
            float32x4_t _min = vdupq_n_f32(min); // 最大值

            for (int i=0; i<size; i++)
            {
                float32x4_t _ptr = vld1q_f32(ptr);// 载入特征值 x
                _ptr = vmaxq_f32(_ptr, _min);     // 下限处理
                _ptr = vminq_f32(_ptr, _max);     // 上限处理
                vst1q_f32(ptr, _ptr);             // 结果存回 内存地址

                ptr += 4;
            }
        }

        return 0;
    }
#endif // __ARM_NEON

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q=0; q<channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

#if __ARM_NEON
        int nn = size >> 2;                  // 除以4 的余数
        int remain = size & 3;               // 剩余
#else
        int remain = size;
#endif

#if __ARM_NEON
        float32x4_t _max = vdupq_n_f32(max); // 最小值
        float32x4_t _min = vdupq_n_f32(min); // 最大值
#if __aarch64__
        for (; nn>0; nn--)
        {
            float32x4_t _ptr = vld1q_f32(ptr);
            _ptr = vmaxq_f32(_ptr, _min);
            _ptr = vminq_f32(_ptr, _max);
            vst1q_f32(ptr, _ptr);
            ptr += 4;
        }
#else
        if (nn > 0)
        {
        asm volatile(
            "0:                             \n"
            "pld        [%1, #128]          \n" // 预取 128位(字节?)
            "vld1.f32   {d0-d1}, [%1: 128]  \n" // q0 寄存器存储 普通人指针处 的值
            "vmax.f32   q0, q0, %q4         \n" // 下限处理
            "vmin.f32   q0, q0, %q5         \n" // 上限处理
            "subs       %0, #1              \n" 
            "vst1.f32   {d0-d1}, [%1: 128]! \n"
            "bne        0b                  \n"

            : "=r"(nn),     // %0
              "=r"(ptr)     // %1
            : "0"(nn),
              "1"(ptr),
              "w"(_min),    // %q4
              "w"(_max)     // %q5
            : "cc", "memory", "q0"
        );
        }
#endif // __aarch64__
#endif // __ARM_NEON

        for (; remain>0; remain--)
        {
            if (*ptr < min)
                *ptr = min;
            if (*ptr > max)
                *ptr = max;
            ptr++;
        }
    }

    return 0;
}

```







