# ARM_NEON_CNN_纯汇编编程_MNN 

[参考1](https://blog.csdn.net/jxt1234and2010/article/details/104012746)

纯汇编开发，优化策略相应的简单很多，基本上**循环展开、指令重排**之后，就有立竿见影的效果。

> 基本流程

1.梳理代码，设计实现方案，提炼出核心的运算部分，先用C实现一遍，保证正确;

2.32位汇编代码初步实现，保证功能正确;

3.汇编代码优化：这一步优化只做循环展开和指令重排，如果有更好的计算方案，先退回 C 重新写;

4.64位的也支持一下：替换一下寄存器名和指令名（可以写脚本完成），然后微调一下函数前后读参数、入出栈与返回的地方（可选）64位的进一步优化一下，毕竟寄存器多了一倍;

## Procedure Call Standard【函数调用标准】

### ARM 32(v7a)

> 通用寄存器 32bit  r0 r1 ... r15     

传参数用: r0 r1 r2 r3  用完要恢复(进栈保存后出栈): **r4 r5 ... r11**

随便使用: r0 r1 r2 r3, r12  不能使用(谨慎使用): r13 r14  r15 

> 向量寄存器128bit  q0 q1 ... q15  可以64bit形式使用 即 d0 d1 d2 d3 ... d30 d31

用完要恢复(进栈保存后出栈): **q4 q5 q6 q7**

随便使用: q0 q1 q2 q3, q8 q9 ... q15

用完恢复是指相应的寄存器在函数返回前必须恢复进入时的值，比如我们要在代码中用 q4，就必须在函数前写一句
```c 
vpush {q4}     // 进栈保存 保护 向量寄存器使用 vpush {} 
push {r4, lr}  // 进栈保存 保护 通用寄存器使用 push {}
```
函数返回前写一句：
```c
vpop {q4}      // 出栈恢复
pop {r4, pc}   // 出栈恢复      通用寄存器使用 pop {}
```

     r12用作子程序间scratch寄存器,记作ip; 在子程序的连接代码段中经常会有这种使用规则.
     r13用作数据栈指针,记做SP,在子程序中寄存器R13不能用做其他用途. 寄存器SP在进入子程序时的值和退出子程序时的值必须相等.
     r14用作连接寄存器,记作lr ; 它用于保存子程序的返回地址,如果在子程序中保存了返回地址,则R14可用作其它的用途.
     r15是程序计数器,记作PC ; 它不能用作其他用途.
     
### ARM 64(v8)


> 通用寄存器 64bit  x0 x1 ... x31 可以32bit形式使用  w0 w1 ... w31 使用低32位

传参数用: x0-x7  用完要恢复(进栈保存后出栈): x19-x28

随便使用: x0-x15  不能使用(谨慎使用): x16 x17 x18, x29 x30 x31

> 向量寄存器128bit  v0 v1 ... v31 

可以另外四种形式使用： **64位:d   32位:s  16位:h   8位:b**

传参数用: 浮点数据传到 v0 v1

用完要恢复(进栈保存后出栈): **v8-v15**

随便使用:  v0-v7  v16-v31

值得注意的是，arm64 的传参为浮点时，会传到 v0.s[0], v0.s[1] …… 而非通用寄存器，这个很坑，建议不要用浮点传参

## 汇编优化实例
> c版本Relu代码

```c
void ReluForward(float* dst, const float* src, size_t sizeDiv4)
{
      for (int i=0; i<4*sizeDiv4; ++i)      // 确保数据长度4对齐
      {
           dst[i] = src[i] >0 ? src[i] : 0; // 小于0截断为0
      }
}
```

> c的NEON版本代码
```c
void ReluCNeon(float* dst, const float* src, size_t sizeDiv4)
{
    float32x4_t limit = vdupq_n_f32(0.0f);   // 4个32位浮点数据 装载到寄存器里面
    for (int i=0; i<sizeDiv4; ++i)           // sizeDiv4
    {
        float32x4_t value = vld1q_f32(src);  // 装在4个32位源数据
        value = vmaxq_f32(value, limit);     // 4个数据 下截断操作
        vst1q_f32(dst, value);               // 更新的值存入目标地址

        dst+=4; // 源数据地址和目标数据地址 +4
        src+=4;
    }
}
```

> 基础汇编

由于ios和android上面函数编译的符号不一致，这里引入一个头文件，定义一个函数声明宏，去屏蔽这种差异：

ArmAsmGlobal.h
```c
.macro asm_function fname
#ifdef __APPLE__
.globl _\fname
_\fname:
#else
.global \fname
\fname:
#endif

```

> 汇编：ReluBasic
```asm
//汇编：ReluBasic
#include "ArmAsmGlobal.h"
asm_function ReluBasic     //指定 汇编函数的 函数名

//函数参数规定
//按照 arm32 的 函数调用标准，以下变量由调用方传至寄存器
//r0: dst, r1: src, r2: sizeDiv4

push {lr}
vmov.i32 q15, #0  // 限制值 limit值 0 存入 q15寄存器  4个32位 0.0数据

cmp r2, #0  // 剩余数据量大小 sizeDiv4 等于0的话就结束循环
beq End     //跳转：beq 表示 r2 等于0时跳转

Loop:       //标志，供跳转用
vld1.32 {q0}, [r1]!     // 读取源数据 4个32位 数据到q0寄存器 从地址r1处  !表示 数据取过后 r1 += 4
vmax.f32 q0, q0, q15    // q0 = max(q0,0)
vst1.32 {q0}, [r0]!     // 更新的值 存入 目的地址r0  !表示 数据存过后 r0 += 4
subs r2, r2, #1         // 这一句 相当于 sub r2, r2, #1  &&  cmp r2, #0
bne Loop                // 跳转：bne 表示 r2 不等于0时跳转

End:
pop {pc}

```

#### 汇编优化  指令流水 循环展开

我们注意到循环主体，语句前后有较强依赖关系

```asm
vld1.32 {q0}, [r1]!
vmax.f32 q0, q0, q15  //q0 依赖于 前一行的读  从内存载入数据到 寄存器q0
vst1.32 {q0}, [r0]!   //q0 依赖于前一行的计算 q0 = max(q0,0)
```

ARM 的CPU一般都有双通道发射能力（跟多核多线程不是同一个概念），在执行如下类型的语句时，可以并发执行，提升效率：
```asm
vld1.32 {q0}, [r1]!
vmax.f32 q1, q1, q15 //不使用 q0，无依赖关系  可以和上面的指令 并发执行
```

为了让我们的汇编代码解除语句前后的依赖关系，先进行一次循环展开：


> 汇编：ReluUnroll   就是每次循环 多操作一些数据  每次干的活 量增大  提高CPU利用率 

```asm

//汇编：ReluUnroll
#include "ArmAsmGlobal.h"
asm_function ReluUnroll   //指定 汇编函数的 函数名

//函数参数规定
//按照 arm32 的 函数调用标准，以下变量由调用方传至寄存器
//r0: dst, r1: src, r2: sizeDiv4

vmov.i32 q15, #0  // 限制值 limit值 0 存入 q15寄存器  4个32位 0.0数据

push {lr}

L4:
cmp r2, #3 // 进入的时候是 除4这里相当于 3*4 = 12
ble L1     // 数据量 <= 12 就不够一次处理16个了


// 一次处理16个数据
L4Loop:
// 载入源数据
vld1.32 {q0, q1}, [r1]!  //   载入8个32位浮点
vld1.32 {q2, q3}, [r1]!  // 再载入8个32位浮点

// 计算16个数据 x = max(x,0)
vmax.f32 q0, q0, q15          //一次处理多点数据 利用cpu并发 提高cpu利用率
vmax.f32 q1, q1, q15
vmax.f32 q2, q2, q15
vmax.f32 q3, q3, q15

// 更新后的数据存入 目标地址
vst1.32 {q0, q1}, [r0]!
vst1.32 {q2, q3}, [r0]!

// 循环条件检测
sub r2, r2, #4  //剩余数据量 - 4*4
cmp r2, #4
bge L4Loop   // 剩余数据/4 >= 4, 会再循环， 进入的时候是 除4，这个相当于 4*4=16


// 处理剩余 的 4/8/12个数据
L1:
cmp r2, #0
beq End

L1Loop:
// 载入4个数据
vld1.32 {q0}, [r1]!
// 处理4个数据
vmax.f32 q0, q0, q15
// 保存4个数据
vst1.32 {q0}, [r0]!
// 循环条件检查
subs r2, r2, #1 // 剩余数据量 - 1*4
bne L1Loop


End:
pop {pc}  // 程序指针寄存器

// 其他剩余的1~3个数据可以再外部使用C语言单独处理
```


> 展开之后，L4Loop 内部的语句已经大部分解除了依赖，但还不完全，为了完全解除，我们需要用个小技巧【汇编重点技巧】：

这个技巧就是将循环主体代码拆成两半，原先的 Loop[AB] 就变成了 A->Loop[BA]->B，然后 BA 由于顺序颠倒，可以实现错排并发。

**汇编：ReluUnrollReorder**

```asm
//汇编：ReluUnrollReorder
#include "ArmAsmGlobal.h"
asm_function ReluUnrollReorder  

push {lr}
vmov.i32 q15, #0  // 限制值 limit值 0 存入 q15寄存器  4个32位 0.0数据

L4:
cmp r2, #3        // 数据量 <= 12   转 L1  处理剩余数据
ble L1

vld1.32 {q0, q1}, [r1]!    // 载入8个
vmax.f32 q0, q0, q15       // 处理4个
vld1.32 {q2, q3}, [r1]!    // 载入8个
vmax.f32 q1, q1, q15       // 处理最前面载入的4个

sub r2, r2, #4    // 数据量 -16
cmp r2, #3
ble L4End         // 数据量 = 16 转 L4End   处理最后16个数据

L4Loop:

vst1.32 {q0, q1}, [r0]!    // 存储 上面L4 中处理好的 8个数据
vmax.f32 q2, q2, q15       // 处理4个
vld1.32 {q0, q1}, [r1]!    // 载入8个
vmax.f32 q3, q3, q15       // 处理4个
vst1.32 {q2, q3}, [r0]!    // 存储8个
vmax.f32 q0, q0, q15       // 处理4个 
vld1.32 {q2, q3}, [r1]!    // 载入8个
vmax.f32 q1, q1, q15       // 处理4个 

// 循环条件检测
sub r2, r2, #4
cmp r2, #4
bge L4Loop

L4End:
vst1.32 {q0, q1}, [r0]!  // 存储8个  上面L4Loop 有两次处理4个 还未存储到内存
vmax.f32 q2, q2, q15     // 处理4个 
vmax.f32 q3, q3, q15     // 处理4个 
vst1.32 {q2, q3}, [r0]!  // 存储8个

// 处理数据量 <= 12  0/4/8/12
L1:
cmp r2, #0
beq End

L1Loop:
vld1.32 {q0}, [r1]!   // 一次处理4个数据
vmax.f32 q0, q0, q15
vst1.32 {q0}, [r0]!
subs r2, r2, #1
bne L1Loop


End:
pop {pc}

```


性能对比

    魅蓝 mental 上测试
    sizeDiv4 = 100000，连续跑10000次（由于 relu 是一个十分简单的op，跑大批量的才能看到效果）
    C-neon Cost time : 4856.960449 ms
    汇编ReluBasic Cost time : 4716.672363 ms
    汇编ReluUnroll Cost time : 2814.848145 ms
    汇编ReluUnrollReorder Cost time : 2359.424072 ms

可以看到：

    1、最简单的汇编和用 neon api 的 C差不大多
    2、同样是汇编，ReluUnrollReorder较ReluBasic足足提升了100%

