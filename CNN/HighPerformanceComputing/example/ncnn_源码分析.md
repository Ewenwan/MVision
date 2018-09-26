# ncnn 框架分析
[本文github链接](https://github.com/Ewenwan/MVision/blob/master/CNN/HighPerformanceComputing/example/ncnn_%E6%BA%90%E7%A0%81%E5%88%86%E6%9E%90.md)

[在ncnn中建立新层](https://github.com/Ewenwan/MVision/blob/master/CNN/HighPerformanceComputing/example/ncnn_%E6%96%B0%E5%BB%BA%E5%B1%82.md)

[参考1](http://hongbomin.com/2017/09/02/ncnn-analysis/)

[参考2](https://github.com/Tencent/ncnn/wiki)

# 1. param 和 bin 文件分析
## param
      7767517   # 文件头 魔数
      75 83     # 层数量  输入输出blob数量
                # 下面有75行
      Input            data             0 1 data 0=227 1=227 2=3
      Convolution      conv1            1 1 data conv1 0=64 1=3 2=1 3=2 4=0 5=1 6=1728
      ReLU             relu_conv1       1 1 conv1 conv1_relu_conv1 0=0.000000
      Pooling          pool1            1 1 conv1_relu_conv1 pool1 0=0 1=3 2=2 3=0 4=0
      Convolution      fire2/squeeze1x1 1 1 pool1 fire2/squeeze1x1 0=16 1=1 2=1 3=1 4=0 5=1 6=1024
      ...
      层类型            层名字   输入blob数量 输出blob数量  输入blob名字 输出blob名字   参数字典
      
      参数字典，每一层的意义不一样：
      数据输入层 Input            data             0 1 data 0=227 1=227 2=3   图像宽度×图像高度×通道数量
      卷积层    Convolution  ...   0=64     1=3      2=1    3=2     4=0    5=1    6=1728           
               0输出通道数 num_output() ; 1卷积核尺寸 kernel_size();  2空洞卷积参数 dilation(); 3卷积步长 stride(); 
               4卷积填充pad_size();       5卷积偏置有无bias_term();   6卷积核参数数量 weight_blob.data_size()；
                                                                  C_OUT * C_in * W_h * W_w = 64*3*3*3 = 1728
      池化层    Pooling      0=0       1=3       2=2        3=0       4=0
                          0池化方式:最大值、均值、随机     1池化核大小 kernel_size();     2池化核步长 stride(); 
                          3池化核填充 pad();   4是否为全局池化 global_pooling();
      激活层    ReLU       0=0.000000     下限阈值 negative_slope();
               ReLU6      0=0.000000     1=6.000000 上下限
      
      综合示例：
      0=1 1=2.5 -23303=2,2.0,3.0
      
      数组关键字 : -23300 减去 0 ~ 19  例如 -23303 表示3个元素的数组
[各层详细表格](https://github.com/Tencent/ncnn/wiki/operation-param-weight-table)
      
## bin

        +---------+---------+---------+---------+---------+---------+
        | weight1 | weight2 | weight3 | weight4 | ....... | weightN |
        +---------+---------+---------+---------+---------+---------+
        ^         ^         ^         ^
        0x0      0x80      0x140     0x1C0

      所有权重数据连接起来, 每个权重占 32bit

      权重数据 weight buffer

      [flag] (optional 可选)
      [raw data]
      [padding] (optional 可选)

          flag : unsigned int, little-endian, indicating the weight storage type, 
                 0 => float32, 
                 0x01306B47 => float16, 
                 otherwise => quantized int8, 
                      may be omitted if the layer implementation forced the storage type explicitly。
          raw data : raw weight data, little-endian, 
                     float32 data or float16 data or quantized table 
                     and indexes depending on the storage type flag。
          padding : padding space for 32bit alignment, may be omitted if already aligned。



## 2. 轻模式
      开启轻模式省内存 set_light_mode(true)
      
      每个layer都会产生blob，除了最后的结果和多分支中间结果，大部分blob都不值得保留，
      开启轻模式可以在运算后自动回收，省下内存。
      
      举个例子：某网络结构为 A -> B -> C，在轻模式下，向ncnn索要C结果时，A结果会在运算B时自动回收，
      而B结果会在运算C时自动回收，最后只保留C结果，后面再需要C结果会直接获得，满足绝大部分深度网络的使用方式。
      
## 3. 网络和运算是分开的
      
      ncnn的net是网络模型，实际使用的是extractor，
      也就是同个net可以有很多个运算实例，而且运算实例互不影响，中间结果保留在extractor内部，
      在多线程使用时共用网络的结构和参数数据，初始化网络模型和参数只需要一遍.
      
      举个例子：全局静态的net实例，初始化一次后，就能不停地生成extractor使用.

## 4. openmp多核加速
```c
// 使用OpenMP加速只需要在串行代码中添加编译指令以及少量API即可。

// 如下是一个向量相加的函数（串行）：

void add(const int* a, const int* b, int* c, const int len)
{
  for(int i=0; i<len; i++)
  {
    c[i] = a[i] + b[i];
  }
}

改成OpenMP多核加速（并行）：

#pragma omp parallel for 
// #pragma omp parallel for  num_threads(opt.num_threads);
void add(const int* a, const int* b, int* c, const int len)
{
  for(int i=0; i<len; i++)
  {
    c[i] = a[i] + b[i];
  }
}

// 理想情况下，加速比大约能达到0.75*cores。
```
## 5. 缓存，仿存，cache
缓存对于高速计算是非常重要的一环，通过合理的安排内存读写，能非常有效的加速计算。   
数据存储一般是行优先存储，而cpu仿存，是会一次多读取当前内存后的连续几个地址中的数据。  

如下面代码的gemm计算:   
![](http://hongbomin.com/2017/09/02/ncnn-analysis/gemm.png)

版本1：  
```c
static void gemm_v1(float* matA, float* matB, float* matC, const int M, const int N, const int K, const int strideA, const int strideB, const int strideC)
{
	for (int i = 0; i < N; i++)// 外层循环 为 列大小，会造成过多的仿存浪费!!!!
	{
		for (int j = 0; j < M; j++)
		{
			float sum = 0.0f;
			for (int k = 0; k < K; k++)
			{
				sum += matA[j*strideA + k] * matB[k*strideB + i];// 矩阵B 是列 访问
			}
			matC[j*strideC + i] = sum;// 矩阵C也是列 访问(j先变换，列方向)
		}
	}
}
```

版本2：  
```c
static void gemm_v2(float* matA, float* matB, float* matC, const int M, const int N, const int K, const int strideA, const int strideB, const int strideC)
{
	for (int i = 0; i < M; i++)// 外层循环 为 行主导，仿存效果好
	{
		for (int j = 0; j < N; j++)
		{
			float sum = 0.0f;
			for (int k = 0; k < K; k++)
			{
				sum += matA[i*strideA + k] * matB[k*strideB + j];// 只有矩阵B是列访问顺序
			}
			matC[i*strideC + j] = sum;
		}
	}
}
```

	gemm_v1比gemm_v2速度会慢很多，尤其是数据量比较大的时候。
	因为在gemm_v1中，matB和matC的访存以列为方向，会出现很多cache不命中的情况。
	而在gemm_v2中则只有matB发生较多cache不命中，而这是gemm计算无法避免的。

	在ncnn中，没有将输入和输出展开成两个大矩阵，会比较耗费内存；
	就是按照卷积的方式，每次只读取，运算的部分内容。
	以卷积计算conv3x3_s1为例，
	每次从matA同时访问4行（一般一次3x3卷积只需要访问3行），
	由于step是1，所以可以同时生成2行的convolution结果。
	可以看到有2行数据直接共用了，缓存利用率得到极大提高。
	
![](http://hongbomin.com/2017/09/02/ncnn-analysis/gemm_row0_col0.png)

![](http://hongbomin.com/2017/09/02/ncnn-analysis/gemm_row1_col0.png)

[快速矩阵乘法 参考](https://github.com/Ewenwan/MVision/blob/master/CNN/HighPerformanceComputing/%E5%BF%AB%E9%80%9F%E7%9F%A9%E9%98%B5%E4%B9%98%E6%B3%95.md)

## 6. 单指令多数据指令SIMD 优化
	SIMD即单指令多数据指令，
	目前在x86平台下有MMX/SSE/AVX系列指令，
	arm平台下有NEON指令。
	一般SIMD指令通过intrinsics或者汇编实现。
[SSE参考]()

[neno参考](http://hongbomin.com/2016/05/13/arm_neon_introduction/)
	

### ARM 下 NENO编程
	ARM CPU最开始只有普通的寄存器，可以进行基本数据类型的基本运算。
	自ARMv5开始引入了VFP（Vector Floating Point）指令，该指令用于向量化加速浮点运算。
	自ARMv7开始正式引入NEON指令，NEON性能远超VFP，因此VFP指令被废弃。
```c

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


// 4. neon优化的vector相乘
static void neon_vector_mul((const std::vector<float>& vec_a, 
                              const std::vector<float>& vec_b, 
			      std::vector<float>& vec_result)
{
	// 检查 数组维度
	assert(vec_a.size() == vec_b.size());
	assert(vec_a.size() == vec_result.size());
	
	// noon 寄存器操作
	int i = 0;
	const auto data_a = vld1q_f32(&vec_a[i]);// 放入寄存器
	const auto data_b = vld1q_f32(&vec_b[i]);// 放入寄存器
	
	float* dst_ptr = &vec_result[i]; // 结果矩阵 指针
	
	const auto data_res = vmulq_f32(data_a, data_b); // 32为寄存器 浮点数乘法
	
	vst1q_32(dst_ptr, data_res);// 将 寄存器乘法结果 复制到 结果数组中
}

// 这段代码中使用了3条NEON指令：vld1q_f32，vmulq_f32，vst1q_f32


```
	


## 7. src目录分析
    /src目录：
    目录顶层下是一些基础代码，如宏定义，平台检测，mat数据结构，layer定义，blob定义，net定义等。
    ./src/layer下是所有的layer定义代码
    ./src/layer/arm是arm下的计算加速的layer
    ./src/layer/x86是x86下的计算加速的layer。

