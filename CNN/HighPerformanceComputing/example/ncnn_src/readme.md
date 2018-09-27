# ncnn　源文件 注释解读

    /src目录：
        目录顶层下是一些基础代码，如宏定义，平台检测，mat数据结构，layer定义，blob定义，net定义等。
          platform.h.in 平台检测
      paramdict.cpp paramdict.h 层参数解析 读取二进制格式、字符串格式、密文格式的参数文件
      opencv.cpp opencv.h       opencv 风格的数据结构 的 最小实现
                                大小结构体 Size 
              矩阵框结构体 Rect_ 交集 并集运算符重载
              点结构体     Point_
              矩阵结构体   Mat     深拷贝 浅拷贝 获取指定矩形框中的roi 读取图像 写图像 双线性插值算法改变大小
            mat.cpp mat.h             3维矩阵数据结构, 在层间传播的就是Mat数据，Blob数据是花架子
                                substract_mean_normalize(); 去均值并归一化
              half2float();               float16 的 data 转换成 float32 的 data
              copy_make_border();         矩阵周围填充
              resize_bilinear_image();    双线性插值
      modelbin.cpp modelbin.h   从文件中载入模型权重、从内存中载入、从数组中载入
      layer.cpp layer.h         层接口，四种前向传播接口函数
      Blob数据是花架子

      net.cpp net.h             ncnn框架接口：
                                注册 用户定义的新层 Net::register_custom_layer();
                                网络载入 模型参数   Net::load_param();
              载入     模型权重   Net::load_model();
              网络blob 输入       Net::input();
              网络前向传          Net::forward_layer();    被Extractor::extract() 执行
              创建网络模型提取器   Net::create_extractor();
                                模型提取器提取某一层输出 Extractor::extract();

        ./src/layer下是所有的layer定义代码
        ./src/layer/arm是arm下的计算加速的layer
        ./src/layer/x86是x86下的计算加速的layer。
