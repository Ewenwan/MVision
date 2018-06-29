# 新层 定义 src/caffe/proto/caffe.proto
```asm
 message LayerParameter {
   optional string name = 1; // the layer name
   optional string type = 2; // the layer type
   optional PowerParameter power_param = 122;
   optional PReLUParameter prelu_param = 131;
   optional PythonParameter python_param = 130;
+  optional QuantizationParameter quantization_param = 145; //////// 新添 层参数 quantization_param
   optional ReductionParameter reduction_param = 136;
   optional ReLUParameter relu_param = 123;
   optional ReshapeParameter reshape_param = 133;
   optional WindowDataParameter window_data_param = 129;
 }
 
 message QuantizationParameter{
 
+	enum Precision {           // 量化策略 3种量化方法 ==========================
+		FIXED_POINT = 0;         // 定点/动态定点
+		MINI_FLOATING_POINT = 1; // 迷你浮点
+		POWER_2_WEIGHTS = 2;     // 二进制指数
+	}
+	optional Precision precision = 1 [default = FIXED_POINT]; // 量化策略 precision 默认为 定点/动态定点

+  enum Rounding {           // 量化的舍入方案 取整方法 ============================
+		NEAREST = 0;             // 最近偶数 
+		STOCHASTIC = 1;          // 随机舍入(向上或向下取整)
+	}
+	optional Rounding rounding_scheme = 2 [default = NEAREST];// 量化的舍入方案 rounding_scheme  默认为 最近偶数

+	// Fixed point precision  定点量化
+  optional uint32 bw_layer_out = 3 [default = 32];
+	optional uint32 bw_params = 4 [default = 32];
+	optional int32 fl_layer_out = 5 [default = 16];
+	optional int32 fl_params = 6 [default = 16];

+	// Mini floating point precision  迷你浮点数 
+  optional uint32 mant_bits = 7 [default = 23]; // 尾数尾数 32位：1+8+23    16位: 1+5+10
+	optional uint32 exp_bits = 8 [default = 8];    // 指数尾数

+	// Power-of-two weights  二进制指数
+  optional int32 exp_min = 9 [default = -8];   // 使用的最小指数  2(-8)
+	optional int32 exp_max = 10 [default = -1];   // 使用的最大指数  2(-1)
+}
```
