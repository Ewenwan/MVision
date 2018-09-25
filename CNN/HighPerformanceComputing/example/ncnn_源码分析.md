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
      
      
