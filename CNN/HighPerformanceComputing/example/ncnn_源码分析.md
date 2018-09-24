# 1. param 和 bin 文件分析
## param
      7767517
      75 83
      Input            data             0 1 data 0=227 1=227 2=3
      Convolution      conv1            1 1 data conv1 0=64 1=3 2=1 3=2 4=0 5=1 6=1728
      ReLU             relu_conv1       1 1 conv1 conv1_relu_conv1 0=0.000000
      Pooling          pool1            1 1 conv1_relu_conv1 pool1 0=0 1=3 2=2 3=0 4=0
      Convolution      fire2/squeeze1x1 1 1 pool1 fire2/squeeze1x1 0=16 1=1 2=1 3=1 4=0 5=1 6=1024
