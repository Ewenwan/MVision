# 轻量级网络--ShuffleNet 分组点卷积+通道重排+逐通道卷积
## 进化过程 ResNet ---> MobileNet ----> ShuffleNet

[论文地址](https://arxiv.org/pdf/1707.01083.pdf)

[tf代码](https://github.com/MG2033/ShuffleNet)

[caffe代码](https://github.com/Ewenwan/ShuffleNet-1)

[shufflenet-ssd](https://github.com/linchaozhang/shufflenet-ssd)

[预训练的ShuffleNet-cafe模型参数文件](https://github.com/msnqqer/ShuffleNet)

[预训练模型文件 password is "bcj6"](https://pan.baidu.com/s/1eS8NOm2)

## ResNet 残差网络  结合不同层特征
     ________________________________>
    |                                 ADD -->  f(x) + x
    x-----> 1x1 + 3x3 + 1x1 卷积 -----> 
    
## MobileNet 普通点卷积 + 逐通道卷积 + 普通点卷积

    1. 步长为1结合x shortcut
    ___________________________________>
    |                                    ADD -->  f(x) + x
    x-----> 1x1 + 3x3DW + 1x1 卷积 ----->  
         “扩张”→“卷积提特征”→ “压缩”
    ResNet是：压缩”→“卷积提特征”→“扩张”，MobileNetV2则是Inverted residuals,即：“扩张”→“卷积提特征”→ “压缩”

    2. 步长为2时不结合x 
    x-----> 1x1 + 3x3DW(步长为2) + 1x1 卷积 ----->   输出
## ShuffleNet 普通点卷积 变为分组点卷积+通道重排   逐通道卷积
普通点卷积时间还是较长

### 版本1：
    ___________________________________________________________>
    |                                                            ADD -->  f(x) + x
    x-----> 1x1分组点卷积 + 通道重排 + 3x3DW + 1x1分组点卷积 ----->
    
### 版本2：（特征图降采样）
    _____________________3*3AvgPool_________________________________>
    |                                                                concat -->  f(x) 链接  x
    x-----> 1x1分组点卷积 + 通道重排 + 3x3DW步长2 + 1x1分组点卷积 ----->  
# 通道重排三部曲 

| group 1  | group 2  |  group 3  |  
| :------: | :------: | :-------: |  
| 1     2  | 3     4  |  5     6  |   

Each nubmer represents a channel of the feature map  
    
## step 1: Reshape  按分组变形成列矩阵
1  2  
3  4   
5  6 
## step 2: transpose  转置
1 3 5  
2 4 6　　
## step 3: flatten    按分组数量 平滑成行向量

| group 1  | group 2  |  group 3  |  
| :-----:  | :------: | :-------: |  
| 1     3  | 5     2  |  4     6  |  
