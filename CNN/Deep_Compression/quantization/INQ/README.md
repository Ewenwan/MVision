# INQ Incremental-Network-Quantization 渐进式神经网络量化

# 给定任意结构的全精度浮点神经网络模型，能将其转换成无损的低比特二进制模型

# 英特尔中国研究院：INQ神经网络无损低比特量化技术 全精度网络输入，输出权值为0或2的整数次幂的网络
![](https://xmfbit.github.io/img/paper-inq-quantize-set.png)

[代码](https://github.com/Ewenwan/Incremental-Network-Quantization)

[论文](https://arxiv.org/pdf/1702.03044.pdf)

[参考1](http://zhidx.com/p/94098.html)

[参考2](https://xmfbit.github.io/2018/01/25/inq-paper/)

### Experimental Results

The authors adopted the proposed method to several model, including AlexNet, VGG-16, GoogleNet, ResNet-18 and ResNet-50. More experiments for exploration was conducted on ResNet-18. Experimental results on ImageNet using center crop validation are shown as follows.

| Network       | Bit-width | Top-1/Top-5 Error | Decrease in Top-1/Top-5 Error | Portion                   |
| ------------- | :-------- | ----------------- | ----------------------------- | ------------------------- |
| AlexNet ref   | 32        | 42.71%/19.77%     |                               |                           |
| AlexNet       | 5         | **42.61%/19.54%** | 0.15%/0.23%                   | {0.3, 0.6, 0.8, 1.0}      |
| VGG-16 ref    | 32        | 31.46%/11.35%     |                               |                           |
| VGG-16        | 5         | **29.18%/9.70%**  | 2.28%/1.65%                   | {0.5, 0.75, 0.875, 1.0}   |
| GoogleNet ref | 32        | 31.11%/10.97%     |                               |                           |
| GoogleNet     | 5         | **30.98%/10.72%** | 0.13%/0.25%                   | {0.2, 0.4, 0.6, 0.8, 1.0} |
| ResNet-18 ref | 32        | 31.73%/11.31      |                               |                           |
| ResNet        | 5         | **31.02%/10.90%** | 0.71%/0.41                    | {0.5, 0.75, 0.875, 1.0}   |
| ResNet-50 ref | 32        | 26.78%/8.76%      |                               |                           |
| ResNet-50     | 5         | **25.19%/7.55%**  | 1.59%/1.21%                   | {0.5, 0.75, 0.875, 1.0}   |

      Number of required epochs for training increasing with the expected bit-width going down.
      The accumulated portions for weight quantization are set as {0.3, 0.5, 0.8, 0.9, 0.95, 1.0}, 
      {0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0}, {0.2, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.975,  1.0} 
      for 4-bits to 2-bits, respectively. Training epochs required for 2-bits finally 
      set to 30 which means that 300 training epochs are required for completing a full quantization procedure.
      In the other words, the proposed method become time-consuming when the network going deeper.

      Although the authors convert weights to the powers of 2 and claim that their 
      method would be efficient with binary shift operation in hardware, 
      the computation in there experiments is still using floating operations.
      Thus they only show the results of model compression instead of speeding up computation.
      
# 量化策略 Method
* 提出了渐进式神经网络量化的思想，引入了三种操作：参数分组，量化，重训练.
      
      即 weight partition（权重划分）, group-wise quantization（分组量化） 和re-training（训练）.
      
      re-training好理解，就是量化之后要继续做finetuning。
      
      前面两个名词解释如下：weight partition是指我们不是对整个权重一股脑地做量化，而是将其划分为两个不相交的集合。
      
      group-wise quantization是指对其中一个集合中的权重做量化，另一组集合中的权重不变，仍然为FP32。
      
      (在权重划分步骤，作者指出，随机地将权重量化，不如根据权重的幅值，优先量化那些绝对值比较大的权重。)
      
      注意，在re-training中，我们只对没有量化的那组参数做参数更新。
      
      简单的说就是在训练得到一个网络模型后，
      首先将这个全精度浮点网络模型中的每一层参数分为两组，
      第一组中的参数直接被量化固定，
      另一组参数通过重训练来补偿量化给模型造成的精度损失。
      然后这三个操作依次迭代应用到刚才的第二组完成重训练之后的全精度浮点参数部分，直到模型全部量化为止。
      
      可以说参数分组分成的这两个部分是互补的作用，
      一个建立低精度模型的基础，
      另一个通过retrain(重训练，微调)补偿精度损失；
      这样迭代最终得到渐进式的量化和精度提升。
      通过巧妙耦合参数分组、量化和重训练操作，该技术抑制了模型量化造成的性能损失，从而在实际中适用于任意结构的神经网络模型。
      
      INQ渐进式网络量化策略：
![](http://zhidx.com/wp-content/uploads/2017/09/85fd56a6c52852178bcb2e3e79681ca%E5%89%AF%E6%9C%AC.png)
      
      （绿线代表当前已经被量化的网络连接；蓝线代表需要重新训练的网络连接）
      
# 具体量化方式
    该技术还包含另外两个亮点。
    其一，在模型量化过程中，所有参数被限制成二进制表示，并包含零值，极限量化结果即为三值网络或者二值网络。
         这种量化使得最后的模型非常适合在硬件上部署和加速。
         比如在FPGA上，复杂的全精度浮点乘法运算将被直接替换为简单的移位操作。
    其二，现有神经网络量化压缩方法在处理二值网络或者三值网络时，为了让模型精度损失不至于太大，
         往往将模型第一层和最后一层参数依然保留为全精度浮点型，
         而我们的技术在对模型的所有参数进行量化的同时，实现了性能的全面领先 。
         
![](http://zhidx.com/wp-content/uploads/2017/09/6a56b64514919da3da833874edc60a8%E5%89%AF%E6%9C%AC.png)
      
            INQ渐进式网络量化示例

            第一行：依次为参数分组、量化与重训练；
            (将所有的权重划分为黑色和白色两个部分（图1）。
             黑色部分的权重进行量化，白色部分不变（图2）。
             然后，使用SGD更新那些白色部分的权重（图3）。)

            第二行：迭代过程

            （绿色区域代表当前已经被量化的网络参数；浅紫区域代表需要重新训练的网络参数）
             在第二次迭代中，我们扩大量化权重的范围，重复进行迭代1中的操作。
             在后面的迭代中，以此类推，只不过要不断调大量化权重的比例，最终使得所有权重都量化为止。

# 具体量化形式
![](https://xmfbit.github.io/img/paper-inq-quantize-set.png)


      假设用bbit表示权重，我们分出1位单独表示0。

      PS：这里插一句。关于为什么要单独分出1位表示0，毕竟这样浪费了(2b vs 2b−1+1)。
      GitHub上有人发issue问，作者也没有正面回复这样做的原因。
      以我的理解，是方便判定0和移位。
      因为作者将权重都舍入到了2的幂次，那肯定是为了后续将乘法变成移位操作。
      而使用剩下的b−1表示，可以方便地读出移位的位数，进行操作。

      这样，剩下的b−1位用来表示2的幂次。
      我们需要决定n1和n2。因为它俩决定了表示范围。它们之间的关系为：

      (n1−n2+1)×2 = 2^(b−1)

      其中，乘以2是考虑到正负对称的表示范围。

      如何确定n1呢（由上式可知，有了b和n1，那么n2就确定了

      作者考虑了待量化权重中的最大值，我们需要设置n1，使其刚好不溢出。
      所以有：

      n1=⌊log2(4s/3)⌋ ，⌊X⌋ 为向下取整

      其中，s是 权重 当中 绝对值 最大的那个数的绝对值，即 s = max|Wl|。
       
      之后做最近舍入就可以了。对于小于最小分辨力2n2的那些权重，将其直接截断为0。
# 代码部分
## 量化部分
      在代码部分，INQ基于Caffe框架，主要修改的地方集中于blob.cpp和sgd_solver.cpp中。
      量化部分的代码如下，首先根据要划分的比例计算出两个集合分界点处的权重大小。
      然后将大于该值的权重进行量化，小于该值的权重保持不变。
      下面的代码其实有点小问题，data_copy使用完之后没有释放。
      关于代码中mask的作用，就是在反向传播更新参数时，已经量化的参数，就不再进行更新，未量化的参数进行更新。
      
```c

// blob.cpp
// INQ  
if(is_quantization)
{
  Dtype* data_copy=(Dtype*) malloc(count_*sizeof(Dtype));
  
  caffe_copy(count_,data_vec,data_copy);// 复制原始数据
  caffe_abs(count_,data_copy,data_copy);// 绝对值
  
  // 对权重进行排序，获取权重的有序序列 升序排列
  std::sort(data_copy,data_copy+count_); //data_copy order from small to large
  
  // caculate the n1
  Dtype max_data = data_copy[count_-1];// 最后一个数为最大值
  int n1=(int)floor(log2(max_data*4.0/3.0));// 按公式计算 n1 量化范围
  
  // quantizate the top 30% of each layer, change the "partition" until partition=0
  // 计算分组间隔点
  int partition=int(count_*0.7)-1;// 量化最后面的30%(值大的部分)
  
  for (int i = 0; i < (count_); ++i) {// 遍历每一个 权重值
  
    if(std::abs(data_vec[i])>=data_copy[partition])// 选取值大于分割点
      {
        data_vec[i] = weightCluster_zero(data_vec[i],n1);// 进行量化
 
        mask_vec[i]=0;// 标记，该量化后的权重参数，在sgd反向传播更新参数时，仅仅对未量化的参数进行更新
      }
  }
  // 代码其实有点小问题，data_copy 使用完之后没有释放
  // free data_copy
 }
```

## 参数更新
      在re-training中，我们只对未量化的那些参数进行更新。
      待更新的参数(未量化的参数)，mask中的值都是1，这样和diff相乘仍然不变；
      不更新的参数(已经量化过的参数)，mask中的值都是0，和diff乘起来，相当于强制把梯度变成了0。
```c

// sgd_solver.cpp
caffe_gpu_mul(net_params[param_id]->count(),net_params[param_id]->gpu_mask(),net_params[param_id]->mutable_gpu_diff(),net_params[param_id]->mutable_gpu_diff());

```
      
