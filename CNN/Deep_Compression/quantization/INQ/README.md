# INQ Incremental-Network-Quantization 
# 英特尔中国研究院：INQ神经网络无损低比特量化技术 全精度网络输入，输出权值为0或2的整数次幂的网络
[代码](https://github.com/Ewenwan/Incremental-Network-Quantization)


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
