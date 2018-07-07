# 融合局部感受野内的空间信息和通道信息来提取信息特征

* SENet  
[Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)
[中文版](http://noahsnail.com/2017/11/20/2017-11-20-Squeeze-and-Excitation%20Networks%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E6%96%87%E7%89%88/)
[中英文对照](http://noahsnail.com/2017/11/20/2017-11-20-Squeeze-and-Excitation%20Networks%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E8%8B%B1%E6%96%87%E5%AF%B9%E7%85%A7/)

[博客翻译](https://blog.csdn.net/Quincuntial/article/details/78605463)

[SENet代码](https://github.com/Ewenwan/SENet/blob/master/README.md)

# Squeeze-and-Excitation Networks <sub>([arXiv](https://arxiv.org/pdf/1709.01507.pdf))</sub>
By Jie Hu<sup>[1]</sup>, Li Shen<sup>[2]</sup>, Gang Sun<sup>[1]</sup>.

[Momenta](https://momenta.ai/)<sup>[1]</sup> and [University of Oxford](http://www.robots.ox.ac.uk/~vgg/)<sup>[2]</sup>.

## Approach
<div align="center">
  <img src="https://github.com/hujie-frank/SENet/blob/master/figures/SE-pipeline.jpg">
</div>
<p align="center">
  Figure 1: Diagram of a Squeeze-and-Excitation building block.
</p>

<div align="center">
   <img src="https://github.com/hujie-frank/SENet/blob/master/figures/SE-Inception-module.jpg" width="420">
  <img src="https://github.com/hujie-frank/SENet/blob/master/figures/SE-ResNet-module.jpg"  width="420">
</div>
<p align="center">
  Figure 2: Schema of SE-Inception and SE-ResNet modules. We set r=16 in all our models.
</p>

## Implementation
In this repository, Squeeze-and-Excitation Networks are implemented by [Caffe](https://github.com/BVLC/caffe).

### Augmentation
| Method | Settings |
|:-:|:-:|
|Random Mirror| True |
|Random Crop| 8% ~ 100% |
|Aspect Ratio | 3/4 ~ 4/3 |
|Random Rotation| -10° ~ 10°|
|Pixel Jitter| -20 ~ 20 |

### Note:
* To achieve efficient training and testing, we combine the consecutive operations ***channel-wise scale*** and ***element-wise summation*** into a single layer **"Axpy"** in the architectures with skip-connections, resulting in a considerable reduction in memory cost and computational burden.

* In addition, we found that the implementation for ***global average pooling*** on GPU supported by cuDNN and BVLC/caffe is less efficient. In this regard, we re-implement the operation which achieves significant acceleration.

## Trained Models

Table 1. Single crop validation error on ImageNet-1k (center 224x224 crop from resized image with shorter side = 256). The SENet-154 is one of our superior models used in [ILSVRC 2017 Image Classification Challenge](http://image-net.org/challenges/LSVRC/2017/index) where we won the 1st place (Team name: [WMW](http://image-net.org/challenges/LSVRC/2017/results)).

| Model | Top-1 | Top-5 | Size | Caffe Model | Caffe Model
|:-:|:-:|:-:|:-:|:-:|:-:|
|SE-BN-Inception| 23.62 | 7.04 | 46 M| [GoogleDrive](https://drive.google.com/file/d/0BwHV3BlNKkWlTWRRbDZYbVB2WWc/view?usp=sharing) | [BaiduYun](https://pan.baidu.com/s/1qYoPdak)
|SE-ResNet-50   | 22.37 | 6.36 | 107 M | [GoogleDrive](https://drive.google.com/file/d/0BwHV3BlNKkWlS2QwZHFzM3RjNzg/view?usp=sharing) | [BaiduYun](https://pan.baidu.com/s/1gf5wsLl)
|SE-ResNet-101  | 21.75  | 5.72 | 189 M | [GoogleDrive](https://drive.google.com/file/d/0BwHV3BlNKkWlTEg4YmcwQ0FoZFU/view?usp=sharing) | [BaiduYun](https://pan.baidu.com/s/1c1FvCWg)
|SE-ResNet-152  | 21.34  | 5.54 | 256 M | [GoogleDrive](https://drive.google.com/file/d/0BwHV3BlNKkWlcFE0Q2NTcWl3WUE/view?usp=sharing) | [BaiduYun](https://pan.baidu.com/s/1dFEnSzR)
|SE-ResNeXt-50 (32 x 4d) | 20.97 | 5.54 | 105 M | [GoogleDrive](https://drive.google.com/file/d/0BwHV3BlNKkWlQ2Z0Q204V1RITjA/view?usp=sharing) | [BaiduYun](https://pan.baidu.com/s/1dFbEmbv)
|SE-ResNeXt-101 (32 x 4d) | 19.81 | 4.96 | 187 M | [GoogleDrive](https://drive.google.com/file/d/0BwHV3BlNKkWleklsNzBiZlprblk/view?usp=sharing) | [BaiduYun](https://pan.baidu.com/s/1qY2wjt6)
|SENet-154 | 18.68 | 4.47 | 440 M | [GoogleDrive](https://drive.google.com/file/d/0BwHV3BlNKkWlbTFZbzFTSXBUTUE/view?usp=sharing) | [BaiduYun](https://pan.baidu.com/s/1o7HdfAE)

Here we obtain better performance than those reported in the paper.
We re-train the SENets described in the paper on a single GPU server with 8 NVIDIA Titan X cards, using a mini-batch of 256 and a initial learning rate of 0.1 with more epoches. 
In contrast, the results reported in the paper were obtained by training the networks with a larger batch size (1024) and learning rate (0.6) across 4 servers. 

## Third-party re-implementations
0. Caffe. SE-mudolues are integrated with a modificated ResNet-50 using a stride 2 in the 3x3 convolution instead of the first 1x1 convolution which obtains better performance: [Repository](https://github.com/shicai/SENet-Caffe).
0. TensorFlow. SE-modules are integrated with a pre-activation ResNet-50 which follows the setup in [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch): [Repository](https://github.com/ppwwyyxx/tensorpack/tree/master/examples/ResNet).
0. TensorFlow. Simple Tensorflow implementation of SENets using Cifar10: [Repository](https://github.com/taki0112/SENet-Tensorflow).
0. MatConvNet. All the released SENets are imported into [MatConvNet](https://github.com/vlfeat/matconvnet): [Repository](https://github.com/albanie/mcnSENets).
0. MXNet. SE-modules are integrated with the ResNeXt and more architectures are coming soon: [Repository](https://github.com/bruinxiong/SENet.mxnet).
0. PyTorch. Implementation of SENets by PyTorch: [Repository](https://github.com/moskomule/senet.pytorch).
0. Chainer. Implementation of SENets by Chainer: [Repository](https://github.com/nutszebra/SENets).
## Citation

If you use Squeeze-and-Excitation Networks in your research, please cite the paper:
    
    @inproceedings{hu2018senet,
      title={Squeeze-and-Excitation Networks},
      author={Jie Hu and Li Shen and Gang Sun},
      journal={IEEE Conference on Computer Vision and Pattern Recognition},
      year={2018}
    }
