# 联结主义时间分类器 Connectionist Temporal Classifier 

为什么要发明CTC，对于真实世界的序列学习任务，数据往往含有噪声和没有预先分割。RNN是一个强大的序列学习模型，但是需要对数据进行预先处理，所以有了CTC我们就能够提升RNN的性能。

用来解决输入序列和输出序列难以一一对应的问题。

举例来说，在语音识别中，我们希望音频中的音素和翻译后的字符可以一一对应，这是训练时一个很天然的想法。但是要对齐是一件很困难的事，有人说话块，有人说话慢，每个人说话快慢不同，不可能手动地对音素和字符对齐，这样太耗时。

再比如，在OCR中使用RNN时，RNN的每一个输出要对应到字符图像中的每一个位置，要手工做这样的标记工作量太大，而且图像中的字符数量不同，字体样式不同，大小不同，导致输出不一定能和每个字符一一对应。

[一文读懂 CRNN文字检测 + CTC文字识别](https://zhuanlan.zhihu.com/p/43534801)

[CTC（Connectionist Temporal Classification）介绍](https://www.cnblogs.com/liaohuiqiang/p/9953978.html)

[关于CTC模型的理解](https://blog.csdn.net/gzj_1101/article/details/80153686)

[tensorflow LSTM+CTC实现端到端的不定长数字串识别](https://www.jianshu.com/p/45828b18f133)

[Use CTC + tensorflow to OCR ](https://github.com/ilovin/lstm_ctc_ocr)

[caffe + WarpCTC](https://github.com/xmfbit/warpctc-caffe)

    适合于输入特征和输出标签之间对齐关系不确定的时间序列问题，
    CTC可以自动端到端地同时优化模型参数和对齐切分的边界。
    
    比如本文例子，32 x 256大小的图片，最大可切分256列，也就是输入特征最大256，
    而输出标签的长度最大设定是18，这种就可以用CTC模型进行优化。
    关于CTC模型，笔者认为可以这样理解，假设32 x 256的图片，数字串标签是"123"，
    把图片按列切分（CTC会优化切分模型），然后分出来的每块再去识别数字，
    找出这块是每个数字或者特殊字符的概率（无法识别的则标记为特殊字符"-"），
    这样就得到了基于输入特征序列（图片）的每一个相互独立建模单元个体（划分出来的块）（包括“-”节点在内）的类属概率分布。
    基于概率分布，算出标签序列是"123"的概率P（123），当然这里设定"123"的概率为所有子序列之和，
    这里子序列包括'-'和'1'、'2'、'3'的连续重复.


    
## 文字识别 OCR

[文字识别OCR方法整理](https://zhuanlan.zhihu.com/p/65707543)

文字识别也是图像领域一个常见问题。然而，对于自然场景图像，首先要定位图像中的文字位置，然后才能进行识别。

所以一般来说，从自然场景图片中进行文字识别，需要包括2个步骤：

1.文字检测：解决的问题是哪里有文字，文字的范围.  
2.文字识别：对定位好的文字区域进行识别，主要解决的问题是每个文字是什么，将图像中的文字区域进转化为字符信息.  

[场景文字检测 — CTPN原理与实现 ](https://zhuanlan.zhihu.com/p/34757009)

对于复杂场景的文字识别，首先要定位文字的位置，即文字检测。

### 文字检测（Text Detection）
文字检测定位图片中的文本区域，而Detection定位精度直接影响后续Recognition结果。

EAST/CTPN/SegLink/PixelLink/TextBoxes/TextBoxes++/TextSnake/MSR/...

CTPN是在ECCV 2016提出的一种文字检测算法。CTPN结合CNN与LSTM深度网络，能有效的检测出复杂场景的横向分布的文字，是目前比较好的文字检测算法。

由于CTPN是从Faster RCNN改进而来，本文默认读者熟悉CNN原理和Faster RCNN网络结构。

[一文读懂Faster RCNN](https://zhuanlan.zhihu.com/p/31426458)

> Faster RCNN其实可以分为4个主要内容：

1.Conv layers。作为一种CNN网络目标检测方法，Faster RCNN首先使用一组基础的conv+relu+pooling层提取image的feature maps。该feature maps被共享用于后续RPN层和全连接层。  
2.Region Proposal Networks。RPN网络用于生成region proposals。该层通过softmax判断anchors属于foreground或者background，再利用bounding box regression修正anchors获得精确的proposals。  
3.Roi Pooling。该层收集输入的feature maps和proposals，综合这些信息后提取proposal feature maps，送入后续全连接层判定目标类别。  
4.Classification。利用proposal feature maps计算proposal的类别，同时再次bounding box regression获得检测框最终的精确位置。  

[CTPN相关：caffe代码](https://github.com/tianzhi0549/CTPN)

原始CTPN只检测横向排列的文字。CTPN结构与Faster R-CNN基本类似，但是加入了LSTM层。

卷积网络之后 使用 双向 LSTM提取特征 (包含空间特征，也包含了LSTM学习到的序列特征) 再经过“FC”卷积层，最后经过类似Faster R-CNN的RPN网络，获得text proposals。

[完全解析RNN, Seq2Seq, Attention注意力机制](https://zhuanlan.zhihu.com/p/51383402)

循环神经网络RNN结构被广泛应用于机器翻译，语音识别，文字识别OCR等方向。

CNN学习的是感受野内的空间信息，LSTM学习的是序列特征。对于文本序列检测，显然既需要CNN抽象空间特征，也需要序列特征（毕竟文字是连续的）。

CTPN中使用双向LSTM，相比一般单向LSTM有什么优势？双向LSTM实际上就是将2个方向相反的LSTM连起来.

> 总结:

1.由于加入LSTM，所以CTPN对水平文字检测效果超级好。  
2.因为Anchor设定的原因，CTPN只能检测横向分布的文字，小幅改进加入水平Anchor即可检测竖直文字。但是由于框架限定，对不规则倾斜文字检测效果非常一般。 

倾斜文字 可以想办法 校准为 水平文字???

3.CTPN加入了双向LSTM学习文字的序列特征，有利于文字检测。但是引入LSTM后，在训练时很容易梯度爆炸，需要小心处理。  
  
### 文字识别（Text Recognition）

识别水平文本行，一般用CRNN或Seq2Seq两种方法.

> 常用文字识别算法主要有两个框架：

1. CNN+RNN+CTC(CRNN+CTC)  
  
2. CNN+Seq2Seq+Attention  

CNN+Seq2Seq+Attention+word2vec

对于特定的弯曲文本行识别，早在CVPR2016就已经有了相关paper：

[Robust Scene Text Recognition with Automatic Rectification](https://arxiv.org/pdf/1603.03915.pdf)

对于弯曲不规则文本，如果按照之前的识别方法，直接将整个文本区域图像强行送入CNN+RNN，由于有大量的无效区域会导致识别效果很差。所以这篇文章提出一种通过**STN网络Spatial Transformer Network(STN)**学习变换参数，将Rectified Image对应的特征送入后续RNN中识别。

[STN网络Spatial Transformer Network(STN)](https://arxiv.org/pdf/1506.02025.pdf)

对于STN网络，可以学习一组点 (x_i^s,y_i^s) 到对应点 (x_i^t,y_i^t) 的变换。而且STN可以插入轻松任意网络结构中学习到对应的变换。

    (x_i^s,
    y_i^s)    =  (c11, c12, c13
                  c21, c22, c23)    *  (x_i^t,
                                        y_i^t,
                                        1) 
**核心就是将传统二维图像变换（如旋转/缩放/仿射等）End2End融入到网络中。**

文字检测和文字识别是分为两个网络分别完成的，所以一直有研究希望将OCR中的Detection+ Recognition合并成一个End2End网络。目前End2End OCR相关研究如下：

[Li_Towards_End-To-End_Text](http://openaccess.thecvf.com/content_ICCV_2017/papers/Li_Towards_End-To-End_Text_ICCV_2017_paper.pdf)

该篇文章采用Faster R-CNN的Two-stage结构：首先Text Proposal Network（即RPN）生成对应的文本区域Text Proposal，后续通过Bounding Box regression和Box Classification进一步精修文本位置。但是不同的是，在RoI Pooling后接入一个LSTM+Attention的文字识别分支中.

但是这样的结构存在问题。举例说明：Faster R-CNN的RPN只是初步产生Proposal，后续还需要再经过一次Bounding Box regression才能获取准确的检测框.

所以Text Proposal不一定很准会对后续识别分支产生巨大影响，导致该算法在复杂数据集上其实并不是很work。



#### 1. CNN+RNN+CTC(CRNN+CTC)  

[OCR_TF_CRNN_CTC 代码 ](https://github.com/bai-shang/OCR_TF_CRNN_CTC)
  
[论文](https://arxiv.org/pdf/1507.05717.pdf)
  
2. CNN+Seq2Seq+Attention  

> 整个CRNN网络可以分为三个部分:

0.假设输入图像大小为 (32, 100,3)，注意提及图像都是 [\text{Height},\text{Width},\text{Channel}] 形式。

1.Convlutional Layers

这里的卷积层就是一个普通的CNN网络，用于提取输入图像的Convolutional feature maps，即将大小为 (32, 100,3) 的图像转换为 (1,25,512) 大小的卷积特征矩阵。

2. Recurrent Layers

这里的循环网络层是一个深层双向LSTM网络，在卷积特征的基础上继续提取文字序列特征。

所谓深层RNN网络，是指超过两层的RNN网络。

3. Transcription Layers

将RNN输出做softmax后，通过转化为字符。

对于Recurrent Layers，如果使用常见的Softmax Loss，则每一列输出都需要对应一个字符元素。那么训练时候每张样本图片都需要标记出每个字符在图片中的位置，再通过CNN感受野对齐到Feature map的每一列获取该列输出对应的Label才能进行训练，如图8。

在实际情况中，标记这种对齐样本非常困难，工作量非常大。另外，由于每张样本的字符数量不同，字体样式不同，字体大小不同，导致每列输出并不一定能与每个字符一一对应。

当然这种问题同样存在于语音识别领域。例如有人说话快，有人说话慢，那么如何进行语音帧对齐，是一直以来困扰语音识别的巨大难题。

所以Connectionist Temporal Classification(CTC)提出一种对不需要对齐的Loss计算方法，用于训练网络，被广泛应用于文本行识别和语音识别中。'

CRNN+CTC总结将CNN/LSTM/CTC三种方法结合：

1.首先CNN提取图像卷积特征   
2.然后LSTM进一步提取图像卷积特征中的序列特征  
3.最后引入CTC解决训练时字符无法对齐的问题  
即提供了一种end2end文字图片识别算法，也算是OCR方向的简单入门文章。  






