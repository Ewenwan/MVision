
# Speech Automatic Speech Recognition,(ASR)

语音识别，通俗来讲，就是将一段语音信号转换成对应的文本信息。具体来说，语音识别是从一段连续声波中采样，将每个采样值量化；然后对量化的采样音频进行分帧，对于每一帧，抽取出一个描述频谱内容的特征向量；最后根据语音信号的特征识别语音所代表的单词。

    包含：

        语言识别ASR 
        语义理解ALU 
        文字转语言TTS  
        文字识别ocr 
        声纹识别 VPR
        回声消除  AEC/AES
[语音识别 RNN LSTM HMM GMM CTC The official repository of the Eesen project](https://github.com/Ewenwan/eesen)

[声纹识别发展综述](https://zhuanlan.zhihu.com/p/67563275)

> 回声消除  AEC/AES

Android 和 WebRTC 里应该都有相应的模块。

回声消除指的是 AEC/AES，在手机上用于消除手机 MIC 录进手机扬声器的对方通话声音，避免对方在通话时听到自己的声音，不是指 dereverberation（去混响）。

自适应回声消除器

回声是由于扬声器放出来的声音经过衰减和延时然后又被麦克风收录产生的。自适应回声消除器简单说就是用一个参数可调的滤波器，通过一个自适应算法，模拟回声产生的信道环境，进而“猜测”回声信号，然后在麦克风收录的信号里“减去”这个信号。

回声消除的效果和采用的算法有关，一般有LMS,NLMS,RLS,APA算法等等，算法太复杂，就不多讲了。。。

从上面的描述你应该可以看出来，你的声音是在对方设备上产生的回声，所以你的声音的回声是在对方设备上消除的，同理，对方声音得回声在你的设备上消除

[即时语音（如：YY语音）中回声消除技术是如何实现的？](https://www.zhihu.com/question/21406954/answer/5072738420

## 语言识别ASR 
![](http://antkillerfarm.github.io/images/img2/speech.png)

![](http://antkillerfarm.github.io/images/img2/speech_2.png)

语音识别的整个流程，主要包含特征提取和解码（声学模型、字典、语言模型）部分。

1. 特征提取：从语音波形中提取出随时间变化的语音特征序列（即将声音信号从时域转换到频域），为声学模型提供合适的特征向量。主要算法有线性预测倒谱系数（LPCC）和梅尔频率倒谱系数（MFCC）。

2. 声学模型：根据声学特性计算每一个特征向量在声学特征上的得分，输入是特征向量，输出为音素信息。最常用的声学建模方式是隐马尔科夫模型（HMM），基于深度学习的发展，深度神经网络（DNN）、卷积神经网络（CNN）、循环神经网络（RNN）等模型在观测概率的建模中取得了非常好的效果。

在语音识别整个流程中，声学模型作为识别系统的底层模型，声学模型的任务是计算 P(O|W)，（即模型生成观察序列的概率），它占据着语音识别大部分的计算开销，决定着语音识别系统的性能。所以，声学模型是语音识别系统中最关键的一部分。

3. 字典：字或者词与音素的对应，中文就是拼音和汉字的对应，英文就是音标与单词的对应。（音素，单词的发音由音素构成。对英语来说，一种常用的音素集是卡内基梅隆大学的一套由 39 个音素构成的音素集，汉语一般直接用全部声母和韵母作为音素集）。

4. 语言模型：通过对大量文本信息进行训练，得到单个字或者词相互关联的概率。语音识别中，最常见的语言模型是 N-Gram。近年，深度神经网络的建模方式也被应用到语言模型中，比如基于 CNN 及 RNN 的语言模型。

5. 解码：通过声学模型、字典、语言模型对提取特征后的音频数据进行文字输出。

[语音识别（一）——概述 HMM -> GMM -> 深度学习RNN  HTK CMU-Sphinx SPTK ](http://antkillerfarm.github.io/graphics/2018/04/16/speech.html)

[语音识别（二）——基本框架, Microphone Array, 声源定位 信号处理和特征提取 MFCC、声学模型(gmm-hmm)、语言模型（Language Model, LM）和解码器(Decoder)](http://antkillerfarm.github.io/graphics/2018/04/17/speech_2.html)()

[语音识别（三）——声源定位、前端处理 语言模型 声学模型, 解码器技术](http://antkillerfarm.github.io/graphics/2018/04/23/speech_3.html)

[语音识别（四）——声音分割，DTW(时域,Dynamic Time Warping动态时间规整算法), Spectrogram(频域,FFT傅里叶变换，声谱图), Cepstrum Analysis, Mel-Frequency Analysis](http://antkillerfarm.github.io/graphics/2018/06/01/speech_4.html)

[语音识别（五）——FBank, 语音识别的评价指标, 声学模型进阶, 语言模型进阶, GMM-HMM高斯混合-隐马尔科夫模型](http://antkillerfarm.github.io/graphics/2018/06/06/speech_5.html)

[中文分词!!!!!!!](https://github.com/Ewenwan/cppjieba)


[阿里巴巴的 DFSMN 声学模型 基于 开源的语音识别工具 Kaldi DFSMN 是 Kaldi 的一个补丁文件，所以，为了使用 DFSMN 模型，我们必须先部署 Kaldi 语音识别工具 ]()

### DFSMN  && Kaldi

[参考](https://www.zhihu.com/search?type=content&q=%E8%AF%AD%E9%9F%B3%E8%AF%86%E5%88%AB%E6%A8%A1%E5%9E%8B%20DFSMN)

目前主流的语音识别系统普遍采用基于深度神经网络和隐马尔可夫（Deep Neural Networks-Hidden Markov Model，DNN-HMM）的声学模型.

声学模型的输入是传统的语音波形经过加窗、分帧，然后提取出来的频谱特征，如 PLP， MFCC 和 FBK等。而模型的输出一般采用不同粒度的声学建模单元，例如单音素 (mono-phone)、单音素状态、绑定的音素状态 (tri-phonestate) 等。从输入到输出之间可以采用不同的神经网络结构，将输入的声学特征映射得到不同输出建模单元的后验概率，然后再结合HMM进行解码得到最终的识别结果。

最早采用的网络结构是前馈全连接神经网路（Feedforward Fully-connected Neural Networks, FNN）。FNN实现固定输入到固定输出的一对一映射，其存在的缺陷是没法有效利用语音信号内在的长时相关性信息。一种改进的方案是采用基于长短时记忆单元（Long-Short Term Memory，LSTM）的循环神经网络（Recurrent Neural Networks，RNN）。LSTM-RNN通过隐层的循环反馈连接，可以将历史信息存储在隐层的节点中，从而可以有效地利用语音信号的长时相关性。

进一步地通过使用双向循环神经网络（BidirectionalRNN），可以有效地利用语音信号历史以及未来的信息，更有利于语音的声学建模。基于循环神经网络的语音声学模型相比于前馈全连接神经网络可以获得显著的性能提升。但是循环神经网络相比于前馈全连接神经网络模型更加复杂，往往包含更多的参数，这会导致模型的训练以及测试都需要更多的计算资源。
另外基于双向循环神经网络的语音声学模型，会面临很大的时延问题，对于实时的语音识别任务不适用。现有的一些改进的模型，例如，基于时延可控的双向长短时记忆单元（Latency Controlled LSTM，LCBLSTM ）[1-2]，以及前馈序列记忆神经网络（Feedforward SequentialMemory Networks，FSMN）[3-5]。

FSMN是近期被提出的一种网络结构，通过在FNN的隐层添加一些可学习的记忆模块，从而可以有效地对语音的长时相关性进行建模。FSMN相比于LCBLSTM不仅可以更加方便地控制时延，而且也能获得更好的性能，需要的计算资源也更少。但是标准的FSMN很难训练非常深的结构，会由于梯度消失问题导致训练效果不好。而深层结构的模型目前在很多领域被证明具有更强的建模能力。因而针对此我们提出了一种改进的FSMN模型，称之为深层的FSMN（DeepFSMN, DFSMN）。进一步地我们结合LFR（lowframe rate）技术构建了一种高效的实时语音识别声学模型，相比于去年我们上线的LCBLSTM声学模型可以获得超过20%的相对性能提升，同时可以获得2-3倍的训练以及解码的加速，可以显著地减少我们的系统实际应用时所需要的计算资源。



DFSMN 特点：跳层连接，更深的层数。和LFR结合。模型尺寸更小，低延迟。
实验结果表明DFSMN是用于声学模型的BLSTM强有力替代方案。

[参考](https://blog.csdn.net/zhanaolu4821/article/details/88977782)

Kaldi 是一个开源的语音识别工具库，隶属于 Apache 基金会，主要由 Daniel Povey 开发和维护。Kaldi 内置功能强大，支持 GMM-HMM、SGMM-HMM、DNN-HMM 等多种语音识别模型的训练和预测。随着深度学习的影响越来越大，Kaldi 目前对 DNN、CNN、LSTM 以及 Bidirectional-LSTM 等神经网络结构均提供模型训练支持。

[ Kaldi ](https://github.com/kaldi-asr/kaldi)

[ DFSMN ](https://github.com/alibaba/Alibaba-MIT-Speech)

配置：

git clone https://github.com/kaldi-asr/kaldi.git kaldi-trunk --origin golden


cd kaldi-trunk/  && git clone https://github.com/alibaba/Alibaba-MIT-Speech

将补丁加载到 Kaldi 分支 git apply --stat Alibaba-MIT-Speech/Alibaba_MIT_Speech_DFSMN.patch
 
测试补丁：git apply --check Alibaba-MIT-Speech/Alibaba_MIT_Speech_DFSMN.patch

添加 Git 账户邮箱和用户名，否则无法应用补丁。

    git config --global user.email "userEmail"
    git config --global user.name "username"
    
应用补丁：git am --signoff < Alibaba-MIT-Speech/Alibaba_MIT_Speech_DFSMN.patch

>  安装 Kaldi

切换到 tools 目录中，自动检测并安装缺少的依赖包，直到出现 all OK 为止。

extras/check_dependencies.sh

编译：make -j6  编译 –j 参数表示内核数，根据自己环境设定运用多少内核工作。

切换到 src 目录下，进行安装。

    cd ../src
    ./configure –shared
    make depend -j6
    
自动安装其它扩展包，执行以下命令：make ext

运行自带的 demo：

    cd ../egs/yesno/s5/
    ./run.sh
### 语音特征提取 MFCC 
MFCC（MeI-Freguency CeptraI Coefficients）是语音特征参数提取方法之一，因其独特的基于倒谱的提取方式，更加符合人类的听觉原理，因而也是最为普遍、最有效的语音特征提取算法。通过 MFCC，我们可以有效地区分出不同的人声，识别不同的说话人。

    预加重 --> 分帧 --> 加窗  --> FFT 离散傅立叶变换（DFT）  --> Mel滤波数组 --> 对数运算 -->  DCT
    
 1. 预加重其实就是将语音信号通过一个高通滤波器，来增强语音信号中的高频部分，并保持在低频到高频的整个频段中，能够使用同样的信噪比求频谱。
 
 2. 分帧是指在给定的音频样本文件中，按照某一个固定的时间长度分割，分割后的每一片样本，称之为一帧。
 
 分帧是先将 N 个采样点集合成一个观测单位，也就是分割后的帧。通常情况下 N 的取值为 512 或 256，涵盖的时间约为 20~30ms。N 值和窗口间隔可动态调整。为避免相邻两帧的变化过大，会让两相邻帧之间有一段重叠区域，此重叠区域包含了 M 个取样点，一般 M 的值约为 N 的 1/2 或 1/3。
语音识别中所采用的信号采样频率一般为 8kHz 或 16kHz。以 8kHz 来说，若帧长度为 256 个采样点，则对应的时间长度是 256/8000*1000=32ms。本次测试中所使用的采样率为 16kHz，窗长 37.5ms（600 个采样点），窗间隔为 10ms（160 个采样点）。

 3. 加窗,在对音频进行分帧之后，需要对每一帧进行加窗，以增加帧左端和右端的连续性，减少频谱泄漏。比较常用的窗口函数为 Hamming 窗。
 
 4. 离散傅立叶变换（DFT）
 
 由于信号在时域上的变换通常很难看出信号的特性，所以通常将它转换为频域上的能量分布来观察，不同的能量分布，代表不同语音的特性。所以在进行了加窗处理后，还需要再经过离散傅里叶变换以得到频谱上的能量分布。对分帧加窗后的各帧信号进行快速傅里叶变换 FFT 得到各帧的频谱。并对语音信号的频谱取模平方得到语音信号的功率谱。
 
 5. Mel 滤波器组
 
 MFCC 考虑人类的听觉特征，先将线性频谱映射到基于听觉感知的 Mel 非线性频谱中，然后转换到倒谱上。在 Mel 频域内，人对音调的感知度为线性关系。举例来说，如果两段语音的 Mel 频率相差两倍，则人耳听起来两者的音调也相差两倍。Mel 滤波器的本质其实是一个尺度规则：通常是将能量通过一组 Mel 尺度的三角形滤波器组，如定义有 MM 个滤波器的滤波器组，采用的滤波器为三角滤波器，中心频率为 f(m),m=1,2…Mf(m),m=1,2…M，MM 通常取 22~26。f(m)f(m)之间的间隔随着 mm 值的减小而缩小，随着 mm 值的增大而增宽.
 
 6. 对频谱进行离散余弦变换（DCT）

使⽤离散余弦变换，进⾏⼀个傅⽴叶变换的逆变换，得到倒谱系数。 由此可以得到 26 个倒谱系数。只取其 [2:13] 个系数，第 1 个用能量的对数替代，这 13 个值即为所需的 13 个 MFCC 倒谱系数。

 动态差分参数的提取（包括一阶差分和二阶差分
 
标准的倒谱参数 MFCC 只反映了语音参数的静态特性，语音的动态特性可以用这些静态特征的差分谱来描述。实验证明：把动、静态特征结合起来才能有效提高系统的识别性能。
 
### CTC(Connectionist Temporal Classifier)
    一般译为联结主义时间分类器 ，
    适合于输入特征和输出标签之间对齐关系不确定的时间序列问题，
    CTC可以自动端到端地同时优化模型参数和对齐切分的边界。
    
[LSTM-CTC 博客详解](https://blog.csdn.net/laolu1573/article/details/78975419)

[Theano implementation of LSTM and CTC to recognize simple english sentence image ](https://github.com/Ewenwan/cnn-lstm-ctc)

## 语义理解NLU 
[图灵NLU 在线语意理解 ](https://github.com/Ewenwan/Ros/blob/master/src/voice_system/src/tl_nlu.cpp)

## 文字转语音TTS
[科大讯飞 TTS](https://github.com/Ewenwan/Ros/blob/master/src/voice_system/src/xf_tts.cpp)

## 文字识别ocr 其实属于图像识别问题了
* CRNN  
[An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition](https://arxiv.org/abs/1507.05717)
[中文版](http://noahsnail.com/2017/08/21/2017-8-21-CRNN%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E6%96%87%E7%89%88/)
[中英文对照](http://noahsnail.com/2017/08/21/2017-8-21-CRNN%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E8%8B%B1%E6%96%87%E5%AF%B9%E7%85%A7/)

[ocn代码示例](https://github.com/fengbingchun/OCR_Test)

[Use CTC + tensorflow to OCR ](https://github.com/ilovin/lstm_ctc_ocr)

* CTPN  
[Detecting Text in Natural Image with Connectionist Text Proposal Network](https://arxiv.org/abs/1609.03605)
[中文版](http://noahsnail.com/2018/02/02/2018-02-02-Detecting%20Text%20in%20Natural%20Image%20with%20Connectionist%20Text%20Proposal%20Network%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E6%96%87%E7%89%88/)
[中英文对照](http://noahsnail.com/2018/02/02/2018-02-02-Detecting%20Text%20in%20Natural%20Image%20with%20Connectionist%20Text%20Proposal%20Network%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E8%8B%B1%E6%96%87%E5%AF%B9%E7%85%A7/)

## 自然语言处理
[自然语言处理算法与实战](https://github.com/Ewenwan/learning-nlp)

    chapter-3 中文分词技术
    chapter-4 词性标注与命名实体识别
    chapter-5 关键词提取
    chapter-6 句法分析
    chapter-7 文本向量化
    chapter-8 情感分析
    chapter-9 NLP中用到的机器学习算法
    chapter-10 基于深度学习的NLP算法

