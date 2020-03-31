# Speech Automatic Speech Recognition,(ASR)

语音识别，通俗来讲，就是将一段语音信号转换成对应的文本信息。具体来说，语音识别是从一段连续声波中采样，将每个采样值量化；然后对量化的采样音频进行分帧，对于每一帧，抽取出一个描述频谱内容的特征向量；最后根据语音信号的特征识别语音所代表的单词。

    包含：

        语言识别ASR 
        语义理解ALU 
        文字转语言TTS  
        文字识别ocr 等
[语音识别 RNN LSTM HMM GMM CTC The official repository of the Eesen project](https://github.com/Ewenwan/eesen)

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

