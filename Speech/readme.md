# Speech Automatic Speech Recognition,(ASR)
    包含：

        语言识别ASR 
        语义理解ALU 
        文字转语言TTS  
        文字识别ocr 等

## 语言识别ASR 
![](http://antkillerfarm.github.io/images/img2/speech.png)

![](http://antkillerfarm.github.io/images/img2/speech_2.png)


[语音识别（一）——概述 HMM -> GMM -> 深度学习RNN  HTK CMU-Sphinx SPTK ](http://antkillerfarm.github.io/graphics/2018/04/16/speech.html)

[语音识别（二）——基本框架, Microphone Array, 声源定位 信号处理和特征提取 MFCC、声学模型(gmm-hmm)、语言模型（Language Model, LM）和解码器(Decoder)](http://antkillerfarm.github.io/graphics/2018/04/17/speech_2.html)()

[语音识别（三）——声源定位、前端处理 语言模型 声学模型, 解码器技术](http://antkillerfarm.github.io/graphics/2018/04/23/speech_3.html)

[语音识别（四）——声音分割，DTW(时域,Dynamic Time Warping动态时间规整算法), Spectrogram(频域,FFT傅里叶变换，声谱图), Cepstrum Analysis, Mel-Frequency Analysis](http://antkillerfarm.github.io/graphics/2018/06/01/speech_4.html)

[语音识别（五）——FBank, 语音识别的评价指标, 声学模型进阶, 语言模型进阶, GMM-HMM高斯混合-隐马尔科夫模型](http://antkillerfarm.github.io/graphics/2018/06/06/speech_5.html)

[中文分词!!!!!!!](https://github.com/Ewenwan/cppjieba)



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

