# vgg16-ssd-voc-300

ssd-voc-300  ristretto  量化  目标检测网络量化

## 1. 对原网络执行量化，迭代不同零花组合策略，分别对网络进行测试，选取测试结果最好的 量化策略
./00_quantize_vgg16-ssd-net-voc.sh

```sh
I0716 19:19:21.624325  6251 net.cpp:219] data_data_0_split does not need backward computation.
I0716 19:19:21.624333  6251 net.cpp:219] data does not need backward computation.
I0716 19:19:21.624341  6251 net.cpp:261] This network produces output detection_eval
I0716 19:19:21.624428  6251 net.cpp:274] Network initialization done.
I0716 19:19:21.696208  6251 net.cpp:796] Ignoring source layer mbox_loss
I0716 19:22:33.964257  6251 quantization.cpp:167] Test loss: 0
I0716 19:22:34.120753  6251 quantization.cpp:217] accuracy : 0.759119
I0716 19:22:35.176486  6251 quantization.cpp:421] ------------------------------
I0716 19:22:35.176540  6251 quantization.cpp:422] Network accuracy analysis for
I0716 19:22:35.176549  6251 quantization.cpp:423] Convolutional (CONV) and fully
I0716 19:22:35.176558  6251 quantization.cpp:424] connected (FC) layers.
I0716 19:22:35.176564  6251 quantization.cpp:425] Baseline 32bit float: 0.766034
I0716 19:22:35.176581  6251 quantization.cpp:426] Dynamic fixed point CONV
I0716 19:22:35.176589  6251 quantization.cpp:427] weights: 
I0716 19:22:35.176594  6251 quantization.cpp:429] 16bit: 	0.759711
I0716 19:22:35.176605  6251 quantization.cpp:429] 8bit: 	0.758266
I0716 19:22:35.176618  6251 quantization.cpp:429] 4bit: 	0.000181086
I0716 19:22:35.176628  6251 quantization.cpp:429] 2bit: 	0.000398298
I0716 19:22:35.176640  6251 quantization.cpp:429] 1bit: 	4.114e-07
I0716 19:22:35.176650  6251 quantization.cpp:432] Dynamic fixed point FC
I0716 19:22:35.176656  6251 quantization.cpp:433] weights: 
I0716 19:22:35.176663  6251 quantization.cpp:435] 16bit: 	0.766034
I0716 19:22:35.176673  6251 quantization.cpp:435] 8bit: 	0.766034
I0716 19:22:35.176683  6251 quantization.cpp:435] 4bit: 	0.766034
I0716 19:22:35.176695  6251 quantization.cpp:435] 2bit: 	0.766034
I0716 19:22:35.176707  6251 quantization.cpp:435] 1bit: 	0.766034
I0716 19:22:35.176717  6251 quantization.cpp:437] Dynamic fixed point layer
I0716 19:22:35.176725  6251 quantization.cpp:438] activations:
I0716 19:22:35.176733  6251 quantization.cpp:440] 16bit: 	0.762796
I0716 19:22:35.176743  6251 quantization.cpp:440] 8bit: 	0.761558
I0716 19:22:35.176753  6251 quantization.cpp:440] 4bit: 	0.533742
I0716 19:22:35.176765  6251 quantization.cpp:443] Dynamic fixed point net:
I0716 19:22:35.176772  6251 quantization.cpp:444] 8bit CONV weights,
I0716 19:22:35.176779  6251 quantization.cpp:445] 1bit FC weights,
I0716 19:22:35.176786  6251 quantization.cpp:446] 8bit layer activations:
I0716 19:22:35.176795  6251 quantization.cpp:447] Accuracy: 0.759119
I0716 19:22:35.176802  6251 quantization.cpp:448] Please fine-tune.





I0716 21:36:23.168421 44803 net.cpp:274] Network initialization done.
I0716 21:36:23.266587 44803 net.cpp:796] Ignoring source layer mbox_loss
I0716 21:36:23.267045 44803 quantization.cpp:217] Running for EvaluateDetection 2000 iterations.
I0716 21:39:33.380177 44803 quantization.cpp:274] Dection test loss: 0
I0716 21:39:33.504156 44803 quantization.cpp:324] accuracy : 0.758634mAP
I0716 21:39:34.502511 44803 quantization.cpp:524] ------------------------------
I0716 21:39:34.502555 44803 quantization.cpp:525] Network accuracy analysis for
I0716 21:39:34.502563 44803 quantization.cpp:526] Convolutional (CONV) and fully
I0716 21:39:34.502571 44803 quantization.cpp:527] connected (FC) layers.
I0716 21:39:34.502578 44803 quantization.cpp:528] Baseline 32bit float: 0.766034
I0716 21:39:34.502591 44803 quantization.cpp:529] Dynamic fixed point CONV
I0716 21:39:34.502599 44803 quantization.cpp:530] weights: 
I0716 21:39:34.502605 44803 quantization.cpp:532] 16bit: 	0.759711
I0716 21:39:34.502617 44803 quantization.cpp:532] 8bit: 	0.758266
I0716 21:39:34.502627 44803 quantization.cpp:532] 4bit: 	0.000181086
I0716 21:39:34.502638 44803 quantization.cpp:532] 2bit: 	0.000398298
I0716 21:39:34.502650 44803 quantization.cpp:532] 1bit: 	4.114e-07
I0716 21:39:34.502660 44803 quantization.cpp:535] Dynamic fixed point FC
I0716 21:39:34.502666 44803 quantization.cpp:536] weights: 
I0716 21:39:34.502673 44803 quantization.cpp:538] 16bit: 	0.766034
I0716 21:39:34.502686 44803 quantization.cpp:538] 8bit: 	0.766034
I0716 21:39:34.502698 44803 quantization.cpp:538] 4bit: 	0.766034
I0716 21:39:34.502712 44803 quantization.cpp:538] 2bit: 	0.766034
I0716 21:39:34.502723 44803 quantization.cpp:538] 1bit: 	0.766034
I0716 21:39:34.502735 44803 quantization.cpp:540] Dynamic fixed point layer
I0716 21:39:34.502743 44803 quantization.cpp:541] activations:
I0716 21:39:34.502751 44803 quantization.cpp:543] 16bit: 	0.763486
I0716 21:39:34.502782 44803 quantization.cpp:543] 8bit: 	0.762462
I0716 21:39:34.502799 44803 quantization.cpp:543] 4bit: 	0.56676
I0716 21:39:34.502810 44803 quantization.cpp:546] Dynamic fixed point net:
I0716 21:39:34.502817 44803 quantization.cpp:547] 8bit CONV weights,
I0716 21:39:34.502832 44803 quantization.cpp:548] 1bit FC weights,
I0716 21:39:34.502840 44803 quantization.cpp:549] 8bit layer activations:
I0716 21:39:34.502846 44803 quantization.cpp:550] Accuracy: 0.758634
I0716 21:39:34.502854 44803 quantization.cpp:551] Please fine-tune.

```


