# -*- coding: utf-8 -*-
# 该文件用来提取训练log，去除不可解析的log后使log文件格式化，生成新的log文件供可视化工具绘图
"""
v2 训练log中各参数的意义:
1. Region Avg IOU：平均的IOU，代表预测的bounding box和ground truth
   的交集与并集之比，期望该值趋近于1。
2. Class：是标注物体的概率，期望该值趋近于1.
3. Obj：期望该值趋近于1.
4. No Obj：期望该值越来越小但不为零.
5. Avg Recall：期望该值趋近1
6. avg：平均损失，期望该值趋近于0
7. rate：当前学习率

v3 训练log中各参数的意义:
1. Region xx: cfg文件中yolo-layer的索引；
   分别表示三个尺度下的预测输出的 数据
   Region 106 Avg IOU: -nan, Class: -nan, Obj: -nan, No Obj: 0.000030, .5R: -nan, .75R: -nan,  count: 0
   Region 82 Avg IOU: 0.729025, Class: 0.584766, Obj: 0.477362, No Obj: 0.003746, .5R: 1.000000, .75R: 0.285714,  count: 7
   Region 94 Avg IOU: 0.387940, Class: 0.911466, Obj: 0.002695, No Obj: 0.000087, .5R: 0.000000, .75R: 0.000000,  count: 1
2. Avg IOU:当前迭代中，预测的box与标注的box的平均交并比，越大越好，期望数值为1；
3. Class: 标注物体的分类准确率，越大越好，期望数值为1；
4. obj: 越大越好，期望数值为1；
5. No obj: 越小越好；
6. .5R: 以IOU=0.5为阈值时候的recall; recall = 检出的正样本/实际的正样本
7. 0.75R: 以IOU=0.75为阈值时候的recall;
8. count:正样本数目。
示例数据：
Region 106 Avg IOU: -nan, Class: -nan, Obj: -nan, No Obj: 0.000088, .5R: -nan, .75R: -nan,  count: 0
Syncing... Done!
36532: 0.598229, 0.648933 avg, 0.008000 rate, 3.898034 seconds, 18704384 images    记录loss
Loaded: 0.000049 seconds
Resizing
384

"""

def extract_log(log_file,new_log_file,key_word):
    f = open(log_file)#打开txt文件
    train_log = open(new_log_file, 'w')#新建一个新文件
    for line in f:
        #每一行 images
        # 去除多gpu的同步log
        if 'Syncing' in line:
            continue
        # 去除 载入图片 
        if 'Loaded' in line:
            continue
            # 去除变换
        if 'Resizing' in line:
            continue        			
        # 去除零错误的log
        if 'nan' in line:
            continue
        if key_word in line:#每一行 的每一个关键字 ：前面的
            train_log.write(line)
    f.close()#关闭文件
    train_log.close()

extract_log('../voc_train_log_0.001.txt','../voc_train_log_ext_loss.txt','images')#包含images的一行有loss
extract_log('../voc_train_log_0.001.txt','../voc_train_log_ext_iou.txt','Region')#包含 Reigion的一行有 iou等
