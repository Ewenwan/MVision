# -*- coding: utf-8 -*-
# 该文件用来可视化 iou
# Region 82 Avg IOU: 0.872406, Class: 0.999075, Obj: 0.999050, No Obj: 0.016427, .5R: 1.000000, .75R: 1.000000,  count: 4
# 三个尺度索引  平均交并比     分类准确率       前景概率        背景           以IOU=0.5为阈值时候的recall  recall = 检出的正样本/实际的正样本  正样本数目      
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt#画图
#%matplotlib inline

lines =1878760
result = pd.read_csv('../voc_train_log_ext_iou.txt', skiprows=[x for x in range(lines) if ((x%10!=9) |(x<1000))] ,error_bad_lines=False, names=['Avg IOU', 'Class', 'Obj', 'No Obj', '.5R', '.7R', 'count'])
result.head()

result['Avg IOU']=result['Avg IOU'].str.split(': ').str.get(1)
result['Class']=result['Class'].str.split(': ').str.get(1)
result['Obj']=result['Obj'].str.split(': ').str.get(1)
result['No Obj']=result['No Obj'].str.split(': ').str.get(1)
result['.5R']=result['.5R'].str.split(': ').str.get(1)
result['.7R']=result['.7R'].str.split(': ').str.get(1)
result['count']=result['count'].str.split(': ').str.get(1)
result.head()
result.tail()

#print(result.head())
# print(result.tail())
# print(result.dtypes)
print(result['Avg IOU'])
print(result['.5R'])
print(result['.7R'])

# result['avg']= pd.to_numeric(result['avg']) 新版本  转成float类型
result['Avg IOU']=result['Avg IOU'].astype(float)
result['Class']=result['Class'].astype(float)
result['Obj']=result['Obj'].astype(float)
result['No Obj']=result['No Obj'].astype(float)
result['.5R']=result['.5R'].astype(float)
result['.7R']=result['.7R'].astype(float)
result['count']=result['count'].astype(float)
result.dtypes

fig = plt.figure()#图窗口
ax = fig.add_subplot(1, 1, 1)#3个子图
#ax.plot(result['Avg IOU'].values,label='Region Avg IOU')
#ax.plot(result['Class'].values,label='Class')
#ax.plot(result['Obj'].values,label='Obj')
#ax.plot(result['No Obj'].values,label='No Obj')
ax.plot(result['.5R'].values,label='.5R Avg Recall')
#ax.plot(result['.7R'].values,label='.7R Avg Recall')
#ax.plot(result['count'].values,label='count')

ax.legend(loc='best')
#ax.set_title('The Region Avg IOU curves')
ax.set_title('The Avg Recall curves')#图标题 ss曲线
ax.set_xlabel('batches')#x轴 显示 迭代次数
fig.savefig('Avg Recall')#保存
#fig.savefig('Avg IOU')
