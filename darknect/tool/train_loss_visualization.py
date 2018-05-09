# -*- coding: utf-8 -*-
# 该文件用来可视化 loss
# 36501: 0.642504, 0.642504 avg, 0.008000 rate, 3.622329 seconds, 18688512 images
# 迭代次数 本次loss 平均loss      学习率        时间      图片数量
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt#画图
#%matplotlib inline

lines =1878760#提取的总行数
# 每一行按 标志提取对应的内容
result = pd.read_csv('../voc_train_log_ext_loss.txt', skiprows=[x for x in range(lines) if ((x%10!=9) |(x<1000))] ,error_bad_lines=False, names=['loss', 'avg', 'rate', 'seconds', 'images'])
result.head()

result['loss']=result['loss'].str.split(' ').str.get(1)
result['avg']=result['avg'].str.split(' ').str.get(1)
result['rate']=result['rate'].str.split(' ').str.get(1)
result['seconds']=result['seconds'].str.split(' ').str.get(1)
result['images']=result['images'].str.split(' ').str.get(1)
result.head()
result.tail()

# print(result.head())
# print(result.tail())
# print(result.dtypes)

print(result['loss'])
print(result['avg'])
print(result['rate'])
print(result['seconds'])
print(result['images'])
# result['avg']= pd.to_numeric(result['avg']) 新版本 转成float类型
result['loss']=result['loss'].astype(float)
result['avg']=result['avg'].astype(float)
result['rate']=result['rate'].astype(float)
result['seconds']=result['seconds'].astype(float)
result['images']=result['images'].astype(float)
result.dtypes

fig = plt.figure()#图窗口
ax = fig.add_subplot(1, 1, 1)#3个子图
ax.plot(result['avg'].values,label='avg_loss')#平均loss
#ax.plot(result['loss'].values,label='loss')
ax.legend(loc='best')
ax.set_title('The loss curves')#图标题 ss曲线
ax.set_xlabel('batches')#下轴 显示 迭代次数
fig.savefig('avg_loss')#保存
#fig.savefig('loss')
