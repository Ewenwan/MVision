# -*- coding: utf-8 -*-
# 该文件用来可视化 loss
# 36501: 0.642504, 0.642504 avg, 0.008000 rate, 3.622329 seconds, 18688512 images
# 迭代次数 本次loss 平均loss      学习率        时间      图片数量
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt#画图
#%matplotlib inline


import seaborn as sns  
sns.set_style("ticks")#背景白色格纹  
# sns.set(style="white", palette="muted", color_codes=True)     #set( )设置主题，调色板更常用 
'''
 darkgrid  黑色格纹
 whitegrid 白色格纹
 dark      黑色
 white     白色
 ticks     空白？
 默认： darkgrid
'''


lines =12000#提取的总行数
# 每一行按 标志提取对应的内容
#result = pd.read_csv('./voc_train_log_ext_loss2.txt', skiprows=[x for x in range(lines) if ((x%10==0))] ,error_bad_lines=False, names=['loss', 'avg', 'rate', 'seconds', 'images'])
result = pd.read_csv('./coco_train_log_ext_loss2.txt', skiprows=[x for x in range(lines) if ((x%100==0))] ,error_bad_lines=False, names=['loss', 'avg', 'rate', 'seconds', 'images'])

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
ax.legend(loc='best')# 图例
ax.set_title('The loss curves')#图标题 ss曲线
ax.set_xlabel('batches')#下轴 显示 迭代次数
fig.savefig('avg_loss_coco2')#保存
fig.show()
#fig.savefig('loss')

'''
1、设置图主题
sns.set(style="white", palette="muted", color_codes=True)     #set( )设置主题，调色板更常用 
 darkgrid  黑色格纹
 whitegrid 白色格纹
 dark      黑色
 white     白色
 ticks     空白？
 默认： darkgrid
2 、distplot( )  kdeplot( )
distplot( )为hist直方图加强版，kdeplot( )为密度曲线图 
df_iris = pd.read_csv('../input/iris.csv')  
fig, axes = plt.subplots(1,2)  
sns.distplot(df_iris['petal length'], ax = axes[0], kde = True, rug = True)  # kde 密度曲线  rug 边际毛毯  
sns.kdeplot(df_iris['petal length'], ax = axes[1], shade=True)               # shade  阴影                         
plt.show()  


3、 箱型图 boxplot()
tips = pd.read_csv('../input/tips.csv')  
sns.set(style="ticks")                                     #设置主题  
sns.boxplot(x="day", y="total_bill", hue="sex", data=tips, palette="PRGn")   #palette 调色板  
plt.show() 

4、 联合分布jointplot()
tips = pd.read_csv('../input/tips.csv')   #右上角显示相关系数  
sns.jointplot("total_bill", "tip", tips)  
# sns.jointplot("total_bill", "tip", tips, kind='reg')  
plt.show()  

5、热点图heatmap()
internal_chars = ['full_sq', 'life_sq', 'floor', 'max_floor', 'build_year', 'num_room', 'kitch_sq', 'state', 'price_doc']
corrmat = train[internal_chars].corr()

f, ax = plt.subplots(figsize=(10, 7))
plt.xticks(rotation='90')
sns.heatmap(corrmat, square=True, linewidths=.5, annot=True)
plt.show()

6、 散点图scatter( )
f, ax = plt.subplots(figsize=(10, 7))
plt.scatter(x=train['full_sq'], y=train['price_doc'], c='r')
plt.xlim(0,500)
plt.show()


7、pointplot画出变量间的关系

grouped_df = train_df.groupby('floor')['price_doc'].aggregate(np.median).reset_index()
plt.figure(figsize=(12,8))

sns.pointplot(grouped_df.floor.values, grouped_df.price_doc.values, alpha=0.8, color=color[2])

plt.ylabel('Median Price', fontsize=12)
plt.xlabel('Floor number', fontsize=12)
plt.xticks(rotation='vertical') plt.show()


8、pairplot( ) 散点 直方图 成对显示
import seaborn as sns  
import matplotlib.pyplot as plt  
iris = pd.read_csv('../input/iris.csv')  
sns.pairplot(iris, vars=["sepal width", "sepal length"],hue='class',palette="husl")    
plt.show()  


9、FacetGrid( )

tips = pd.read_csv('../input/tips.csv')  
g = sns.FacetGrid(tips, col="time",  row="smoker")  
g = g.map(plt.hist, "total_bill",  color="r")  
plt.show()  


10、barplot( )
f, ax=plt.subplots(figsize=(12,20))

#orient='h'表示是水平展示的，alpha表示颜色的深浅程度
sns.barplot(y=group_df.sub_area.values, x=group_df.price_doc.values,orient='h', alpha=0.8, color='red')

#设置y轴、X轴的坐标名字与字体大小
plt.ylabel('price_doc', fontsize=16)
plt.xlabel('sub_area', fontsize=16)

#设置X轴的各列下标字体是水平的
plt.xticks(rotation='horizontal')

#设置Y轴下标的字体大小
plt.yticks(fontsize=15)
plt.show()

注：如果orient='v'表示成竖直显示的话，
    一定要记得y=group_df.sub_area.values, x=group_df.price_doc.values调换一下坐标轴，否则报错


11、bar图
import matplotlib.pyplot as plt
import numpy as np
plt.rc('font', family='SimHei', size=13)

num = np.array([13325, 9403, 9227, 8651])
ratio = np.array([0.75, 0.76, 0.72, 0.75])
men = num * ratio
women = num * (1-ratio)
x = ['聊天','支付','团购\n优惠券','在线视频']

width = 0.5
idx = np.arange(len(x))
plt.bar(idx, men, width, color='red', label='男性用户')
plt.bar(idx, women, width, bottom=men, color='yellow', label='女性用户')  #这一块可是设置bottom,top，如果是水平放置的，可以设置right或者left。
plt.xlabel('应用类别')
plt.ylabel('男女分布')
plt.xticks(idx+width/2, x, rotation=40)

#bar图上显示数字

for a,b in zip(idx,men):

    plt.text(a, b+0.05, '%.0f' % b, ha='center', va= 'bottom',fontsize=12)
for a,b,c in zip(idx,women,men):
    plt.text(a, b+c+0.5, '%.0f' % b, ha='center', va= 'bottom',fontsize=12)

plt.legend()
plt.show()

'''
