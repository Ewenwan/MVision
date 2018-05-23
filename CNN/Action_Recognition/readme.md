# Video Analysis 相关领域 之Action Recognition(行为识别)
[论文总结参考](https://blog.csdn.net/whfshuaisi/article/details/79116265)
# 1. 任务特点及分析
## 目的
    给一个视频片段进行分类，类别通常是各类人的动作

## 特点
    简化了问题，一般使用的数据库都先将动作分割好了，一个视频片断中包含一段明确的动作，
    时间较短（几秒钟）且有唯一确定的label。
    所以也可以看作是输入为视频，输出为动作标签的多分类问题。
    此外，动作识别数据库中的动作一般都比较明确，周围的干扰也相对较少（不那么real-world）。
    有点像图像分析中的Image Classification任务。
## 难点/关键点
    强有力的特征：
        即如何在视频中提取出能更好的描述视频判断的特征。
        特征越强，模型的效果通常较好。
    特征的编码（encode）/融合（fusion）：
        这一部分包括两个方面，
        第一个方面是非时序的，在使用多种特征的时候如何编码/融合这些特征以获得更好的效果；
        另外一个方面是时序上的，由于视频很重要的一个特性就是其时序信息，
             一些动作看单帧的图像是无法判断的，只能通过时序上的变化判断，
             所以需要将时序上的特征进行编码或者融合，获得对于视频整体的描述。
    算法速度：
        虽然在发论文刷数据库的时候算法的速度并不是第一位的。
        但高效的算法更有可能应用到实际场景中去.
# 2. 常用数据库
    行为识别的数据库比较多，这里主要介绍两个最常用的数据库，也是近年这个方向的论文必做的数据库。
    1. UCF101:来源为YouTube视频，共计101类动作，13320段视频。
       共有5个大类的动作：
                1)人-物交互；
                2)肢体运动；
                3)人-人交互；
                4)弹奏乐器；
                5)运动。
[数据库主页](http://crcv.ucf.edu/data/UCF101.php)
                
    2. HMDB51:来源为YouTube视频，共计51类动作，约7000段视频。
      HMDB: a large human motion database
[数据库主页](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads)

    3. 在Actioin Recognition中，实际上还有一类骨架数据库，
    比如MSR Action 3D，HDM05，SBU Kinect Interaction Dataset等。
    这些数据库已经提取了每帧视频中人的骨架信息，基于骨架信息判断运动类型。 

# 3. 研究进展
## 3.1 传统方法
### 密集轨迹算法(DT算法) iDT（improved dense trajectories)特征
     ”Action recognition with improved trajectories”
[iDT算法](https://blog.csdn.net/wzmsltw/article/details/53023363) 
[iDT算法用法与代码解析](https://blog.csdn.net/wzmsltw/article/details/53221179)
[Stacked Fisher Vector 编码 基本原理 ](https://blog.csdn.net/wzmsltw/article/details/52050112)

    基本思路：
            DT算法的基本思路为利用光流场来获得视频序列中的一些轨迹，
            再沿着轨迹提取HOF，HOG，MBH，trajectory4种特征，其中HOF基于灰度图计算，
            另外几个均基于dense optical flow计算。
            最后利用FV（Fisher Vector）方法对特征进行编码，再基于编码结果训练SVM分类器。
            而iDT改进的地方在于它利用前后两帧视频之间的光流以及SURF关键点进行匹配，
            从而消除/减弱相机运动带来的影响，改进后的光流图像被成为warp optical flow




