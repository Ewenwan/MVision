# svm分类
# 对fv编码后的特征进行一对多svm分类器分类

[libsvm源码](https://github.com/Ewenwan/libsvm)

# 简介
    LibSVM是台湾林智仁(Chih-Jen Lin)教授2001年开发的一套支持向量机的库，
    这套库运算速度还是挺快的，可以很方便的对数据做分类或回归。由于libSVM程序小，
    运用灵活，输入参数少，并且是开源的，易于扩展，因此成为目前国内应用最多的SVM的库。
    
# 文件夹简介
    Java——    主要是应用于java平台；
    Python——  是用来参数优选的工具，稍后介绍；
    svm-toy——一个可视化的工具，用来展示训练数据和分类界面，里面是源码，其编译后的程序在windows文件夹下；
    tools——  主要包含四个python文件，用来数据集抽样(subset)，参数优选（grid），集成测试(easy),数据检查(checkdata)；
    windows——包含libSVM四个exe程序包，我们所用的库就是他们，里面还有个heart_scale，
             是一个样本文件，可以用记事本打开，用来测试用的。
    其他.h和.cpp文件都是程序的源码，可以编译出相应的.exe文件。
    其中，最重要的是svm.h和svm.cpp文件，svm-predict.c、svm-scale.c和svm-train.c（还有一个svm-toy.c在svm-toy文件夹中）
    都是调用的这个文件中的接口函数，编译后就是windows下相应的四个exe程序。
    另外，里面的 README 跟 FAQ也是很好的文件，对于初学者如果E文过得去，可以看一下。
    
    下面以svm-train为例，简单的介绍下，
    怎么编译：（这步很简单，也没必要，对于仅仅使用libsvm库的人来说，windows下的4个exe包已经足够了，
    之所以加这步，是为了那些做深入研究的人，可以按照自己的思路改变一下svm.cpp，然后编译验证）
    
    
