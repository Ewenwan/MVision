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
    
# 编译
    进入到/home/xxxxxx/libsvm-3.20。输入命令
    make 
    
# 4.执行libsvm
    下来解释一下libsvm的程序怎么用.你能够先拿libsvm 附的heart_scale来做输入，
    底下也以它为例,看到这里你应该也了解使用 SVM 的流程大概就是：

    1.准备数据并做成指定格式 (有必要时需 svmscale)
    2.用svmtrain来训练成 model
    3.对新的输入，使用 svmpredic来预測新数据的类别
    
# 4.1 svm-train 寻训练
    svmtrain 的语法大致就是:
    svm-train [options] training_set_file [model_file]
    
    training_set_file 就是之前的格式，而 model_file 
    假设不给就会叫[training_set_file].model.options 能够先不要给。

    下列程序执行結果会产生 heart_scale.model 文件：(屏幕输出不是非常重要，沒有错误就好了) 
    执行代码：

    ./svm-train heart_scale
    输出结果

    ====================== 
    optimization finished, #iter = 219 
    nu = 0.431030 
    obj = -100.877286, rho = 0.424632 
    nSV = 132, nBSV = 107 
    Total nSV = 132

    ======================
    
# 5.2 svm-predict 预测
svmpredict 的语法是 :

svm-predict  test_file model_file  output_file

(1)test_file就是我们要预測的数据,它的格式svmtrain的输入，
    也就是training_set_file是一样的,只是每行最前面的label能够省略(由于预測就是要预測那个label)。
    但假设test_file有label的值的话，predict完会顺便拿predict出来的值跟test_file里面写的值去做比对，
    这代表：test_file写的label是真正的分类结果拿来跟我们预測的结果比对就能够知道预測的效果。
    所以我们能够拿原training set当做test_file再丟给svm-predict去预測(由于格式一样),看看正确率有多高，
    方便后面调參数.其他參数就非常好理解了

(2)model_file就是svm-train出来的文件;
    heart_scale.out： 
    执行代码

    ./svm-predict heart_scale heart_scale.model heart_scale.out
    
    得到输出：

    ===================================== 
    Accuracy = 86.6667% (234/270) (classification) 
    Mean squared error = 0.533333 (regression) 
    Squared correlation coefficient = 0.532639(regression)

    =====================================
# fv 编码后的特征分类实例


