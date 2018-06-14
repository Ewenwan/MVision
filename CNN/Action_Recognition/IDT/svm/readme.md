# svm分类
# 对fv编码后的特征进行一对多svm分类器分类

[libsvm源码](https://github.com/Ewenwan/libsvm)

[libSVM源码分析](https://blog.csdn.net/cyningsun/article/details/8705648)

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

    其中，#iter为迭代次数，
    nu是你选择的核函数类型的参数，
    obj为SVM文件转换为的二次规划求解得到的最小值，
    rho为判决函数的偏置项b，
    nSV为标准支持向量个数(0<a[i]<c)，
    nBSV为边界上的支持向量个数(a[i]=c)，
    Total nSV为支持向量总个数
    对于两类来说，因为只有一个分类模型Total 
    nSV = nSV，但是对于多类，这个是各个分类模型的nSV之和）。

    在目录下，还可以看到产生了一个train.model文件，可以用记事本打开，记录了训练后的结果。
    
    svm_type c_svc//所选择的svm类型，默认为c_svc
    kernel_type rbf//训练采用的核函数类型，此处为RBF核
    gamma 0.0769231//RBF核的参数γ
    nr_class 2//类别数，此处为两分类问题
    total_sv 132//支持向量总个数
    rho 0.424462//判决函数的偏置项b
    label 1 -1//原始文件中的类别标识
    nr_sv 64 68//每个类的支持向量机的个数
    SV//以下为各个类的权系数及相应的支持向量

    1 1:0.166667 2:1 3:-0.333333 … 10:-0.903226 11:-1 12:-1 13:1
    0.5104832128985164 1:0.125 2:1 3:0.333333 … 10:-0.806452 12:-0.333333 13:0.5
    ...
    -1 1:-0.375 2:1 3:-0.333333…. 10:-1 11:-1 12:-1 13:1
    -1 1:0.166667 2:1 3:1 …. 10:-0.870968 12:-1 13:0.5
   
# svmtrain的用法

    vmtrain我们在前面已经接触过，他主要实现对训练数据集的训练，并可以获得SVM模型。
    用法： 
    svmtrain [options] training_set_file [model_file]
    其中，options为操作参数，可用的选项即表示的涵义如下所示:
    -s设置svm类型：
            0 – C-SVC
            1 – v-SVC
            2 – one-class-SVM
            3 –ε-SVR
            4 – n - SVR

    -t设置核函数类型，默认值为2
            0 --线性核：u'*v

            1 --多项式核：(g*u'*v+coef0)degree

            2 -- RBF核：exp(-γ*||u-v||2)

            3 -- sigmoid核：tanh(γ*u'*v+coef0)
    -d degree:设置多项式核中degree的值，默认为3
    -gγ:设置核函数中γ的值，默认为1/k，k为特征（或者说是属性）数；
    -r coef 0:设置核函数中的coef 0，默认值为0；
    -c cost：设置C-SVC、ε-SVR、n - SVR中从惩罚系数C，默认值为1；
    -n v：设置v-SVC、one-class-SVM与n - SVR中参数n，默认值0.5；
    -pε：设置v-SVR的损失函数中的e，默认值为0.1；
    -m cachesize：设置cache内存大小，以MB为单位，默认值为40；
    -eε：设置终止准则中的可容忍偏差，默认值为0.001；
    -h shrinking：是否使用启发式，可选值为0或1，默认值为1；
    -b概率估计：是否计算SVC或SVR的概率估计，可选值0或1，默认0；
    -wi weight：对各类样本的惩罚系数C加权，默认值为1；
    -v n：n折交叉验证模式；
    model_file：可选项，为要保存的结果文件，称为模型文件，以便在预测时使用。

    默认情况下，只需要给函数提供一个样本文件名就可以了，
    但为了能保存结果，还是要提供一个结果文件名，
    比如:test.model,则命令为：
        svmtrain test.txt test.model   

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
    
# svmpredict的用法

    svmpredict是根据训练获得的模型，对数据集合进行预测。

    用法：svmpredict [options] test_file model_file output_file

    其中，options为操作参数，可用的选项即表示的涵义如下所示:

         -b probability_estimates——
                       是否需要进行概率估计预测，可选值为0或者1，默认值为0。

         model_file —— 是由svmtrain产生的模型文件；

         test_file —— 是要进行预测的数据文件，格式也要符合libsvm格式，
                      即使不知道label的值，也要任意填一个，
                      svmpredict会在output_file中给出正确的label结果，
                      如果知道label的值，就会输出正确率；

         output_file ——是svmpredict的输出文件，表示预测的结果值。
    
# libSVM的数据格式

        Label 1:value 2:value ….

        Label：是类别的标识，比如上节train.model中提到的1 -1，
        你可以自己随意定，比如-10，0，15。当然，如果是回归，这是目标值，就要实事求是了。

        Value：就是要训练的数据，从分类的角度来说就是特征值，数据之间用空格隔开
        比如: -15 1:0.708 2:1056 3:-0.3333
        需要注意的是，如果特征值为0，特征冒号前面的(姑且称做序号)可以不连续。如：
        -15 1:0.708 3:-0.3333
        表明第2个特征值为0，从编程的角度来说，这样做可以减少内存的使用，
        并提高做矩阵内积时的运算速度。
        我们平时在matlab中产生的数据都是没有序号的常规矩阵，所以为了方便最好编一个程序进行转化。

# svmscale的用法

    svmscale是用来对原始样本进行缩放的，范围可以自己定，一般是[0,1]或[-1,1]。
    缩放的目的主要是:
    1）防止某个特征过大或过小，从而在训练中起的作用不平衡；
    2）为了计算速度。
       因为在核计算中，会用到内积运算或exp运算，不平衡的数据可能造成计算困难。

    用法：svmscale  [-l lower] [-u upper]
                    [-y y_lower y_upper]
                    [-s save_filename]
                    [-r restore_filename] filename

    其中，[]中都是可选项：
         -l：设定数据下限；lower：设定的数据下限值，缺省为-1
         -u：设定数据上限；upper：设定的数据上限值，缺省为 1
         -y：是否对目标值同时进行缩放；y_lower为下限值，y_upper为上限值；
         -s save_filename：表示将缩放的规则保存为文件save_filename；
         -r restore_filename：表示将按照已经存在的规则文件restore_filename进行缩放；
         filename：待缩放的数据文件，文件格式按照libsvm格式。

    默认情况下，只需要输入要缩放的文件名就可以了：比如(已经存在的文件为test.txt)

    svmscale test.txt

    这时，test.txt中的数据已经变成[-1,1]之间的数据了。
    但是，这样原来的数据就被覆盖了，为了让规划好的数据另存为其他的文件，
    我们用一个dos的重定向符 > 来另存为(假设为out.txt)：
          svmscale test.txt > out.txt

    运行后，我们就可以看到目录下多了一个out.txt文件，那就是规范后的数据。
    假如，我们想设定数据范围[0,1]，并把规则保存为test.range文件:

        svmscale –l 0 –u 1 –s test.range test.txt > out.txt

    这时，目录下又多了一个test.range文件，可以用记事本打开，
    下次就可以用-r test.range来载入了。


# fv 编码后的特征分类实例
    一对多 多分类svm
    10类分类
    同一类的 标签为1~10 其他类为-1( 一对所有（One-Versus-All OVA） )
    训练多个svm二类分类器
    
## 1、一对所有（One-Versus-All OVA） 
    给定m个类，需要训练m个二类分类器。其中的分类器 i 是将 i 类数据设置为类1（正类），
    其它所有m-1个i类以外的类共同设置为类2（负类），这样，针对每一个类都需要训练一个二类分类器，
    最后，我们一共有 m 个分类器。对于一个需要分类的数据 x，将使用投票的方式来确定x的类别。
    比如分类器 i 对数据 x 进行预测，如果获得的是正类结果，
    就说明用分类器 i 对 x 进行分类的结果是: x 属于 i 类，那么，类i获得一票。
    如果获得的是负类结果，那说明 x 属于 i 类以外的其他类，那么，除 i 以外的每个类都获得一票。
    最后统计得票最多的类，将是x的类属性。

## 2、所有对所有（All-Versus-All AVA） 
    给定m个类，对m个类中的每两个类都训练一个分类器，总共的二类分类器个数为 m(m-1)/2 .
    比如有三个类，1，2，3，那么需要有三个分类器，分别是针对：1和2类，1和3类，2和3类。
    对于一个需要分类的数据x,它需要经过所有分类器的预测，也同样使用投票的方式来决定x最终的类属性。
    但是，此方法与”一对所有”方法相比，需要的分类器较多，并且因为在分类预测时，
    可能存在多个类票数相同的情况，从而使得数据x属于多个类别，影响分类精度。 
    对于多分类在matlab中的实现来说，matlab自带的svm分类函数只能使用函数实现二分类，
    多分类问题不能直接解决，需要根据上面提到的多分类的方法，自己实现。
    虽然matlab自带的函数不能直接解决多酚类问题，但是我们可以应用libsvm工具包。
    libsvm工具包采用第二种“多对多”的方法来直接实现多分类，
    可以解决的分类问题（包括C- SVC、n - SVC ）、回归问题（包括e - SVR、n - SVR ）
    以及分布估计（one-class-SVM ）等，
    并提供了线性、多项式、径向基和S形函数四种常用的核函数供选择。

## fv编码分类 选项
    # svm_option='-s 0 -t 0 -q -w0 0.5 -w1 0.5 -c 100 -b 1'; 
    svm_option='-s 0 -t 0 -q -c 100 -b 1'; 
    -s 0 – C-SVC
    -t 0 --线性核：u'*v
    -wi weight：对各类样本的惩罚系数C加权，默认值为1；
    -c cost：设置C-SVC、ε-SVR、n-SVR中从惩罚系数C，默认值为1；
    -b概率估计：是否计算SVC或SVR的概率估计，可选值0或1，默认0；
    
    
    一对多svm 
    double(trainLabel==i) 可以得到标签：
    1   特征
    1   特征
    1   特征
    0  
    0
    0
    0
    0
    0
    0
    train训练：
    
        numLabels=max(trainLabel);//类别数量
        model = cell(numLabels,1);// 一对多分类器的话，就应该有类别数量个二分类器
        for i=1:numLabels
            model{i} = svmtrain(double(trainLabel==i), trainData, '-q -c 100 -t 0 -b 1');//训练每一个 一对多二分类器并保存
        end
    
    测试预测阶段：
        prob = zeros(size(testData,1),numLabels);// 每个视频有 类别个数个预测标签
        for i=1:numLabels//每一个模型
            [~,~,p] = svmpredict(double(testLabel==i), testData, model{i}, '-b 1');
            prob(:,i) = p(:,model{i}.Label==1);    %# probability of class==k 每一列是一个模型的预测输出
        end
        
        % predict the class with the highest probability
        [~,pred] = max(prob,[],2);
        acc = sum(pred == testLabel) ./ numel(testLabel);    %# accuracy
        
    多对对svm libsvmv本身就支持
    不过标签和上面的不一样
    1
    1
    1
    2
    2
    2
    2
    2
    3
    3
    4
    4
    5
    5
    5
    6
    6
    7
    7
    7
    8
    8
    8
    8
    9
    9
    9
    10
    10
    10

    
