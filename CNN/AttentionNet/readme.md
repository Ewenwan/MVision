# Attention Net Attention Model（注意力模型）

[台大李宏毅老师 Machine Learning, Deep Learning and Structured Learning 包含RNN Attention模块](http://speech.ee.ntu.edu.tw/~tlkagk/courses_MLSD15_2.html)

[浅谈Attention-based Model【原理篇】 上述课程 部分笔记](https://blog.csdn.net/u010159842/article/details/80473462)

[完全图解RNN、RNN变体、Seq2Seq、Attention机制-知乎](https://zhuanlan.zhihu.com/p/28054589)

[Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)

Attention即为注意力，人脑在对于的不同部分的注意力是不同的。需要attention的原因是非常直观的，比如，我们期末考试的时候，我们需要老师划重点，划重点的目的就是为了尽量将我们的attention放在这部分的内容上，以期用最少的付出获取尽可能高的分数；再比如我们到一个新的班级，吸引我们attention的是不是颜值比较高的人？普通的模型可以看成所有部分的attention都是一样的，而这里的attention-based model对于不同的部分，重要的程度则不同。

Attention-based Model其实就是一个相似性的度量，当前的输入与目标状态越相似，那么在当前的输入的权重就会越大，说明当前的输出越依赖于当前的输入。严格来说，Attention并算不上是一种新的model，而仅仅是在以往的模型中加入attention的思想，所以Attention-based Model或者Attention Mechanism是比较合理的叫法，而非Attention Model。

Attention Mechanism可以帮助模型对输入的X每个部分赋予不同的权重，抽取出更加关键及重要的信息，使模型做出更加准确的判断，同时不会对模型的计算和存储带来更大的开销，这也是Attention Mechanism应用如此广泛的原因。

> 从Attention的作用角度出发，Attention分为两类： 

* **1.空间注意力 Spatial Attention，同一时期不同部分的关联**
* **2.时间注意力 Temporal Attention，不同时期内容的关联**

这样的分类更多的是从应用层面上，而从 Attention的作用方法上，可以将其分为 Soft Attention 和 Hard Attention，这既我们所说的， Attention输出的向量分布是一种one-hot的独热分布还是soft的软分布，这直接影响对于上下文信息的选择作用。

> CNN with Attention

主要分为两种，一种是spatial attention, 另外一种是channel attention。 
CNN每一层都会输出一个C x H x W的特征图，C就是通道，代表卷积核的数量，亦为特征的数量，H 和W就是原始图片经过压缩后的图，spatial attention就是对于所有的通道，在二维平面上，对H x W尺寸的图学习到一个权重，对每个像素都会学习到一个权重。你可以想象成一个像素是C维的一个向量，深度是C，在C个维度上，权重都是一样的，但是在平面上，权重不一样。这方面的论文已经很多了，重点关注一下image/video caption。相反的，channel attention就是对每个C，在channel维度上，学习到不同的权重，平面维度上权重相同。spatial 和 channel attention可以理解为关注图片的不同区域和关注图片的不同特征。channel attention写的最好的一篇论文个人感觉是SCA-CNN

> attention机制听起来高达上，其实就是学出一个权重分布，再拿这个权重分布施加在原来的特征之上，就可以叫做attention。简单来说： 

**（1）这个加权可以是保留所有分量均做加权（即soft attention）；也可以是在分布中以某种采样策略选取部分分量（即hard attention）。**

**（2）这个加权可以作用在空间尺度上，给不同空间区域加权；也可以作用在channel尺度上，给不同通道特征加权；甚至特征图上每个元素加权。 **

**（3）这个加权还可以作用在不同时刻历史特征上，如Machine Translation，以及我前段时间做的视频相关的工作。**


深度学习里的Attention model其实模拟的是人脑的注意力模型，举个例子来说，当我们观赏一幅画时，虽然我们可以看到整幅画的全貌，但是在我们深入仔细地观察时，其实眼睛聚焦的就只有很小的一块，这个时候人的大脑主要关注在这一小块图案上，也就是说这个时候人脑对整幅图的关注并不是均衡的，是有一定的权重区分的。这就是深度学习里的Attention Model的核心思想。

Attention模型最初应用于图像识别，模仿人看图像时，目光的焦点在不同的物体上移动。当神经网络对图像或语言进行识别时，每次集中于部分特征上，识别更加准确。如何衡量特征的重要性呢？最直观的方法就是权重，因此，Attention模型的结果就是在每次识别时，首先计算每个特征的权值，然后对特征进行加权求和，权值越大，该特征对当前识别的贡献就大。 

[RAM： Recurrent Models of Visual Attention 学习笔记](https://blog.csdn.net/c602273091/article/details/79059445)

RAM model讲得是视觉的注意力机制，说人识别一个东西的时候，如果比较大的话，是由局部构造出整体的概念。人的视觉注意力在选择局部区域的时候，是有一种很好的机制的，会往需要更少的步数和更能判断这个事物的方向进行的，我们把这个过程叫做Attention。由此，我们把这个机制引入AI领域。使用RNN这种可以进行sequential decision的模型引入，然后因为在选择action部分不可导，因为找到目标函数无法进行求导，只能进采样模拟期望，所以引入了reinforcment leanrning来得到policy进而选择action。

首先输入时一副完整的图片，一开始是没有action的，所以随机挑选一个patch，然后送入了RNN网络中，由RNN产生的输出作为action，这个action可以是hard attention，就是根据概率a~P(a|X)进行采样，或者是直接由概率最大的P(a|X)执行。有了action以后就可以从图片中选择某个位置的sub image送到RNN中作为input，另外一方面的input来自于上一个的hidden layer的输出。通过同样的网络经过T step之后，就进行classification，这里得到了最终的reward，（把calssification是否判断正确作为reward）就可以进行BPTT，同时也可以根据policy gradient的方法更新policy function。可以发现这个网络算是比较简单，也只有一个hidden layer，我觉得应该是加入了RL之后比较难训练。


将卷积神经网络应用于大型图像的计算量很大，因为计算量与图像像素数成线性关系。我们提出了一种新颖的循环神经网络模型，可以从图像或视频中提取信息，方法是自适应地选择一系列区域或位置，并仅以高分辨率处理选定区域。与卷积神经网络一样，所提出的模型具有内置的平移不变性程度，但其执行的计算量可以独立于输入图像大小进行控制。虽然模型是不可区分的，但可以使用强化学习方法来学习，以学习特定于任务的策略。

人类感知的一个重要特性是不倾向于一次处理整个场景。 相反，人类有选择地将注意力集中在视觉空间的某些部分上，以获取需要的信息，并随时间将不同视角的信息相结合，以建立场景的内部表示，指导未来的眼球运动和决策制定。 由于需要处理更少的“像素”，因此将场景中的部分计算资源集中在一起可节省“带宽”。 但它也大大降低了任务的复杂性，因为感兴趣的对象可以放置在固定的中心，固定区域之外的视觉环境（“杂乱”）的不相关特征自然被忽略。

该模型是一个循环神经网络（RNN），它按顺序处理输入，一次一个地处理图像（或视频帧）内的不同位置，并递增地组合来自这些注视的信息以建立场景的动态内部表示，或环境。基于过去的信息和任务的需求，模型不是一次处理整个图像甚至是边界框，而是在每一步中选择下一个要注意的位置。我们的模型中的参数数量和它执行的计算量可以独立于输入图像的大小来控制，而卷积网络的计算需与图像像素的数量线性地成比例。我们描述了一个端到端的优化程序，该程序允许模型直接针对给定的任务进行训练，并最大限度地提高可能取决于模型做出的整个决策序列的性能测量。该过程使用反向传播来训练神经网络组件和策略梯度以解决由于控制问题导致的非差异性。

我们表明，我们的模型可以有效的学习特定于任务的策略，如多图像分类任务以及动态视觉控制问题。 我们的结果还表明，基于关注的模型可能比卷积神经网络更好地处理杂波和大输入图像。

对于对象检测，已经做了很多工作来降低广泛的滑动窗口范例的成本，主要着眼于减少评估完整分类器的窗口的数量.

循环注意力模型 Attention rnn 

[参考](https://github.com/scutan90/DeepLearning-500-questions/blob/master/ch06_%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(RNN)/%E7%AC%AC%E5%85%AD%E7%AB%A0_%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(RNN).md)

> **1.单个神经元 基本的单层网络结构**

在进一步了解RNN之前，先给出最基本的单层网络结构，输入是x，经过变换Wx+b和激活函数f得到输出y：

![](https://github.com/Ewenwan/MVision/blob/master/CNN/AttentionNet/img/single_cnn.jpg)


> **2.图解经典RNN结构

在实际应用中，我们还会遇到很多序列形的数据，如：

a.自然语言处理问题,x1可以看做是第一个单词，x2可以看做是第二个单词，依次类推。
b.语音处理,此时，x1、x2、x3……是每帧的声音信号。
c.时间序列问题,例如每天的股票价格等等。

序列形的数据就不太好用原始的神经网络处理了。为了建模序列问题，RNN引入了隐状态h（hidden state）的概念，h可以对序列形的数据提取特征，接着再转换为输出。

为了便于理解，先从h1的计算开始看：

![](https://github.com/Ewenwan/MVision/blob/master/CNN/AttentionNet/img/rnn_1.jpg)

注：h0是初始隐藏状态，图中的圆圈表示向量，箭头表示对向量做变换。

h2的计算和h1类似。要注意的是，在计算时，每一步使用的参数U、W、b都是一样的，也就是说每个步骤的参数都是共享的，这是RNN的重要特点，一定要牢记。

![](https://github.com/Ewenwan/MVision/blob/master/CNN/AttentionNet/img/rnn_2.jpg)

依次计算剩下来的h3、h4也是类似的（使用相同的参数U、W、b，计算隐藏层共享参数 U W b）：

    h3 =f(U*h2 + W*x3 + b)
   
    h4 =f(U*h3 + W*x3 + b)

![](https://github.com/Ewenwan/MVision/blob/master/CNN/AttentionNet/img/rnn_3.jpg)

我们这里为了方便起见，只画出序列长度为4的情况，实际上，这个计算过程可以无限地持续下去。

我们目前的RNN还没有输出，得到输出值的方法就是直接通过对隐藏状态h进行类似 最开始x的计算方式：

采用 Softmax 作为激活函数,

    y1=Softmax(V*h1 + c)

![](https://github.com/Ewenwan/MVision/blob/master/CNN/AttentionNet/img/rnn_4.jpg)

剩下的输出类似进行（使用和y1同样的参数V和c 同样隐藏层解码也共享参数V和c）：

    y2 = Softmax(V*h2 + c)
    y3 = Softmax(V*h3 + c)
    y4 = Softmax(V*h4 + c)

![](https://github.com/Ewenwan/MVision/blob/master/CNN/AttentionNet/img/rnn_5.jpg)

这就是最经典的RNN结构，它的输入是x1, x2, .....xn，输出为y1, y2, ...yn，也就是说，输入和输出序列必须要是等长的。

由于这个限制的存在，经典RNN的适用范围比较小，但也有一些问题适合用经典的RNN结构建模，如：

计算视频中每一帧的分类标签。因为要对每一帧进行计算，因此输入和输出序列等长。
输入为字符，输出为下一个字符的概率。
这就是著名的[Char RNN](https://zhuanlan.zhihu.com/p/29212896)
可以用来生成文章，诗歌，甚至是代码，非常有意思）。

 > **3.vector-to-sequence结构 一入多出**
 
 有时我们要处理的问题输入是一个单独的值，输出是一个序列。此时，有两种主要建模方式：

方式一：可只在其中的某一个序列进行计算，比如序列第一个进行输入计算，其建模方式如下：

![](https://github.com/Ewenwan/MVision/blob/master/CNN/AttentionNet/img/rnn_1_n.jpg)

方式二：把输入信息X作为每个阶段的输入，其建模方式如下：

![](https://github.com/Ewenwan/MVision/blob/master/CNN/AttentionNet/img/rnn_1n_n.jpg)

这种 1 VS N 的结构可以处理的问题有：

a.从图像生成文字（image caption），此时输入的X就是图像的特征，而输出的y序列就是一段句子.

b.从类别生成语音或音乐等

 > **4.sequence-to-vector结构 多入一出**
 
 有时我们要处理的问题输入是一个序列，输出是一个单独的值，此时通常在最后的一个隐含状态h上进行输出变换，其建模如下所示：

![](https://github.com/Ewenwan/MVision/blob/master/CNN/AttentionNet/img/rnn_n_1.jpg)

这种结构通常用来处理序列分类问题。
如：
a.输入一段文字判别它所属的类别，
b.输入一个句子判断其情感倾向，
c.输入一段视频并判断它的类别等等。

![]()



![]()
