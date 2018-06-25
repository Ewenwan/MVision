# 哈希函数两比特缩放量化 BWNH 
[论文](https://arxiv.org/pdf/1802.02733.pdf)

[博客解析](https://blog.csdn.net/ajj15120321/article/details/80571748)

![](http://file.elecfans.com/web1/M00/55/79/pIYBAFssV_WAE7dRAACHJnpcRMk945.png)

[保留内积哈希方法是沈老师团队在15年ICCV上提出的 Learning Binary Codes for Maximum Inner Product Search ](https://webpages.uncc.edu/~szhang16/paper/ICCV15_binary.pdf)

    通过Hashing方法做的网络权值二值化工作。
    第一个公式是我们最常用的哈希算法的公式(保留内积哈希方法是沈老师团队在15年ICCV上提出的)，其中S表示相似性，
    后面是两个哈希函数之间的内积。
    我们在神经网络做权值量化的时候采用第二个公式，
    第一项表示输出的feature map，其中X代表输入的feature map，W表示量化前的权值，
    第二项表示量化后输出的feature map，其中B相当于量化后的权值，
    通过第二个公式就将网络的量化转化成类似第一个公式的Hashing方式。
    通过最后一行的定义，就可以用Hashing的方法来求解Binary约束。
    
    本文在二值化权重(BWN)方面做出了创新，发表在AAAI2018上，作者是自动化所程建团队。
    本文的主要贡献是提出了一个新的训练BWN的方法，
    揭示了哈希与BW(Binary Weights)之间的关联，表明训练BWN的方法在本质上可以当做一个哈希问题。
    基于这个方法，本文还提出了一种交替更新的方法来有效的学习hash codes而不是直接学习Weights。
    在小数据和大数据集上表现的比之前的方法要好。
    
    为了减轻用哈希方法所带来的loss，
    本文将binary codes乘以了一个scaling factor并用交替优化的策略来更新binary codes以及factor.
    
