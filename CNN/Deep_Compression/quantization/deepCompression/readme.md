
# 1. 深度神经网络压缩 Deep Compression
    为了进一步压缩网络，考虑让若干个权值共享同一个权值，
    这一需要存储的数据量也大大减少。
    在论文中，采用kmeans算法来将权值进行聚类，
    在每一个类中，所有的权值共享该类的聚类质心，
    因此最终存储的结果就是一个码书和索引表。
    
    1.对权值聚类 
        论文中采用kmeans聚类算法，
        通过优化所有类内元素到聚类中心的差距（within-cluster sum of squares ）来确定最终的聚类结果.
        
    2. 聚类中心初始化 

        常用的初始化方式包括3种： 
        a) 随机初始化。
           即从原始数据种随机产生k个观察值作为聚类中心。 

        b) 密度分布初始化。
           现将累计概率密度CDF的y值分布线性划分，
           然后根据每个划分点的y值找到与CDF曲线的交点，再找到该交点对应的x轴坐标，将其作为初始聚类中心。 

        c) 线性初始化。
            将原始数据的最小值到最大值之间的线性划分作为初始聚类中心。 

        三种初始化方式的示意图如下所示： 

![](https://img-blog.csdn.net/20161026183710142)

    由于大权值比小权值更重要（参加HanSong15年论文），
    而线性初始化方式则能更好地保留大权值中心，
    因此文中采用这一方式，
    后面的实验结果也验证了这个结论。 
    
    3. 前向反馈和后项传播 
        前向时需要将每个权值用其对应的聚类中心代替，
        后向计算每个类内的权值梯度，
        然后将其梯度和反传，
        用来更新聚类中心，
        如图： 
        
![](https://img-blog.csdn.net/20161026184233327)

        共享权值后，就可以用一个码书和对应的index来表征。
        假设原始权值用32bit浮点型表示，量化区间为256，
        即8bit，共有n个权值，量化后需要存储n个8bit索引和256个聚类中心值，
        则可以计算出压缩率compression ratio: 
            r = 32*n / (8*n + 256*32 )≈4 
            可以看出，如果采用8bit编码，则至少能达到4倍压缩率。

[通过减少精度的方法来优化网络的方法总结](https://arxiv.org/pdf/1703.09039.pdf)


 
# 降低数据数值范围。
        其实也可以算作量化
        默认情况下数据是单精度浮点数，占32位。
        有研究发现，改用半精度浮点数(16位)
        几乎不会影响性能。谷歌TPU使用8位整型来
        表示数据。极端情况是数值范围为二值
        或三值(0/1或-1/0/1)，
        这样仅用位运算即可快速完成所有计算，
        但如何对二值或三值网络进行训练是一个关键。
        通常做法是网络前馈过程为二值或三值，
        梯度更新过程为实数值。


# Deep Compression

![Pipeline of Deep Compression](./fig/Pipeline of Deep Compression.png)

In this paper the authors introduces "deep compression" to compress model size of deep convolutional neural networks. The proposed method consists of three stage: pruning, trained quantization and [Huffman coding](https://en.wikipedia.org/wiki/Huffman_coding). The authors first prunes weights by learning the important connections. Second, quantize weights to enforce weight sharing. Third, apply Huffman coding to reduce storage further. 

## Proposed Methods

[Network Pruning](http://papers.nips.cc/paper/5784-learning-both-weights-and-connections-for-efficient-neural-network.pdf): First, learn connectivity by normal network training. Second, remove weights below threshold. Third,  retrain the network to learn the final weights for remaining sparse connections. After pruning, the sparse structure is stored using compress sparse row (CSR) or compress sparse column (CSC).

Trained Quantization: use $$k$$-clusters with $$b$$-bits to represent $$n$$-connections of the network. The compression rate can be computed by:
$$
r=\frac{nb}{nlog_2(k)+kb} \tag{1}
$$
The authors use k-means algorithm to get $$k$$-clusters $$C=\{c_1, c_2,\dots,c_k\}$$ to represent  $$n$$ original weights $$W=\{w_1, w_2, \dots,w_n\}$$, and the optimized objective function is:
$$
\arg\min_C\sum_{i=1}^{k}\sum_{w\in c_i} |w-c_i|^2 \tag{2}
$$
As the initialization of k-means algorithm is quite important, the authors explored three initial methods including forgy (random), density-based, linear. The authors suggested that using linear initialization can get better result as it has better representation to the few large weights which are important to the networks.

Once the centroids of weights are decided by k-means clustering, the index of sharing weight table is stored for each connections and used when conducting forward or backward propagation. Gradients for the shared weights are computed by:
$$
\frac{\partial \mathcal{L}}{\partial C_k} = \sum_{i,j}\frac{\partial \mathcal{L}}{\partial W_{i,j}}\frac{\partial W_{i,j}}{\partial C_k} = \sum_{i,j}\frac{\partial{\mathcal L}}{\partial W_{i,j}}\mathbb{1}(I_{i,j}=k), \tag{3}
$$
where $$\mathcal{L}$$ indicates loss, $$W_{i,j}$$ indicates weight in the $$i$$-th column and $$j$$-th row, $$C_k$$ indicates the $$k$$-th centroid of the layer and $$\mathbb{1}(\cdot)$$ is an indicator function.

## Experimental Results

The authors conducted experiments on two data sets: on MNIST, they used LeNet-300-100 and LeNet-5 while used AlexNet and VGG-16 on ImageNet to evaluate the proposed methods. Results on MINIST are summarized as follows:

| Model                    | Top-1 (/ Top-5 Error) | Accuracy Loss | Parameters | Compression Rate |
| ------------------------ | --------------------- | ------------- | ---------- | ---------------- |
| LeNet-300-100 reference  | 1.64%                 | -             | 1070 KB    | -                |
| LeNet-300-100 compressed | 1.58%                 | 0.06%         | 27 KB      | 40 $$\times$$    |
| LeNet-5 reference        | 0.80%                 | -             | 1720 KB    | -                |
| LeNet-5 compressed       | 0.74%                 | 0.06%         | 44 KB      | 39 $$\times$$    |
| AlexNet reference        | 42.78% / 19.73%       | -             | 240 MB     | -                |
| AlexNet compressed       | 42.78% / 19.70%       | 0% / 0.03%    | 6.9 MB     | 35 $$\times$$    |
| VGG-16 reference         | 31.50% / 11.32%       | -             | 552 MB     | -                |
| VGG-16 compressed        | 31.17% / 10.91%       | 0.33% / 0.41% | 11.3 MB    | 49 $$\times$$    |
