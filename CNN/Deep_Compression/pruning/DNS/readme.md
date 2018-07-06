# Dynamic network surgery   修剪+拼接恢复
[NIPS’16论文链接](https://arxiv.org/pdf/1608.04493.pdf)

[代码链接](https://github.com/Ewenwan/Dynamic-Network-Surgery)

这篇文章也是关于参数的修剪，但是多了一个拼接的步骤，可以大幅度恢复修剪造成的精度损失，并且能有效的提升压缩率。


## Experimental Results

The authors conducted experiments on several models including LeNet-5, LeNet-300-100 and AlexNet. The experimental results can be summarized as follows:

| Model                   | Top-1 Error   | Parameters | Iterations | Compression    |
| ----------------------- | ------------- | ---------- | ---------- | -------------- |
| LeNet-5 reference       | 0.91%         | 431K       | 10K        |                |
| LeNet-5 pruned          | 0.91%         | 4.0K       | 16K        | 108$$\times$$  |
| LeNet-100-300 reference | 2.28%         | 267K       | 10K        |                |
| LeNet-100-300 pruned    | 1.99%         | 4.8K       | 25K        | 56$$\times$$   |
| AlexNet reference       | 43.42%/-      | 61M        | 450K       |                |
| AlexNet pruned          | 43.09%/19.99% | 3.45M      | 700K       | 17.7$$\times$$ |

More detail comparison with work of Han et. al. on AlexNet using single crop validation on ImageNet are shown as follows:

| Layer | Parameters | Remaining Parameters Rate of Han et. al.(%) | Remaining Parameters Rate(%) |
| ----- | ---------- | ---------------------------------------- | ---------------------------- |
| conv1 | 35K        | ~84%                                     | 53.8%                        |
| conv2 | 307K       | ~38%                                     | 40.6%                        |
| conv3 | 885K       | ~35%                                     | 29.0%                        |
| conv4 | 664K       | ~37%                                     | 32.3%                        |
| conv5 | 443K       | ~37%                                     | 32.5%                        |
| fc1   | 38M        | ~9%                                      | 3.7%                         |
| fc2   | 17M        | ~9%                                      | 6.6%                         |
| fc3   | 4M         | ~25%                                     | 4.6%                         |
| Total | 61M        | ~11%                                     | 5.7%                         |
