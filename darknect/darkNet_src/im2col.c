// 卷积 使用 展开矩阵 矩阵相乘方式 实现
// 众所周知caffe中卷积采用的是im2col和sgemm的方式。
#include "im2col.h"
#include <stdio.h>
float im2col_get_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}
// caffe中的数据是行优先（row-major）存储的。
// 参考 caffe的中卷积的实现 
//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu(
     float* data_im,// 输入图片数量
     int channels,  // 单张图片的 通道数量
	 int height,    // 图片高度
	 int width,     // 宽度
     int ksize,     // 卷积核尺寸
	 int stride,    // 步长
	 int pad,       // 填充
	 float* data_col)// 输出 展开
{
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;// 特征图展开  列长度 总行数
    for (c = 0; c < channels_col; ++c) {
		// caffe源码实现是一行一行来实现的
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
            }
        }
    }
}


// 单通道图像
/*
图像 ：

一个图像 input_num=1;
图像通道 input_channel=1;
图像高 input_h=4;
图像宽 input_w=4;

卷积核 ：
kernel高 kernel_h=3;
kernel宽 kernel_w=3;
stride=1；pad=0；

卷积后，输出图像的计算公式：
output_h=(input_h-kernel_h)/stride + 1;
output_w=(input_w-kernel_w)/stride + 1;

想要把卷积运算 展开成 两个矩阵相乘的方式
有卷积方式 可知， 在特征图对应卷积和大小的矩阵 和 卷积核对应元素想乘后相加

所以将 特征图中 对应 卷积核尺寸大小的矩阵 展开成一列

如 特征图：

0  1  2  3 
4  5  6  7
8  9  10 11
12 13 14 15

在无填充的情况下 可以有4个3*3的矩阵
0  1  2 
4  5  6 
8  9  10

1  2  3 
5  6  7
9  10 11

4  5  6
8  9  10
12 13 14

5  6  7
9  10 11
13 14 15

对于的3*3矩阵位置的元素展开成一列(行优先)
0  1  4  5
1  2  5  6
2  3  6  7
4  5  8  9
5  6  9  10
6  7  10 11
8  9  12 13
9  10 13 14
10 11 14 15     9*4  9行(卷积核大小^2) 4列(划分的矩阵个数 (4-3+1)^2 )

参考
https://blog.csdn.net/mrhiuser/article/details/52672824

然后 将卷积核 展开成一个行向量 

1  2  3
4  5  6
7  8  9

>>> 行优先展开成一行
1  2  3 4  5  6 7  8  9
矩阵相乘

									0  1  4  5
									1  2  5  6
									2  3  6  7
1  2  3 4  5  6 7  8  9		*		4  5  8  9
									5  6  9  10
									6  7  10 11
									8  9  12 13
									9  10 13 14
									10 11 14 15 
得到四个值
*/

// 多通道的展开
/*
假设有三个通道（R、G、B）图像通道 input_channel=3;

图像在内存中的存储是：
首先是连续存储第一通道的数据，
然后再存储第二通道的数据，
最后存储第三通道的数据。

按上述 方式分别展开 各通道
然后 按行 叠加在后面
 27行 * 4列

 3通道的 卷积核 也按通道分别 展开 得到 1*27 的行向量
*/
