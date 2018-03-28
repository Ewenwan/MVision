/*
离散傅里叶变换
说变换就是把一堆的数据变成另一堆的数据的方法。
对一张图像使用傅立叶变换就是将它分解成正弦和余弦两部分。

离散傅立叶变换的一个应用是决定图片中物体的几何方向.
比如，在文字识别中首先要搞清楚文字是不是水平排列的? 
看一些文字，你就会注意到文本行一般是水平的而字母则有些垂直分布。
文本段的这两个主要方向也是可以从傅立叶变换之后的图像看出来。
我们使用这个 水平文本图像 以及 旋转文本图像 来展示离散傅立叶变换的结果.



也就是将图像从空间域(spatial domain)转换到频域(frequency domain)。 
这一转换的理论基础来自于以下事实：任一函数都可以表示成无数个正弦和余弦函数的和的形式。
傅立叶变换就是一个用来将函数分解的工具

转换之后的频域值是复数， 因此，显示傅立叶变换之后的结果需要使用实数图像(real image) 
加虚数图像(complex image), 或者幅度图像(magitude image)加相位图像(phase image)。 
在实际的图像处理过程中，仅仅使用了幅度图像，因为幅度图像包含了原图像的几乎所有我们需要的几何信息。
 然而，如果你想通过修改幅度图像或者相位图像的方法来间接修改原空间图像，
你需要使用逆傅立叶变换得到修改后的空间图像，这样你就必须同时保留幅度图像和相位图像了.


*/
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>

using namespace cv;
using namespace std;

static void help(char* progName)
{
    cout << endl
        <<  "This program demonstrated the use of the discrete Fourier transform (DFT). " << endl
        <<  "The dft of an image is taken and it's power spectrum is displayed."          << endl
        <<  "Usage:"                                                                      << endl
        << progName << " [image_name -- default ../data/lena.jpg] "               << endl << endl;
}

int main(int argc, char ** argv)
{
    help(argv[0]);

    const char* filename = argc >=2 ? argv[1] : "../data/lena.jpg";
//===========读取图像=====================
    Mat I = imread(filename, IMREAD_GRAYSCALE);
    if (I.empty())
    {
        cout << "The image" << filename  << " could not be loaded." << endl;
        return -1;
    }


//==========扩展到最优尺寸
// 当图像的尺寸是2， 3，5的整数倍时，计算速度最快。 
// 因此，为了达到快速计算的目的，经常通过添凑新的边缘像素的方法获取最佳图像尺寸。

    Mat padded;                          //expand input image to optimal size
    int m = getOptimalDFTSize( I.rows );
    int n = getOptimalDFTSize( I.cols ); // on the border add zero values
    // 扩展出来的多的边框 0填充
    copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));

// 保存 dft变换的实部 和 虚部
// 傅立叶变换的结果是复数，这就是说对于每个原图像值，结果是两个图像值。 
// 此外，频域值范围远远超过空间值范围， 因此至少要将频域储存在 float 格式中。
// 结果我们将输入图像转换成浮点类型，并多加一个额外通道来储存复数部分:
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI); // 实际值 和 复数值 通道 合成为 complexI
    dft(complexI, complexI);    // this way the result may fit in the source matrix
    // compute the magnitude and switch to logarithmic scale
// => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
//===DFT变换后再分解成 实部 和 虚部======
    split(complexI, planes); //分割  planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
// 将复数转换为幅度. 复数包含实数部分(Re)和复数部分 (imaginary - Im)。 
// 离散傅立叶变换的结果是复数，对应的幅度可以表示为: sqrt(Re(DFT(I))^2 + Im(DFT(I))^2)
    magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
    Mat magI = planes[0];
// 对数尺度(logarithmic scale)缩放. 傅立叶变换的幅度值范围大到不适合在屏幕上显示。
    magI += Scalar::all(1);                    // switch to logarithmic scale
    log(magI, magI);

// 剪切和重分布幅度图象限. 还记得我们在第一步时延扩了图像吗? 那现在是时候将新添加的像素剔除了。
// crop the spectrum, if it has an odd number of rows or columns
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = magI.cols/2;
    int cy = magI.rows/2;

    Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

    Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);

// 归一化. 这一步的目的仍然是为了显示。
    normalize(magI, magI, 0, 1, NORM_MINMAX); // Transform the matrix with float values into a
                                            // viewable image form (float between values 0 and 1).

    imshow("Input Image"       , I   );    // Show the result
    imshow("spectrum magnitude", magI);
    waitKey();

    return 0;
}
