/*ADCensus 代价计算  = AD＋Census
     1.  AD即 Absolute Difference 三通道颜色差值绝对值之和求均值
     2. Census Feature特征原理很简单
　　　　　　　         在指定窗口内比较周围亮度值与中心点的大小，匹配距离用汉明距表示。
　　　　　　　　　　　　　   Census保留周边像素空间信息，对光照变化有一定鲁棒性。

就是在给定的窗口内，比较中心像素与周围邻居像素之间的大小关系，大了就为1，小了就为0，
然后每个像素都对应一个二值编码序列，然后通过海明距离来表示两个像素的相似程度，

     3. 信息结合
　　　　　　　　　　　　　　　　cost = r(Cad , lamd1) + r(Cces, lamd2)
        r(C , lamd) = 1 - exp(- c/ lamd)
   　 　　Cross-based 代价聚合:
        自适应窗口代价聚合，
		在设定的最大窗口范围内搜索，
　　　　　　　　　　　　　满足下面三个约束条件确定每个像素的十字坐标，完成自适应窗口的构建。
　　　　　　　　　　　　　　　Scanline 代价聚合优化


*/
#include "adcensuscv.h"

ADCensusCV::ADCensusCV(const Mat &leftImage, const Mat &rightImage, Size censusWin, float lambdaAD, float lambdaCensus)
{
    this->leftImage = leftImage;
    this->rightImage = rightImage;
    this->censusWin = censusWin;
    this->lambdaAD = lambdaAD;
    this->lambdaCensus = lambdaCensus;
}
// 对应像素差的绝对值 3通道均值  Absolute Differences
float ADCensusCV::ad(int wL, int hL, int wR, int hR) const
{
    float dist = 0;
    const Vec3b &colorLP = leftImage.at<Vec3b>(hL, wL);
    const Vec3b &colorRP = rightImage.at<Vec3b>(hR, wR);

    for(uchar color = 0; color < 3; ++color)
    {
        dist += std::abs(colorLP[color] - colorRP[color]);
    }
    return (dist / 3);//3通道均值
}
// census值
float ADCensusCV::census(int wL, int hL, int wR, int hR) const
{
    float dist = 0;
    const Vec3b &colorRefL = leftImage.at<Vec3b>(hL, wL);//中心点 颜色
    const Vec3b &colorRefR = rightImage.at<Vec3b>(hR, wR);

    for(int h = -censusWin.height / 2; h <= censusWin.height / 2; ++h)
    {
        for(int w = -censusWin.width / 2; w <= censusWin.width / 2; ++w)
        {// 在指定窗口内比较周围亮度值与中心点的大小
            const Vec3b &colorLP = leftImage.at<Vec3b>(hL + h, wL + w);
            const Vec3b &colorRP = rightImage.at<Vec3b>(hR + h, wR + w);
            for(uchar color = 0; color < 3; ++color)
            {
      // bool diff = (colorLP[color] < colorRefL[color]) ^ (colorRP[color] < colorRefR[color]);
         bool diff = (colorLP[color] - colorRefL[color]) * (colorRP[color] - colorRefR[color]) < 0;
                dist += (diff)? 1: 0;// 匹配距离用汉明距表示 
// 都比中心点大/小　保留周边像素空间信息，对光照变化有一定鲁棒性
            }
        }
    }

    return dist;
}
// ad + census值 信息结合
float ADCensusCV::adCensus(int wL, int hL, int wR, int hR) const
{
    float dist;

    // compute Absolute Difference cost
    float cAD = ad(wL, hL, wR, hR);

    // compute Census cost
    float cCensus = census(wL, hL, wR, hR);

    // combine the two costs
    dist = 1 - exp(-cAD / lambdaAD);
    dist += 1 - exp(-cCensus / lambdaCensus);

    return dist;
}
