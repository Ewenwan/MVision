#include "elas.h"
#include <opencv2/opencv.hpp>
using namespace cv;
int main()
{
	//Mat leftim=imread("left01.jpg");
	//Mat rightim=imread("right01.jpg");
  
        //Mat leftim=imread("view1.png");
	//Mat rightim=imread("view5.png");
	
        Mat leftim=imread("../img/aloe_left.pgm");
	Mat rightim=imread("../img/aloe_right.pgm");
	
	Mat dest;
	StereoELAS elas(0,128);// 最小视差  视差范围

	// we can set various parameter
		//elas.elas.param.ipol_gap_width=;
		//elas.elas.param.speckle_size=getParameter("speckle_size");
		//elas.elas.param.speckle_sim_threshold=getParameter("speckle_sim");
	
	elas(leftim,rightim,dest,100);// 边界延拓
	
	Mat show;
	dest.convertTo(show,CV_8U,1.0/8);
	imshow("disp",show);
	imwrite("disp2.jpg",show);
	waitKey();
	return 0;
}
