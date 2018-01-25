/**
* This file is part of ORB-SLAM2.
* 
* 
* 
*/

#ifndef ORBEXTRACTOR_H
#define ORBEXTRACTOR_H

#include <vector>
#include <list>
#include <opencv/cv.h>


namespace ORB_SLAM2
{

    class ExtractorNode
    {
    public:
	ExtractorNode():bNoMore(false){}

	void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

	std::vector<cv::KeyPoint> vKeys;
	cv::Point2i UL, UR, BL, BR;
	std::list<ExtractorNode>::iterator lit;
	bool bNoMore;
    };

    class ORBextractor
    {
    public:
	
	enum {HARRIS_SCORE=0, FAST_SCORE=1 };

	ORBextractor(int nfeatures, float scaleFactor, int nlevels,
		    int iniThFAST, int minThFAST);

	~ORBextractor(){}//析构函数

	// Compute the ORB features and descriptors on an image.
	// ORB are dispersed on the image using an octree.
	// Mask is ignored in the current implementation.
	void operator()( cv::InputArray image, cv::InputArray mask,
	  std::vector<cv::KeyPoint>& keypoints,
	  cv::OutputArray descriptors);

	int inline GetLevels(){
	    return nlevels;}

	float inline GetScaleFactor(){
	    return scaleFactor;}

	std::vector<float> inline GetScaleFactors(){
	    return mvScaleFactor;
	}

	std::vector<float> inline GetInverseScaleFactors(){
	    return mvInvScaleFactor;
	}

	std::vector<float> inline GetScaleSigmaSquares(){
	    return mvLevelSigma2;
	}

	std::vector<float> inline GetInverseScaleSigmaSquares(){
	    return mvInvLevelSigma2;
	}

	std::vector<cv::Mat> mvImagePyramid;

    protected:

	void ComputePyramid(cv::Mat image);
	void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);    
	std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
					      const int &maxX, const int &minY, const int &maxY, const int &nFeatures, const int &level);

	void ComputeKeyPointsOld(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);
	std::vector<cv::Point> pattern;

	int nfeatures;
	double scaleFactor;
	int nlevels;
	int iniThFAST;
	int minThFAST;

	std::vector<int> mnFeaturesPerLevel;

	std::vector<int> umax;

	std::vector<float> mvScaleFactor;
	std::vector<float> mvInvScaleFactor;    
	std::vector<float> mvLevelSigma2;
	std::vector<float> mvInvLevelSigma2;
    };

} //namespace ORB_SLAM

#endif

