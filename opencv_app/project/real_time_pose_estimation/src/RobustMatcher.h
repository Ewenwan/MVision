/*
  鲁棒匹配 两者相互看对眼了　
 * 图像1匹配到图像2的点　和图像2 匹配 图像1的点相互对应　
　* 才算是匹配
 * RobustMatcher.h

 */

#ifndef ROBUSTMATCHER_H_
#define ROBUSTMATCHER_H_

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

class RobustMatcher {
public:
  // 默认构造函数
  RobustMatcher() : ratio_(0.8f)
  {
    // ORB is the default feature
    detector_ = cv::ORB::create();
    extractor_ = cv::ORB::create();

    // BruteFroce matcher with Norm Hamming is the default matcher
    matcher_ = cv::makePtr<cv::BFMatcher>((int)cv::NORM_HAMMING, false);

  }

  // 虚析构函数
  virtual ~RobustMatcher();

  // 设置特征点检测器　　 feature detector
  void setFeatureDetector(const cv::Ptr<cv::FeatureDetector>& detect) {  detector_ = detect; }

  // 设置特征点的描述子提取器　descriptor extractor
  void setDescriptorExtractor(const cv::Ptr<cv::DescriptorExtractor>& desc) { extractor_ = desc; }

  // 设置描述子匹配器
  void setDescriptorMatcher(const cv::Ptr<cv::DescriptorMatcher>& match) {  matcher_ = match; }

  // 计算图像的关键点 
  // 给定图像　按照给定的　特征点检测器detector_ 　
  // 进行检测　　detector_->detect(image, keypoints);
  void computeKeyPoints( const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints);

  // 计算所给图像　关键点　的描述子
  // 给定图像　关键点
  // 使用描述子提取器　extractor_ 
  // 进行提取　extractor_->compute(image, keypoints, descriptors);
  void computeDescriptors( const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

  // 设置比率　Set ratio parameter for the ratio test
  void setRatio( float rat) { ratio_ = rat; }

 // 最匹配　和　次匹配　的比值大于　一个阈值　我认为这个匹配比较好
  int ratioTest(std::vector<std::vector<cv::DMatch> > &matches);

  // 返回　相互匹配的　匹配
  // image 1 -> image 2  == image 2 -> image 1
  void symmetryTest( const std::vector<std::vector<cv::DMatch> >& matches1,
                     const std::vector<std::vector<cv::DMatch> >& matches2,
                     std::vector<cv::DMatch>& symMatches );

  //　鲁棒匹配　相互匹配　看对眼　才会长久
  void robustMatch( const cv::Mat& frame, std::vector<cv::DMatch>& good_matches,
                      std::vector<cv::KeyPoint>& keypoints_frame,
                      const cv::Mat& descriptors_model );

 // 快速鲁棒匹配不需要相互看对眼　单相思也可以
 void fastRobustMatch( const cv::Mat& frame, std::vector<cv::DMatch>& good_matches,
                       std::vector<cv::KeyPoint>& keypoints_frame,
                       const cv::Mat& descriptors_model );

private:
  // pointer to the feature point detector object
  cv::Ptr<cv::FeatureDetector> detector_;
  // pointer to the feature descriptor extractor object
  cv::Ptr<cv::DescriptorExtractor> extractor_;
  // pointer to the matcher object
  cv::Ptr<cv::DescriptorMatcher> matcher_;
  // max ratio between 1st and 2nd NN
  float ratio_;
};

#endif /* ROBUSTMATCHER_H_ */
