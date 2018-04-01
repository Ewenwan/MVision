/* 鲁棒匹配 两者相互看对眼了　
 * 图像1匹配到图像2的点　和图像2 匹配 图像1的点相互对应　
　* 才算是匹配　　
 * RobustMatcher.cpp
 *
 *  Created on: Jun 4, 2014
 *      Author: eriba
 */

#include "RobustMatcher.h"
#include <time.h>

#include <opencv2/features2d/features2d.hpp>

// 虚析构函数
RobustMatcher::~RobustMatcher()
{
  // TODO Auto-generated destructor stub
}

// 计算图像的关键点
// 给定图像　按照给定的　特征点检测器detector_ 　
// 进行检测　　detector_->detect(image, keypoints);
void RobustMatcher::computeKeyPoints( const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints)
{
  detector_->detect(image, keypoints);
}

void RobustMatcher::computeDescriptors( const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
  extractor_->compute(image, keypoints, descriptors);
}

 // 最匹配　和　次匹配　的比值大于　一个阈值　我认为这个匹配比较好
int RobustMatcher::ratioTest(std::vector<std::vector<cv::DMatch> > &matches)
{
  int removed = 0;
  // 对于每一个匹配点对　matches
  for ( std::vector<std::vector<cv::DMatch> >::iterator
        matchIterator= matches.begin(); matchIterator!= matches.end(); ++matchIterator)
  {
    // 如果检测到了两个最近点　if 2 NN has been identified
    if (matchIterator->size() > 1)
    {
      // 检测最近的距离和次近的距离的比值是否超过阈值
      if ((*matchIterator)[0].distance / (*matchIterator)[1].distance > ratio_)
      {
        matchIterator->clear(); //超过阈值  不好　最匹配距离和　次匹配距离差不多　这个匹配不好
        removed++;
      }
    }
    else
    { // 没有两个匹配点　直接排除　does not have 2 neighbours
      matchIterator->clear(); // remove match　
      removed++;
    }
  }
  return removed;
}


// 返回　相互匹配的 匹配点对
// image 1 -> image 2  == image 2 -> image 1
void RobustMatcher::symmetryTest( const std::vector<std::vector<cv::DMatch> >& matches1,
                     const std::vector<std::vector<cv::DMatch> >& matches2,
                     std::vector<cv::DMatch>& symMatches )
{

  // for all matches image 1 -> image 2
   for (std::vector<std::vector<cv::DMatch> >::const_iterator
       matchIterator1 = matches1.begin(); matchIterator1 != matches1.end(); ++matchIterator1)
   {

      // ignore deleted matches
      if (matchIterator1->empty() || matchIterator1->size() < 2)
         continue;

      // for all matches image 2 -> image 1
      for (std::vector<std::vector<cv::DMatch> >::const_iterator
          matchIterator2 = matches2.begin(); matchIterator2 != matches2.end(); ++matchIterator2)
      {
        // ignore deleted matches
        if (matchIterator2->empty() || matchIterator2->size() < 2)
           continue;

        // Match symmetry test
        if ((*matchIterator1)[0].queryIdx ==
            (*matchIterator2)[0].trainIdx &&
            (*matchIterator2)[0].queryIdx ==
            (*matchIterator1)[0].trainIdx)
        {
            // add symmetrical match
            symMatches.push_back(
              cv::DMatch((*matchIterator1)[0].queryIdx,
                         (*matchIterator1)[0].trainIdx,
                         (*matchIterator1)[0].distance));
            break; // next match in image 1 -> image 2
        }
      }
   }

}

// 鲁棒匹配　两者相互看对眼了　
// 图像1匹配到图像2的点　和图像2 匹配 图像1的点相互对应　 才算是匹配　　
void RobustMatcher::robustMatch( const cv::Mat& frame, std::vector<cv::DMatch>& good_matches,
              std::vector<cv::KeyPoint>& keypoints_frame, const cv::Mat& descriptors_model )
{

  // 1a. Detection of the ORB features
  this->computeKeyPoints(frame, keypoints_frame);

  // 1b. Extraction of the ORB descriptors
  cv::Mat descriptors_frame;
  this->computeDescriptors(frame, keypoints_frame, descriptors_frame);

  // 2. Match the two image descriptors
  std::vector<std::vector<cv::DMatch> > matches12, matches21;

  // 2a. From image 1 to image 2
  matcher_->knnMatch(descriptors_frame, descriptors_model, matches12, 2); // return 2 nearest neighbours

  // 2b. From image 2 to image 1
  matcher_->knnMatch(descriptors_model, descriptors_frame, matches21, 2); // return 2 nearest neighbours

  // 3. Remove matches for which NN ratio is > than threshold
  // clean image 1 -> image 2 matches
  ratioTest(matches12);
  // clean image 2 -> image 1 matches
  ratioTest(matches21);

  // 4. Remove non-symmetrical matches
  symmetryTest(matches12, matches21, good_matches);

}

 // 快速鲁棒匹配不需要相互看对眼　单相思也可以
void RobustMatcher::fastRobustMatch( const cv::Mat& frame, std::vector<cv::DMatch>& good_matches,
                                 std::vector<cv::KeyPoint>& keypoints_frame,
                                 const cv::Mat& descriptors_model )
{
  good_matches.clear();

  // 1a. Detection of the ORB features
  this->computeKeyPoints(frame, keypoints_frame);

  // 1b. Extraction of the ORB descriptors
  cv::Mat descriptors_frame;
  this->computeDescriptors(frame, keypoints_frame, descriptors_frame);

  // 2. Match the two image descriptors
  std::vector<std::vector<cv::DMatch> > matches;
  matcher_->knnMatch(descriptors_frame, descriptors_model, matches, 2);

  // 3. Remove matches for which NN ratio is > than threshold
  ratioTest(matches);

  // 4. Fill good matches container
  for ( std::vector<std::vector<cv::DMatch> >::iterator
         matchIterator= matches.begin(); matchIterator!= matches.end(); ++matchIterator)
  {
    if (!matchIterator->empty()) good_matches.push_back((*matchIterator)[0]);
  }

}
