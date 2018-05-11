#include "FrameData.h"
#include <iostream>
#include <string>
#include <fstream>
#include <time.h>

#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
//#include <opencv2/nonfree/features2d.hpp>

#include "g2o/core/sparse_optimizer.h"

using namespace cv;
using namespace std;
using namespace pcl;

void showmsg()
{
    cerr<< "You haven't set the dataset file path:"<<endl;
    cout<< "--this project use TUM RGB-D benchmark as test data"<<endl;
    cout<< "--you can down load it from the website!"<<endl;
    cout<< "--three para , argv demo: ./rgbdslam ~/rgbddata/ rgb.txt depth.txt "<<endl;
    cout<< "the defalut file path is loaded"<<endl;
}

int FeatureMatch(Mat img_1, 
                 Mat img_2,
                 vector< KeyPoint > &keypoints_1,
                 vector< KeyPoint > &keypoints_2,
                 vector< DMatch > &good_matches
                 )
{
    if(!img_1.data || !img_2.data)
    {
        cout << "Error reading images"<<endl;
    }

    // detect
    SurfFeatureDetector detector( 400 );
    
    detector.detect(img_1, keypoints_1);
    detector.detect(img_2, keypoints_2);
    // descriptor
    SurfDescriptorExtractor extractor;
    Mat descriptors_1, descriptors_2;
    extractor.compute(img_1,keypoints_1,descriptors_1);
    extractor.compute(img_2,keypoints_2,descriptors_2);
    // match
    FlannBasedMatcher matcher;
    vector< DMatch > matches;
    matcher.match( descriptors_1, descriptors_2, matches);

    double max_dist =0;double min_dist = 100.00;
    for(int i=0; i< descriptors_1.rows; i++) 
    {
        double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }

    for(int i =0; i< descriptors_1.rows; i++)
    {
        if(matches[i].distance <= max(2 * min_dist, 0.02))
        {
            good_matches.push_back(matches[i]);
           // cout <<"points:"<< keypoints_1[matches[i].trainIdx].pt << keypoints_2[matches[i].queryIdx].pt <<endl;
        }
    }
    
    // draw good matches
    Mat img_matches;
    drawMatches(img_1,keypoints_1,img_2,keypoints_2,good_matches, img_matches,Scalar::all(-1),Scalar::all(-1),vector<char>(),DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
   // imshow("good match", img_matches);
   // waitKey(0);
    imwrite("/home/hyj/rgbddata/match.png",img_matches);
    return 0;
}

bool EstimateMotion(const PointCloud<PointXYZRGBA>::Ptr oldPointCloud,
                   const PointCloud<PointXYZRGBA>::Ptr currentPointCloud,
                   const vector< KeyPoint > keypoints1,
                   const vector< KeyPoint > keypoints2,
                   const vector< DMatch > goodmatches,
                   Eigen::Matrix4f &rt
                  )
{ 
    PointCloud< PointXYZRGBA >::Ptr cloudin( new PointCloud< PointXYZRGBA >);
    PointCloud< PointXYZRGBA >::Ptr cloudout( new PointCloud< PointXYZRGBA >);
    
    for(vector< DMatch > ::const_iterator it = goodmatches.begin(); it != goodmatches.end(); ++it)
    {
        Point current_pt = keypoints1[it->trainIdx].pt;
        Point old_pt = keypoints2[it->queryIdx].pt;
        if(current_pt.x > 640 || current_pt.x < 0|| current_pt.y > 480||current_pt.y < 0)
        {
            continue;
        }

        if(old_pt.x > 640 || old_pt.x < 0|| old_pt.y > 480||old_pt.y < 0)
        {
            continue;
        }
        cout<< current_pt << old_pt << endl;

        // filter the NaN data
        if( isFinite( currentPointCloud->at(int(current_pt.x),int(current_pt.y)))
            && isFinite(  oldPointCloud->at(int(old_pt.x),int(old_pt.y)) ))
        {
           // cout <<  currentPointCloud->at(int(current_pt.x),int(current_pt.y))<<endl;
           // cout <<   oldPointCloud->at(int(old_pt.x),int(old_pt.y)) <<endl;
            cloudin->push_back( currentPointCloud->at(int(current_pt.x),int(current_pt.y)) );
            cloudout->push_back( oldPointCloud->at(int(old_pt.x),int(old_pt.y)) );
        }
        else
        { cout<< "NaN data in point cloud"<<endl;}

    }

    /* pcl1.7的icp没有ransac,得自己写 */
    cout << "start icp :"<<endl;
    IterativeClosestPoint< PointXYZRGBA, PointXYZRGBA > icp;
    icp.setInputSource(cloudin);
    icp.setInputTarget(cloudout);
    icp.setMaxCorrespondenceDistance(0.08); // 8cm
    icp.setMaximumIterations(50);
    PointCloud< PointXYZRGBA > Final;
    icp.align(Final);
    if(icp.hasConverged())
    {
        cout << "icp has converged: "<< icp.hasConverged()<<endl;
        rt = icp.getFinalTransformation() ;
        cout << rt <<endl;
        return true;
    }
    else 
        return false;

}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        showmsg();
        
    }
    else
    {

    }
    clock_t begin,end;
    double time_spend;
    
    FrameData GrabData("/home/hyj/rgbddata/");

    begin = clock();
    FrameRGBD current_frame,old_frame;
    GrabData.nextFrame(old_frame);
    end = clock();
    time_spend = (double)(end - begin)/CLOCKS_PER_SEC;
    cout << "GOT A FRAME , Time SPEND:"<<time_spend<<endl;
    
    PointCloud< PointXYZRGBA >::Ptr SumCloud(new PointCloud< PointXYZRGBA >);
    PointCloud< PointXYZRGBA >::Ptr out(new PointCloud< PointXYZRGBA >);
    PointCloud< PointXYZRGBA >::Ptr showout(new PointCloud< PointXYZRGBA >);
    
    Eigen::Matrix4f T_i = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f temp_t;

    VoxelGrid< PointXYZRGBA > filter;
    visualization::CloudViewer viewer("test");
     SumCloud = old_frame.pointcloudPtr;

    //while( GrabData.nextFrame(current_frame))
    for (int i = 0;i<20;i++)
    {
        GrabData.nextFrame(current_frame);
        vector< KeyPoint > keypoints_1;
        vector< KeyPoint > keypoints_2;
        vector< DMatch > good_matches;
       // cout << "feature match"<<endl;
        FeatureMatch(old_frame.rgbimg,current_frame.rgbimg,keypoints_1,keypoints_2,good_matches);
       // cout << "end match,icp start:"<<endl;
        if(EstimateMotion(old_frame.pointcloudPtr,current_frame.pointcloudPtr,keypoints_1,keypoints_2,good_matches, temp_t ))
        {
            T_i = temp_t * T_i;
            Eigen::Matrix4f t = T_i.inverse();
            transformPointCloud( *current_frame.pointcloudPtr,*out,t);
         // keypoints_1.clear();
      //  keypoints_2.clear();
       // good_matches.clear();
             (*SumCloud) += *out ;
        }
         old_frame = current_frame;

    }

         filter.setInputCloud(SumCloud);
         filter.setLeafSize(0.01f,0.01f,0.01f);
         filter.filter(*showout);
         viewer.showCloud(showout);

    /*
     GrabData.nextFrame(current_frame);
    {
        vector< KeyPoint > keypoints_1;
        vector< KeyPoint > keypoints_2;
        vector< DMatch > good_matches;

        FeatureMatch(old_frame.rgbimg,current_frame.rgbimg,keypoints_1,keypoints_2,good_matches);
        EstimateMotion(old_frame.pointcloudPtr,current_frame.pointcloudPtr,keypoints_1,keypoints_2,good_matches );

        old_frame = current_frame;
    }
    */
    while( ! viewer.wasStopped())
    {
        //imshow("rgb",current_frame.rgbimg);
        //imshow("dpt",current_dpt);
        //waitKey(0);

    }
    
    return 0;
}
