#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstring>

using namespace std;
using namespace cv;

int main( int argc, char** argv )
{
  // show help
  //! [help]
  //! [help]

  // declares all required variables
  //! [vars]
  Rect2d roi;
  Mat frame;
  //! [vars]

  // create a tracker object
  //! [create]
  Ptr<Tracker> tracker = Tracker::create( "KCF" );
  //! [create]
// 跟踪器的创建可选以下几种，代表使用的跟踪算法；
// MIL
// BOOSTING
// MEDIANFLOW
// TLD
//  KCF

  // set input video
  //! [setvideo]
  // std::string video = argv[1];
  VideoCapture cap(0);// 打开摄像头
  //! [setvideo]
  if( !cap.isOpened() ) 
  { 
    printf("打开摄像头失败\r\n");
    return -1;
  }
  int track_flag_ok=0;
  // get bounding box
  //! [getframe]
  cap.read(frame);// 读取第一帧===============
  cout<< frame.size() << endl; 
  //cap >> frame;
  //! [getframe]
  //! [selectroi]
  roi=selectROI("tracker",frame);// 选取目标框============
  //! [selectroi]

  //quit if ROI was not selected
  if(roi.width==0 || roi.height==0)
    return 0;

  // initialize the tracker
  //! [init]
  tracker->init(frame,roi);// 初始化 目标框===================
  //! [init]
  track_flag_ok=1;
  // perform the tracking process
  printf("Start the tracking process, press ESC to quit.\n");
  //for ( ;; ){
 while(cap.read(frame)) 
  {
    // get frame from the video
    // cap >> frame;
   if(!track_flag_ok)
   {
     roi=selectROI("tracker",frame);// 选取目标框============
     tracker->init(frame,roi);// 初始化 目标框===================
     track_flag_ok=1;
    }
    // stop the program if no more images
    if(frame.rows==0 || frame.cols==0)
      break;
 
    // update the tracking result
    //! [update]
    int flag_t = tracker->update(frame,roi);// 获得跟踪结果
   //  cout<< flag_t << endl; // 发现不了跟踪失败
    //! [update] 

    //! [visualization]
    // draw the tracked object

    if ( ((roi.x+roi.width/2)<0) ||  ((roi.x+roi.width/2)>640) ||
         ((roi.y+roi.height/2)<0) ||  ((roi.y+roi.height/2)>480)
       )
    { 
       printf("lost.\n"); 
       track_flag_ok=0; 
       continue;
    }
   
    rectangle( frame, roi, Scalar( 255, 0, 0 ), 2, 1 );// 绘制目标跟踪结果
    //  cout<< roi.x << "\t "<< roi.y << "\t "<<roi.width << "\t "<<roi.height << endl; 
    // if(roi.width==0 || roi.height==0)  printf("lost.\n");

    // show image with the tracked object
    imshow("tracker",frame);
    //! [visualization]

    //quit on ESC button
    if(waitKey(1)==27)break;
  }

  return 0;
}
