/*
 * 全局 3D-2D 点匹配算法 3D点是转化到第一帧帧图像 相机坐标系(世界坐标系)下的点
 * 匹配算法 也是当前描述子和  地图描述子匹配 对应的 3D点
 * 地图点 第一帧 的3D点全部加入地图 此后 新的一帧 图像在地图中 没有匹配到的 像素点 转换成世界坐标系下的三维点后 加入地图
 * 同时对地图优化 不在当前帧的点（看不见的点） 删除 匹配次数不高的 点   删除 视角过大删除
 */

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <algorithm>
#include <boost/timer.hpp>

#include "myslam/config.h"
#include "myslam/visual_odometry.h"
#include "myslam/g2o_types.h"

namespace myslam
{
	
	  //【1】默认默认构造函数
      VisualOdometry::VisualOdometry() :
	  state_ ( INITIALIZING ), ref_ ( nullptr ), curr_ ( nullptr ), map_ ( new Map ), num_lost_ ( 0 ), num_inliers_ ( 0 ), matcher_flann_ ( new cv::flann::LshIndexParams ( 5,10,2 ) )
      {
	      num_of_features_    = Config::get<int> ( "number_of_features" );// 特征数量   整型 int
	      scale_factor_       = Config::get<double> ( "scale_factor" );   // 尺度因子 缩小
	      level_pyramid_      = Config::get<int> ( "level_pyramid" );     // 层级 
	      match_ratio_        = Config::get<float> ( "match_ratio" );     // 匹配 参数
	      max_num_lost_       = Config::get<float> ( "max_num_lost" );    // 最大丢失次数
	      min_inliers_        = Config::get<int> ( "min_inliers" );       // 最小内点数量
	      key_frame_min_rot   = Config::get<double> ( "keyframe_rotation" );//最小的旋转      成为关键帧的条件
	      key_frame_min_trans = Config::get<double> ( "keyframe_translation" ); // 最小平移  
	      map_point_erase_ratio_ = Config::get<double> ("map_point_erase_ratio");//
	      orb_ = cv::ORB::create ( num_of_features_, scale_factor_, level_pyramid_ );//创建orb特征
      }
	  // 【2】析构函数   shared_ptr 智能指针 自带析构函数功能
      VisualOdometry::~VisualOdometry()
      {}

      bool VisualOdometry::addFrame ( Frame::Ptr frame )
      {
	  switch ( state_ )
	  {
	  case INITIALIZING:
	  {
		  state_ = OK;//初始化之后切换状态为 OK
		  curr_ = ref_ = frame;//初始化 参考帧 当前帧 
		  extractKeyPoints();  //提取关键点
		  computeDescriptors();//计算描述子
		  addKeyFrame();       // the first frame is a key-frame  添加关键帧  到地图中  还有 3D点
		  break;
	  }
	  case OK:
	  {
		  curr_ = frame;
		  curr_->T_c_w_ = ref_->T_c_w_;
		  extractKeyPoints();  //提取关键点
		  computeDescriptors();//计算描述子
		    featureMatching(); //特征点匹配 得到 特征点匹配点对
		    poseEstimationPnP();//位置估计 加入非线性优化算法 根据特征点匹配点对  计算坐标转换 及估计相机位姿
	      if ( checkEstimatedPose() == true ) // a good estimation
	      {
		  curr_->T_c_w_ = T_c_w_estimated_;
		  optimizeMap();
		  num_lost_ = 0;
		  if ( checkKeyFrame() == true ) // is a key-frame
		  {
		      addKeyFrame();
		  }
	      }
	      else // bad estimation due to various reasons
	      {
		  num_lost_++;
		  if ( num_lost_ > max_num_lost_ )
		  {
		      state_ = LOST;
		  }
		  return false;
	      }
	      break;
	  }
	  case LOST:
	  {
	      cout<<"vo has lost."<<endl;
	      break;
	  }
	  }

	  return true;
      }

	    // 提取关键点
      void VisualOdometry::extractKeyPoints()
      {
	  boost::timer timer;
	  orb_->detect ( curr_->color_, keypoints_curr_ );
	  cout<<"extract keypoints cost time: "<<timer.elapsed() <<endl;
      }
	    //计算描述子
      void VisualOdometry::computeDescriptors()
      {
	  boost::timer timer;
	  orb_->compute ( curr_->color_, keypoints_curr_, descriptors_curr_ );
	  cout<<"descriptor computation cost time: "<<timer.elapsed() <<endl;
      }
	    //特征点描述子匹配
	    // 生成 地图 3D点和对应 的 描述子
      void VisualOdometry::featureMatching()
      {
	  boost::timer timer;
	  vector<cv::DMatch> matches;// 描述子匹配
	  // select the candidates in map 
	  Mat desp_map;//3D点对于的  描述子 地图
	  vector<MapPoint::Ptr> candidate;          // 路标点 地图点 
	  for ( auto& allpoints: map_->map_points_ )// 所有的 地图中的点 已经转换到 第一帧相机坐标系下(世界坐标系)
	  {
	      MapPoint::Ptr& p = allpoints.second;
	      // check if p in curr frame image 
	      if ( curr_->isInFrame(p->pos_) )// 在当前帧 当前视野下
	      {
		  // add to candidate 
		  p->visible_times_++;                 // 观察到的次数
		  candidate.push_back( p );            // 地图3D点
		  desp_map.push_back( p->descriptor_ );// 地图3D点对应的描述子
	      }
	  }
      
	  matcher_flann_.match ( desp_map, descriptors_curr_, matches );// 大规模匹配算法  一帧特征点描述子 和 地图描述子匹配
	  // select the best matches   匹配对最小的距离
	  float min_dis = std::min_element (
			      matches.begin(), matches.end(),
			      [] ( const cv::DMatch& m1, const cv::DMatch& m2 )
	  {
	      return m1.distance < m2.distance;
	  } )->distance;

	  match_3dpts_.clear();// 匹配到的地图三维点 
	  match_2dkp_index_.clear();//当前帧 特征点 匹配到3D点 的二维点序号
	  for ( cv::DMatch& m : matches )
	  {
	      if ( m.distance < max<float> ( min_dis*match_ratio_, 30.0 ) )
	      {
		  match_3dpts_.push_back( candidate[m.queryIdx] );//匹配到的 地图3D点
		  match_2dkp_index_.push_back( m.trainIdx );//对应的 序号 以对于 2D的像素点
	      }
	  }
	  cout<<"good matches: "<<match_3dpts_.size() <<endl;
	  cout<<"match cost time: "<<timer.elapsed() <<endl;
      }

	    //位置估计 加入非线性优化算法
	  //2D-3D 位姿估计
      void VisualOdometry::poseEstimationPnP()
      {
	  // construct the 3d 2d observations
	  vector<cv::Point3f> pts3d;// 3D点
	  vector<cv::Point2f> pts2d;// 2D点

	  for ( int index:match_2dkp_index_ )//匹配 序号 对应的 2维像素坐标点
	  {
	      pts2d.push_back ( keypoints_curr_[index].pt );// 从关键点中提取 有匹配点对的 关键点
	  }
	  for ( MapPoint::Ptr pt:match_3dpts_ )//对应匹配的 地图3D点
	  {
	      pts3d.push_back( pt->getPositionCV() );//转换成 CV格式的 3D点 
	  }
          // 相机内参数
	  Mat K = ( cv::Mat_<double> ( 3,3 ) <<
		    ref_->camera_->fx_, 0, ref_->camera_->cx_,
		    0, ref_->camera_->fy_, ref_->camera_->cy_,
		    0,0,1
		  );
	  Mat rvec, tvec, inliers;
		// 采集采样序列  PnP算法求解  2D-3D点对求解 旋转向量 rvec,  平移矩阵 tvec    符合Rt的点数量 在 回归到的系数方程上（误差范围内）
	  cv::solvePnPRansac ( pts3d, pts2d, K, Mat(), rvec, tvec, false, 100, 4.0, 0.99, inliers );
	  num_inliers_ = inliers.rows;
	  cout<<"pnp inliers: "<<num_inliers_<<endl;
	      // PnP算法求解到的初始解
	  T_c_w_estimated_ = SE3 (
				SO3 ( rvec.at<double> ( 0,0 ), rvec.at<double> ( 1,0 ), rvec.at<double> ( 2,0 ) ),
				Vector3d ( tvec.at<double> ( 0,0 ), tvec.at<double> ( 1,0 ), tvec.at<double> ( 2,0 ) )
			    );

	  //在上述PNP初始 求解之后加入图优化
	  // using bundle adjustment to optimize the pose
	  typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,2>> Block;// 矩阵块 分解 求解器
	  Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>();
	  Block* solver_ptr = new Block ( linearSolver );
	  g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( solver_ptr );// 迭代优化算法
	  g2o::SparseOptimizer optimizer;
	  optimizer.setAlgorithm ( solver );
	// 添加顶点
	  g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();//顶点 位姿 
	  pose->setId ( 0 );
	  pose->setEstimate ( g2o::SE3Quat (// 估计值
	      T_c_w_estimated_.rotation_matrix(), T_c_w_estimated_.translation()//旋转矩阵 和平移矩阵
	  ));
	  optimizer.addVertex ( pose );

	  //添加边 edges
	  for ( int i=0; i<inliers.rows; i++ )
	  {
	      int index = inliers.at<int> ( i,0 );
	      // 3D -> 2D projection
	      EdgeProjectXYZ2UVPoseOnly* edge = new EdgeProjectXYZ2UVPoseOnly();//  2D-3D位姿 估计 边 仅优化 位姿
	      edge->setId ( i );// id
	      edge->setVertex ( 0, pose );//连接的顶点
	      edge->camera_ = curr_->camera_.get();//相机参数
	      edge->point_ = Vector3d ( pts3d[index].x, pts3d[index].y, pts3d[index].z );//3D点
	      edge->setMeasurement ( Vector2d ( pts2d[index].x, pts2d[index].y ) );//对应 2维 像素坐标点
	      edge->setInformation ( Eigen::Matrix2d::Identity() );//误差权重 信息矩阵
	      optimizer.addEdge ( edge );
	      // set the inlier map points 
	      match_3dpts_[index]->matched_times_++;// 地图 3D点 已经被匹配的次数记录
	  }

	  optimizer.initializeOptimization();
	  optimizer.optimize ( 10 );

	  T_c_w_estimated_ = SE3 (
	      pose->estimate().rotation(),//旋转矩阵
	      pose->estimate().translation()//平移矩阵
	  );
	  
	  cout<<"T_c_w_estimated_: "<<endl<<T_c_w_estimated_.matrix()<<endl;
      }

	    //检查估计的位姿
      bool VisualOdometry::checkEstimatedPose()
      {
	  // check if the estimated pose is good
	  if ( num_inliers_ < min_inliers_ )// 随机采样序列算法 求得的符合 R t 的内点 点对数量过少 计算错误
	  {
	      cout<<"reject because inlier is too small: "<<num_inliers_<<endl;
	      return false;
	  }
	  // if the motion is too large, it is probably wrong
	  SE3 T_r_c = ref_->T_c_w_ * T_c_w_estimated_.inverse();//T_c_w_ 上一帧 图像对世界坐标系 的T ;
	  //T_c_w_estimated_当前帧 图像对世界坐标系 的T     前后两帧的 R t
	  
	  Sophus::Vector6d d = T_r_c.log();
	  if ( d.norm() > 5.0 )//姿态变化较大 不太可能
	  {
	      cout<<"reject because motion is too large: "<<d.norm() <<endl;
	      return false;
	  }
	  return true;
      }

	    //检查是否为关键帧
      bool VisualOdometry::checkKeyFrame()
      {
	  SE3 T_r_c = ref_->T_c_w_ * T_c_w_estimated_.inverse();
	  //T_c_w_ 上一帧 图像对世界坐标系 的T ;
	  //T_c_w_estimated_当前帧 图像对世界坐标系 的T     前后两帧的 R t
	  Sophus::Vector6d d = T_r_c.log();//前后两帧的 R t
	  Vector3d trans = d.head<3>();//平移
	  Vector3d rot = d.tail<3>();//旋转矩阵
	  if ( rot.norm() >key_frame_min_rot || trans.norm() >key_frame_min_trans )//平移 换旋转 超过 一个限度 认为是 关键帧
	      return true;
	  return false;
      }


      void VisualOdometry::addKeyFrame()
      {
	  if ( map_->keyframes_.empty() )
	  {
	      // first key-frame, add all 3d points into map
	      for ( size_t i=0; i<keypoints_curr_.size(); i++ )
	      {
		  double d = curr_->findDepth ( keypoints_curr_[i] );
		  if ( d < 0 ) 
		      continue;
		  Vector3d p_world = ref_->camera_->pixel2world (
		      Vector2d ( keypoints_curr_[i].pt.x, keypoints_curr_[i].pt.y ), curr_->T_c_w_, d
		  );
		  Vector3d n = p_world - ref_->getCamCenter();//Normal of viewing direction 
		  n.normalize();
		  MapPoint::Ptr map_point = MapPoint::createMapPoint(
		      p_world, n, descriptors_curr_.row(i).clone(), curr_.get()
		  );
		  map_->insertMapPoint( map_point );//添加 地图点 坐标 点到地图  第一帧 添加所有的点 到地图中
	      }
	  }
	  
	  map_->insertKeyFrame ( curr_ );//添加关键帧
	  ref_ = curr_;
      }

      //添加点到地图中
      void VisualOdometry::addMapPoints()
      {
	  // add the new map points into map
	  vector<bool> matched(keypoints_curr_.size(), false); // 匹配标志
	  for ( int index:match_2dkp_index_ )//匹配点 对 序号
	      matched[index] = true;// 匹配到的点 置位 匹配标志
	  for ( int i=0; i<keypoints_curr_.size(); i++ )
	  {
	      if ( matched[i] == true )   
		  continue;// 跳过后面 新的2维点 和 前地图 有匹配的 说明实际物理点已在地图中 对于没有匹配的点 符合条件的 转换成 实际坐标系三维点 后 加入地图
	      double d = ref_->findDepth ( keypoints_curr_[i] );
	      if ( d<0 )  // 深度为0  也跳过后面 
		  continue;
	      // 没有匹配的 二维像素点 转换到 第一幅图像 世界坐标系下
	      Vector3d p_world = ref_->camera_->pixel2world (
		  Vector2d ( keypoints_curr_[i].pt.x, keypoints_curr_[i].pt.y ), 
		  curr_->T_c_w_, d
	      );
	      
	      Vector3d n = p_world - ref_->getCamCenter();// 
	      n.normalize();
	      MapPoint::Ptr map_point = MapPoint::createMapPoint(
		  p_world, n, descriptors_curr_.row(i).clone(), curr_.get()
	      );
	      map_->insertMapPoint( map_point );
	  }
      }

      void VisualOdometry::optimizeMap()
      {
	  // remove the hardly seen and no visible points  剔除看不见的点
	  for ( auto iter = map_->map_points_.begin(); iter != map_->map_points_.end(); )
	  {
	      if ( !curr_->isInFrame(iter->second->pos_) )
	      {
		  iter = map_->map_points_.erase(iter);//不在当前帧下  删除
		  continue;
	      }
	      float match_ratio = float(iter->second->matched_times_)/iter->second->visible_times_;
	      if ( match_ratio < map_point_erase_ratio_ )//匹配次数不高的 点 删除
	      {
		  iter = map_->map_points_.erase(iter);
		  continue;
	      }
	      
	      double angle = getViewAngle( curr_, iter->second );
	      if ( angle > M_PI/6. )//视角过大删除
	      {
		  iter = map_->map_points_.erase(iter);
		  continue;
	      }
	      if ( iter->second->good_ == false )
	      {
		  // TODO try triangulate this map point 
	      }
	      iter++;
	  }
	  
	  if ( match_2dkp_index_.size()<100 )//匹配点对小于 100个  两幅图像重合度过小 添加点到新地图
	      addMapPoints();// 添加 没有匹配到的点
	  if ( map_->map_points_.size() > 1000 )  //地图点数 过大
	  {
	      // TODO map is too large, remove some one 
	      map_point_erase_ratio_ += 0.05;//增大 匹配率 门槛限制
	  }
	  else 
	      map_point_erase_ratio_ = 0.1;//
	  cout<<"map points: "<<map_->map_points_.size()<<endl;
      }

      double VisualOdometry::getViewAngle ( Frame::Ptr frame, MapPoint::Ptr point )
      {
	  Vector3d n = point->pos_ - frame->getCamCenter();
	  n.normalize();
	  return acos( n.transpose()*point->norm_ );
      }


}
