# lsdslam代码笔记
# 目录
    0.1. question 关键问题
    0.2. 算法框架
    0.3. 代码解析
        0.3.1. 数据结构
            0.3.1.1. Frame
            0.3.1.2. FrameMemory
            0.3.1.3. FramePoseStruct
        0.3.2. Tracking thread
        0.3.3. Mapping thread
        0.3.4. Depth estimation
            0.3.4.1. DepthMapPixelHypothesis
            0.3.4.2. DepthMap
        0.3.5. Map optimization

# 0.1. question 关键问题

        frame的reactivate是 为了节省内存资源，
        要是现在的位置(帧)可以在已经有的keyframes找到一个相似的，就把之前的那个载入进去。

        要注意的是tracking对应的是SE3Track（欧式变换跟踪）,
        闭环，constraintSearch对应的是Sim3Track(相似变换跟踪),关心的是尺度的统一性。

        (对于单目尺度问题，从小尺度的地方到大尺度的地方怎么解决，追踪的精度是跟场景的深度是有内在的关系)，

        LSD SLAM avoids this issue by using the fact that the scene depth and the tracking accuracy has inherent correlation

        lsd-slam中SlamSystem::SlamSystem(int w, int h, Eigen::Matrix3f K, bool enableSLAM)， 注意变量 enableSlam
        他说明这个代码既可以是VO(视觉里程计，只定位),也可以是slam(定位+建图)

        slam对应的就多了两个线程 optimizationThread ,constraintSearchThread，
        优化线程只在 constraintSearch 线程添加了新的约束之后，才开始优化，
        就是变量 newConstraintAdded 设置为true

        constraintSearchThread 添加约束，
        注意的是相邻关键帧的约束是通过 if(parent != 0 && forceParent), 
        forceParent来添加的，因为当前帧的parent就是前一帧
# 0.2. 算法框架
![](https://images2015.cnblogs.com/blog/542140/201707/542140-20170708103001456-41527615.png)

# 0.3. 代码解析
## 0.3.1. 数据结构
### 0.3.1.1. Frame
```c
class Frame
{
public:
    friend class FrameMemory;
    
    Frame(int id, int width, int height, const Eigen::Matrix3f& K, double timestamp, const unsigned char* image);

    /** Prepares this frame for stereo comparisons with the other frame (computes some intermediate values that will be needed) */
    void prepareForStereoWith(Frame* other, Sim3 thisToOther, const Eigen::Matrix3f& K, const int level);

    //只对关键帧有作用，非关键帧是空的
    /** Pointers to all adjacent Frames in graph. empty for non-keyframes.*/
    std::unordered_set< Frame*, std::hash<Frame*>, std::equal_to<Frame*>,
        Eigen::aligned_allocator< Frame* > > neighbors;
    
    //Tracking Reference for quick test， this is used for re-localization and re-Keyframe positioning.
    //一种快速的tracking reference
    Eigen::Vector3f* permaRef_posData;  // (x,y,z)
    Eigen::Vector2f* permaRef_colorAndVarData;  // (I, Var)
    int permaRefNumPts; 

private:
    //获取图像的数据，gredient ，inverse depth ,variance ,color
    //调用相应的buildImage，buildGradients,buildMaxGradients,buildIDepthAndIDepthVar
    void require(int dataFlags, int level = 0);
    void release(int dataFlags, bool pyramidsOnly, bool invalidateOnly);

    struct Data
    {
        //PYRAMID_LEVELS = 5 > 5 ? 5 : 5 
        float* image[PYRAMID_LEVELS];   
        bool imageValid[PYRAMID_LEVELS];
        
        Eigen::Vector4f* gradients[PYRAMID_LEVELS];
        bool gradientsValid[PYRAMID_LEVELS];
        
        float* maxGradients[PYRAMID_LEVELS];
        bool maxGradientsValid[PYRAMID_LEVELS];
        
        // negative depthvalues are actually allowed, so setting this to -1 does NOT invalidate the pixel's depth.
        // a pixel is valid iff idepthVar[i] > 0.
        float* idepth[PYRAMID_LEVELS];
        bool idepthValid[PYRAMID_LEVELS];
        
        // MUST contain -1 for invalid pixel (that dont have depth)!!
        float* idepthVar[PYRAMID_LEVELS];
        bool idepthVarValid[PYRAMID_LEVELS];
        
        // data from initial tracking, indicating which pixels in the reference frame ware good or not.
        // deleted as soon as frame is used for mapping.
        bool* refPixelWasGood;
    }

    Data data;
    
    /** Releases everything which can be recalculated, but keeps the minimal
      * representation in memory. Use release(Frame::ALL, false) to store on disk instead.
      * ONLY CALL THIS, if an exclusive lock on activeMutex is owned! */
    bool minimizeInMemory();
}

Frame::Frame(int id, int width, int height, const Eigen::Matrix3f& K, double timestamp, const unsigned char* image)
{
    //设置data的一系列值，还有一些参数
    initialize(id, width, height, K, timestamp);
    data.image[0] = FrameMemory::getInstance().getFloatBuffer(data.width[0]*data.height[0]);
    //copy from image to data.image[0]
}

void Frame::require(int dataFlags, int level)
{
    if ((dataFlags & IMAGE) && ! data.imageValid[level])
    {
        ///就是金字塔的上一层数据等于下一层数据的周围4个取平均
        buildImage(level);
    }
    if ((dataFlags & GRADIENTS) && ! data.gradientsValid[level])
    {
        //四个元素中只有前三个是有赋值,dx,dy,color
        buildGradients(level);
    }
    if ((dataFlags & MAX_GRADIENTS) && ! data.maxGradientsValid[level])
    {
        //周围四个元素中最大的梯度对应的值
        buildMaxGradients(level);
    }
    if (((dataFlags & IDEPTH) && ! data.idepthValid[level])
        || ((dataFlags & IDEPTH_VAR) && ! data.idepthVarValid[level]))
    {
        //还是上一层的周围四个元素数据通过方差的加权得到逆深度和相应的方差
        buildIDepthAndIDepthVar(level);
    }
}

void Frame::release(int dataFlags, bool pyramidsOnly, bool invalidateOnly)
{
    //返还相应的buffer
}


bool Frame::minimizeInMemory()
{
    if(activeMutex.timed_lock(boost::posix_time::milliseconds(10)))
    {
        buildMutex.lock();
    
        release(IMAGE | IDEPTH | IDEPTH_VAR, true, false);
        release(GRADIENTS | MAX_GRADIENTS, false, false);

        clear_refPixelWasGood();

        buildMutex.unlock();
        activeMutex.unlock();
        return true;
    }
    return false;
}
```
### 0.3.1.2. FrameMemory 内存管理 
```c
class FrameMemory
{
public:
    static FrameMemory& getInstance();  

    boost::shared_lock<boost::shared_mutex> activateFrame(Frame* frame);
    void deactivateFrame(Frame* frame);
    void pruneActiveFrames();

private:
    //
    std::unordered_map< void*, unsigned int > bufferSizes;
    //第一个元素是buffer size,第二个有几个这样的buffer
    std::unordered_map< unsigned int, std::vector< void* > > availableBuffers;

    boost::mutex activeFramesMutex;
    std::list<Frame*> activeFrames;
}

//只是内存的管理
void FrameMemory::pruneActiveFrames()
{
    boost::unique_lock<boost::mutex> lock(activeFramesMutex);

    while((int)activeFrames.size() > maxLoopClosureCandidates + 20)
    {
        if(!activeFrames.back()->minimizeInMemory())
        {
            if(!activeFrames.back()->minimizeInMemory())
            {
                printf("failed to minimize frame %d twice. maybe some active-lock is lingering?\n",activeFrames.back()->id());
                return;  // pre-emptive return if could not deactivate.
            }
        }
        activeFrames.back()->isActive = false;
        activeFrames.pop_back();
    }
}
```
### 0.3.1.3. FramePoseStruct 位姿结构
```c

//这个类有parent tracking ,优化之后，变成keyframe之后的变量设置
class FramePoseStruct{
public:

    //trackingParent就是reference keyframe的pose
    // parent, the frame originally tracked on. never changes.
    FramePoseStruct* trackingParent;

    // set initially as tracking result (then it's a SE(3)),
    // and is changed only once, when the frame becomes a KF (->rescale).
    //变成keyframe之后的，rescale值，尺度问题
    Sim3 thisToParent_raw;

    int frameID;
    Frame* frame;

    void setPoseGraphOptResult(Sim3 camToWorld);
    void applyPoseGraphOptResult();
private：

    // absolute position (camToWorld).
    // can change when optimization offset is merged.
    Sim3 camToWorld;  //    camToWorld = camToWorld_new;(FramePoseStruct::applyPoseGraphOptResult)
    
    // new, optimized absolute position. is added on mergeOptimization.
    Sim3 camToWorld_new;

    // whether camToWorld_new is newer than camToWorld
    bool hasUnmergedPose;
}

//要是parent tracking reference的pose优化之后，之后的child的值都要修改，pose-graph
Sim3 FramePoseStruct::getCamToWorld(int recursionDepth)
{
    // prevent stack overflow
    assert(recursionDepth < 5000);

    // if the node is in the graph, it's absolute pose is only changed by optimization.
    if(isOptimized) return camToWorld;

    // return chached pose, if still valid.
    if(cacheValidFor == cacheValidCounter)
        return camToWorld;

    // return id if there is no parent (very first frame)
    if(trackingParent == nullptr)
        return camToWorld = Sim3();

    // abs. pose is computed from the parent's abs. pose, and cached.
    cacheValidFor = cacheValidCounter;

    return camToWorld = trackingParent->getCamToWorld(recursionDepth+1) * thisToParent_raw;
}

```
## 0.3.2. Tracking thread 跟踪线程
```c
//第一帧图像的初始化
//不需要特征法的特别处理，后续优化第一帧图像的深度
void SlamSystem::randomInit(uchar* image, double timeStamp, int id)
{
    currentKeyFrame.reset(new Frame(id, width, height, K, timeStamp, image));
    map->initializeRandomly(currentKeyFrame.get());
    keyFrameGraph->addFrame(currentKeyFrame.get()); 

    if(doSlam)
    {
        keyFrameGraph->idToKeyFrameMutex.lock();
        keyFrameGraph->idToKeyFrame.insert(std::make_pair(currentKeyFrame->id(), currentKeyFrame));
        keyFrameGraph->idToKeyFrameMutex.unlock();
    }
}

//tracking
void SlamSystem::trackFrame(uchar *image , unsigned int frameID, bool blockUntilMapped,double timestamp)
{
    // Create new frame
    std::shared_ptr<Frame> trackingNewFrame(new Frame(frameID, width, height, K, timestamp, image));

//进行重定位
    if(!trackingIsGood)
    {
        relocalizer.updateCurrentFrame(trackingNewFrame);
        return;
    }
//设置trackingReference
    if(trackingReference->keyframe != currentKeyFrame.get() || currentKeyFrame->depthHasBeenUpdatedFlag)
    {
        trackingReference->importFrame(currentKeyFrame.get());
        currentKeyFrame->depthHasBeenUpdatedFlag = false;
        trackingReferenceFrameSharedPT = currentKeyFrame;
    }

    FramePoseStruct* trackingReferencePose = trackingReference->keyframe->pose;
//使用frameGraph中最近的一帧相对于trackingReference的pose作为当前帧优化的初始值
//因此要高速相机，帧与帧之间的运动要小，不然非凸函数很难收敛到正确的值
    SE3 frameToReference_initialEstimate = se3FromSim3(
            trackingReferencePose->getCamToWorld().inverse() * keyFrameGraph->allFramePoses.back()->getCamToWorld());

//SE3 track 的调用
    SE3 newRefToFrame_poseUpdate = tracker->trackFrame(
        trackingReference,
        trackingNewFrame.get(),
        frameToReference_initialEstimate);

//添加到frameGraph
    keyFrameGraph->addFrame(trackingNewFrame.get());

//当前帧是不是要设置为KeyFrame
    if (!my_createNewKeyframe && currentKeyFrame->numMappedOnThisTotal > MIN_NUM_MAPPED)
    {
        Sophus::Vector3d dist = newRefToFrame_poseUpdate.translation() * currentKeyFrame->meanIdepth;
        float minVal = fmin(0.2f + keyFrameGraph->keyframesAll.size() * 0.8f / INITIALIZATION_PHASE_COUNT,1.0f);

        if(keyFrameGraph->keyframesAll.size() < INITIALIZATION_PHASE_COUNT) minVal *= 0.7;

        lastTrackingClosenessScore = trackableKeyFrameSearch->getRefFrameScore(dist.dot(dist), tracker->pointUsage);

        if (lastTrackingClosenessScore > minVal)
        {
            createNewKeyFrame = true;
        }
    }

    if(unmappedTrackedFrames.size() < 50 || (unmappedTrackedFrames.size() < 100 && trackingNewFrame->getTrackingParent()->numMappedOnThisTotal < 10))
        unmappedTrackedFrames.push_back(trackingNewFrame);

    if(blockUntilMapped && trackingIsGood)
    {
        boost::unique_lock<boost::mutex> lock(newFrameMappedMutex);
        while(unmappedTrackedFrames.size() > 0)
        {
            //printf("TRACKING IS BLOCKING, waiting for %d frames to finish mapping.\n", (int)unmappedTrackedFrames.size());
            newFrameMappedSignal.wait(lock);
        }
        lock.unlock();
    }
}

SE3 SE3Tracker::trackFrame(TrackingReference* reference,Frame* frame,const SE3& frameToReference_initialEstimate)
{
    Sophus::SE3f referenceToFrame = frameToReference_initialEstimate.inverse().cast<float>();
//优化的最小二乘法6x6
    NormalEquationsLeastSquares ls;
    
    for(int lvl=SE3TRACKING_MAX_LEVEL-1;lvl >= SE3TRACKING_MIN_LEVEL;lvl--)
    {
//将keyframe上的点反投影到当前keyframe坐标，得到3维点云
        reference->makePointCloud(lvl);

//这是一个宏定义，call的函数是calcResidualAndBuffers
        callOptimized(calcResidualAndBuffers, (reference->posData[lvl], reference->colorAndVarData[lvl], SE3TRACKING_MIN_LEVEL == lvl ? reference->pointPosInXYGrid[lvl] : 0, reference->numData[lvl], frame, referenceToFrame, lvl, (plotTracking && lvl == SE3TRACKING_MIN_LEVEL)));

//buf_warped_size小于0.01* (width>>lvl)*(height>>lvl) ，track失败，diverged
        if(buf_warped_size < MIN_GOODPERALL_PIXEL_ABSMIN * (width>>lvl)*(height>>lvl))
        {
            diverged = true;
            trackingWasGood = false;
            return SE3();
        }

        float lastErr = callOptimized(calcWeightsAndResidual,(referenceToFrame));
 
        for(int iteration=0; iteration < settings.maxItsPerLvl[lvl]; iteration++)
        {
            callOptimized(calculateWarpUpdate,(ls));

            while(true)
            {
                // solve LS system with current lambda
                Vector6 b = -ls.b;
                Matrix6x6 A = ls.A;
//这个是什么意思
                for(int i=0;i<6;i++) A(i,i) *= 1+LM_lambda;
                Vector6 inc = A.ldlt().solve(b);
                incTry++;
            
                // apply increment. pretty sure this way round is correct, but hard to test.
                Sophus::SE3f new_referenceToFrame = Sophus::SE3f::exp((inc)) * referenceToFrame;

                // re-evaluate residual
                callOptimized(calcResidualAndBuffers, (reference->posData[lvl], reference->colorAndVarData[lvl], SE3TRACKING_MIN_LEVEL == lvl ? reference->pointPosInXYGrid[lvl] : 0, reference->numData[lvl], frame, new_referenceToFrame, lvl, (plotTracking && lvl == SE3TRACKING_MIN_LEVEL)));
                if(buf_warped_size < MIN_GOODPERALL_PIXEL_ABSMIN* (width>>lvl)*(height>>lvl))
                {
                    diverged = true;
                    trackingWasGood = false;
                    return SE3();
                }

                float error = callOptimized(calcWeightsAndResidual,(new_referenceToFrame));
                numCalcResidualCalls[lvl]++;

            }
        }
    }
    trackingWasGood = !diverged
            && lastGoodCount / (frame->width(SE3TRACKING_MIN_LEVEL)*frame->height(SE3TRACKING_MIN_LEVEL)) > MIN_GOODPERALL_PIXEL
            && lastGoodCount / (lastGoodCount + lastBadCount) > MIN_GOODPERGOODBAD_PIXEL;

    if(trackingWasGood)
        reference->keyframe->numFramesTrackedOnThis++;

    return toSophus(referenceToFrame.inverse());

}

float SE3Tracker::calcResidualAndBuffers(const Eigen::Vector3f* refPoint,const Eigen::Vector2f* refColVar,int* idxBuf,
int refNum,Frame* frame,const Sophus::SE3f& referenceToFrame,int level,bool plotResidual)
{
    Eigen::Matrix3f rotMat = referenceToFrame.rotationMatrix();
    Eigen::Vector3f transVec = referenceToFrame.translation();

    for(;refPoint<refPoint_max; refPoint++, refColVar++, idxBuf++)
    {
        Eigen::Vector3f Wxp = rotMat * (*refPoint) + transVec;

//得到在当前帧图像坐标内的坐标
        float u_new = (Wxp[0]/Wxp[2])*fx_l + cx_l;
        float v_new = (Wxp[1]/Wxp[2])*fy_l + cy_l;

//三个元素是: dx,dy,color
        Eigen::Vector3f resInterp = getInterpolatedElement43(frame_gradients, u_new, v_new, w);
//在两个frame之间color的变换
        float c1 = affineEstimation_a * (*refColVar)[0] + affineEstimation_b;
        float c2 = resInterp[2];
        float residual = c1 - c2;
//权重，residual的阀值是5
        float weight = fabsf(residual) < 5.0f ? 1 : 5.0f / fabsf(residual);
        sxx += c1*c1*weight;
        syy += c2*c2*weight;
        sx += c1*weight;
        sy += c2*weight;
        sw += weight;
//判断这个点是不是一个好的track点
        bool isGood = residual*residual / (MAX_DIFF_CONSTANT + MAX_DIFF_GRAD_MULT*(resInterp[0]*resInterp[0] + resInterp[1]*resInterp[1])) < 1;


//这些数据的记录
        *(buf_warped_x+idx) = Wxp(0);
        *(buf_warped_y+idx) = Wxp(1);
        *(buf_warped_z+idx) = Wxp(2);

        *(buf_warped_dx+idx) = fx_l * resInterp[0];
        *(buf_warped_dy+idx) = fy_l * resInterp[1];
        *(buf_warped_residual+idx) = residual;

        *(buf_d+idx) = 1.0f / (*refPoint)[2];
        *(buf_idepthVar+idx) = (*refColVar)[1];
        idx++;

        if(isGood)
        {
            sumResUnweighted += residual*residual;
            sumSignedRes += residual;
            goodCount++;
        }
        else
            badCount++;


        float depthChange = (*refPoint)[2] / Wxp[2];    // if depth becomes larger: pixel becomes "smaller", hence count it less.
        usageCount += depthChange < 1 ? depthChange : 1;

    }       sx += c1*weight;

    lastMeanRes = sumSignedRes / goodCount;

    affineEstimation_a_lastIt = sqrtf((syy - sy*sy/sw) / (sxx - sx*sx/sw));
    affineEstimation_b_lastIt = (sy - affineEstimation_a_lastIt*sx)/sw;

    return sumResUnweighted / goodCount;

}

//////////////////////// 计算误差 和误差权重
float SE3Tracker::calcWeightsAndResidual(const Sophus::SE3f& referenceToFrame)
{
    float tx = referenceToFrame.translation()[0];
    float ty = referenceToFrame.translation()[1];
    float tz = referenceToFrame.translation()[2];

    float sumRes = 0;

//
    for(int i=0;i<buf_warped_size;i++)
    {
        float px = *(buf_warped_x+i);   // x'
        float py = *(buf_warped_y+i);   // y'
        float pz = *(buf_warped_z+i);   // z'
        float d = *(buf_d+i);   // d
        float rp = *(buf_warped_residual+i); // r_p
        float gx = *(buf_warped_dx+i);  // \delta_x I
        float gy = *(buf_warped_dy+i);  // \delta_y I
        float s = settings.var_weight * *(buf_idepthVar+i); // \sigma_d^2

        //di /d(u,v) ,d(p)/d(p') = d(rp+t)/d(p') , d(p')/d d,展开求解就可以
        // calc dw/dd (first 2 components):
        float g0 = (tx * pz - tz * px) / (pz*pz*d);
        float g1 = (ty * pz - tz * py) / (pz*pz*d);

        // calc w_p
        float drpdd = gx * g0 + gy * g1;    // ommitting the minus
        float w_p = 1.0f / ((cameraPixelNoise2) + s * drpdd * drpdd);

        float weighted_rp = fabs(rp*sqrtf(w_p));

        float wh = fabs(weighted_rp < (settings.huber_d/2) ? 1 : (settings.huber_d/2) / weighted_rp);

        sumRes += wh * w_p * rp*rp;


        *(buf_weight_p+i) = wh * w_p;
    }

    return sumRes / buf_warped_size;
}

//
Vector6 SE3Tracker::calculateWarpUpdate(NormalEquationsLeastSquares &ls)
{
    ls.initialize(width*height);
    for(int i=0;i<buf_warped_size;i++)
    {
        float px = *(buf_warped_x+i);
        float py = *(buf_warped_y+i);
        float pz = *(buf_warped_z+i);
        float r =  *(buf_warped_residual+i);
        float gx = *(buf_warped_dx+i);
        float gy = *(buf_warped_dy+i);
        // step 3 + step 5 comp 6d error vector

        float z = 1.0f / pz;
        float z_sqr = 1.0f / (pz*pz);
        Vector6 v;
        v[0] = z*gx + 0;
        v[1] = 0 +         z*gy;
        v[2] = (-px * z_sqr) * gx +
              (-py * z_sqr) * gy;
        v[3] = (-px * py * z_sqr) * gx +
              (-(1.0 + py * py * z_sqr)) * gy;
        v[4] = (1.0 + px * px * z_sqr) * gx +
              (px * py * z_sqr) * gy;
        v[5] = (-py * z) * gx +
              (px * z) * gy;

        // step 6: integrate into A and b:
//r是color，v是jacobian
        ls.update(v, r, *(buf_weight_p+i));
    }
    Vector6 result;

    // solve ls
    ls.finish();
    ls.solve(result);

    return result;
}


```
![](https://images2015.cnblogs.com/blog/542140/201707/542140-20170708103039472-1068389235.png)
    
## 0.3.3. Mapping thread 建图线程
```c
// PUSHED in tracking, READ & CLEARED in mapping
std::deque< std::shared_ptr<Frame> > unmappedTrackedFrames;

bool SlamSystem::doMappingIteration()
{
    //变量设置为optimization线程已经优化过的pose变换 
    mergeOptimizationOffset();

    if(trackingIsGood)
    {
        //doMapping false 的话，只有tracking线程，没有mappinp线程，也就是不能应对快速运动
        if(!doMapping)
        {
            //printf("tryToChange refframe, lastScore %f!\n", lastTrackingClosenessScore);
            if(lastTrackingClosenessScore > 1)
                changeKeyframe(true, false, lastTrackingClosenessScore * 0.75);

            if (displayDepthMap || depthMapScreenshotFlag)
                debugDisplayDepthMap();

            return false;
        }

        //创建关键帧
        if (createNewKeyFrame)
        {
            finishCurrentKeyframe(); 
            changeKeyframe(false, true, 1.0f);


            if (displayDepthMap || depthMapScreenshotFlag)
                debugDisplayDepthMap();
        }
        //更新当前的关键帧
        else
        {
            bool didSomething = updateKeyframe();

            if (displayDepthMap || depthMapScreenshotFlag)
                debugDisplayDepthMap();
            if(!didSomething)
                return false;
        }

        return true;
    }
    //重定位的方案
    else
    {
        // invalidate map if it was valid.
        if(map->isValid())
        {
            if(currentKeyFrame->numMappedOnThisTotal >= MIN_NUM_MAPPED)
                finishCurrentKeyframe(); 
            elseedgeErrorSum
                discardCurrentKeyframe();

            map->invalidate();
        }

        // start relocalizer if it isnt running already
        if(!relocalizer.isRunning)
            relocalizer.start(keyFrameGraph->keyframesAll);

        // did we find a frame to relocalize with?
        if(relocalizer.waitResult(50))
            takeRelocalizeResult();


        return true;
    }
}

bool SlamSystem::updateKeyframe()
{
    std::deque< std::shared_ptr<Frame> > references;

    if(unmappedTrackedFrames.size() > 0)
    {
        map->updateKeyframe(references);
    }
}


void DepthMap::updateKeyframe(std::deque< std::shared_ptr<Frame> > referenceFrames)
{
    for(std::shared_ptr<Frame> frame : referenceFrames)
    {
        Sim3 refToKf;
        if(frame->pose->trackingParent->frameID == activeKeyFrame->id())
            refToKf = frame->pose->thisToParent_raw;
        else
            refToKf = activeKeyFrame->getScaledCamToWorld().inverse() *  frame->getScaledCamToWorld();

        frame->prepareForStereoWith(activeKeyFrame, refToKf, K, 0);
    }

    observeDepth();  // threadReducer.reduce(boost::biobserveDepthCreatend(&DepthMap::observeDepthRow, this, _1, _2, _3), 3, height-3, 10);

    regularizeDepthMapFillHoles();

    regularizeDepthMap(false, VAL_SUM_MIN_FOR_KEEP);

    //设置activeKeyFrame的深度图
    if(!activeKeyFrame->depthHasBeenUpdatedFlag)
    {
        activeKeyFrame->setDepth(currentDepthMap);
    }

}

void SlamSystem::finishCurrentKeyframe()
{
    map->finalizeKeyFrame();
    if(SLAMEnabled){
        mappingTrackingReference->importFrame(currentKeyFrame.get());
        currentKeyFrame->setPermaRef(mappingTrackingReference);
        mappingTrackingReference->invalidate();

    }
}

void slamYSystem::changeKeyframe(bool noCreate, bool force , float maxScore){
    if(doKFReActivation && SLAMEnabled)
    {
        newReferenceKF = trackableKeyFrameSearch->findRePositionCandidate(newKeyframeCandidate.get(), maxScore);
    }
    if(newReferenceKF != 0)
        loadNewCurrentKeyframe(newReferenceKF);
    else{
        if(force)
        {
            if(noCreate)
            {
                trackingIsGood = false;
                nextRelocIdx = -1;
                printf("mapping is disabled & moved outside of known map. Starting Relocalizer!\n");
            }
            else
                createNewCurrentKeyframe(newKeyframeCandidate);
        }
    }
}

void SlamSystem::createNewCurrentKeyframe(std::shared_ptr<Frame> newKeyframeCandidate)
{
    if(SLAMEnabled)
    {
        // add NEW keyframe to id-lookup
        keyFrameGraph->idToKeyFrameMutex.lock();
        keyFrameGraph->idToKeyFrame.insert(std::make_pair(newKeyframeCandidate->id(), newKeyframeCandidate));
        keyFrameGraph->idToKeyFrameMutex.unlock();
    }

    // propagate & make new.
    map->createKeyFrame(newKeyframeCandidate.get());


}

void DepthMap::createKeyFrame(Frame* new_keyframe)
{
    propagateDepth(new_keyframe);

    regularizeDepthMap(true, VAL_SUM_MIN_FOR_KEEP);

    regularizeDepthMapFillHoles();

    regularizeDepthMap(false, VAL_SUM_MIN_FOR_KEEP);

    activeKeyFrame->setDepth(currentDepthMap);

}

```

## 0.3.4. Depth estimation 深度估计线程
### 0.3.4.1. DepthMapPixelHypothesis 深度图 假设
     DepthMap中的每一个点的深度符合高斯分布:
```c
class DepthMapPixelHypothesis
{
public:
    /** Counter for validity, basically how many successful observations are incorporated. */
    int validity_counter;
    
    /** Actual Gaussian Distribution.*/
    float idepth;
    float idepth_var;

    /** Smoothed Gaussian Distribution.*/
    float idepth_smoothed;
    float idepth_var_smoothed;

    inline DepthMapPixelHypothesis(
            const float &my_idepth,
            const float &my_idepth_smoothed,
            const float &my_idepth_var,
            const float &my_idepth_var_smoothed,
            const int &my_validity_counter) :
                isValid(true),
                blacklisted(0),
                nextStereoFrameMinID(0),
                validity_counter(my_validity_counter),
                idepth(my_idepth),
                idepth_var(my_idepth_var),
                idepth_smoothed(my_idepth_smoothed),
                idepth_var_smoothed(my_idepth_var_smoothed) {};
    
    //
    cv::Vec3b getVisualizationColor(int lastFrameID) const;

}

```
###  0.3.4.2. DepthMap深度图
```c
class DepthMap
{
public:

    DepthMap(int w, int h, const Eigen::Matrix3f& K);

//传进去的参数是deque,队列的形式
    void updateKeyframe(std::deque< std::shared_ptr<Frame> > referenceFrames);

    void createKeyFrame(Frame* new_keyframe);
private:

    inline float doLineStereo(
            const float u, const float v, const float epxn, const float epyn,
            const float min_idepth, const float prior_idepth, float max_idepth,
            const Frame* const referenceFrame, const float* referenceFrameImage,
            float &result_idepth, float &result_var, float &result_eplLength,
            RunningStats* const stats);


}

void DepthMap::updateKeyframe(std::deque< std::shared_ptr<Frame> > referenceFrames)
{
    //最old的reference frame
    oldest_referenceFrame = referenceFrames.front().get();
    //最young的referemce frame
    newest_referenceFrame = referenceFrames.back().get();
    referenceFrameByID.clear();
    referenceFrameByID_offset = oldest_referenceFrame->id();

    for(std::shared_ptr<Frame> frame : referenceFrames)
    {
        //相对于activeKeyFrame的姿态
        Sim3 refToKf;

        frame->prepareForStereoWith(activeKeyFrame, refToKf, K, 0);
    }

    observeDepth();  //  thread 调用observeDepthRow，更新每一个点的depth

    regularizeDepthMapFillHoles();

    regularizeDepthMap(false, VAL_SUM_MIN_FOR_KEEP);

    activeKeyFrame->setDepth(currentDepthMap);

}

void DepthMap::createKeyFrame(Frame* new_keyframe)
{
    boost::shared_lock<boost::shared_mutex> lock2 = new_keyframe->getActiveLock();

    SE3 oldToNew_SE3 = se3FromSim3(new_keyframe->pose->thisToParent_raw).inverse();

//
    propagateDepth(new_keyframe);

//注意activeKeyFrame的设置
    activeKeyFrame = new_keyframe;
    activeKeyFramelock = activeKeyFrame->getActiveLock();
    activeKeyFrameImageData = new_keyframe->image(0);
    activeKeyFrameIsReactivated = false;

    regularizeDepthMap(true, VAL_SUM_MIN_FOR_KEEP);

    regularizeDepthMapFillHoles();

    regularizeDepthMap(false, VAL_SUM_MIN_FOR_KEEP);

// make mean inverse depth be one.
    float sumIdepth=0, numIdepth=0;
    for(DepthMapPixelHypothesis* source = currentDepthMap; source < currentDepthMap+width*height; source++)
    {
        if(!source->isValid)
            continue;
        sumIdepth += source->idepth_smoothed;
        numIdepth++;
    }
    float rescaleFactor = numIdepth / sumIdepth;
    float rescaleFactor2 = rescaleFactor*rescaleFactor;

    for(DepthMapPixelHypothesis* source = currentDepthMap; source < currentDepthMap+width*height; source++)
    {
        if(!source->isValid)
            continue;
        source->idepth *= rescaleFactor;
        source->idepth_smoothed *= rescaleFactor;
        source->idepth_var *= rescaleFactor2;
        source->idepth_var_smoothed *= rescaleFactor2;
    }
    activeKeyFrame->pose->thisToParent_raw = sim3FromSE3(oldToNew_SE3.inverse(), rescaleFactor);
    activeKeyFrame->pose->invalidateCache();
    
    activeKeyFrame->setDepth(currentDepthMap);

    
}

void DepthMap::observeDepthRow(int yMin, int yMax, RunningStats* stats)
{
    for(int y=yMin;y<yMax; y++)
        for(int x=3;x<width-3;x++){
            if(!hasHypothesis)
                success = observeDepthCreate(x, y, idx, stats);
            else
                success = observeDepthUpdate(x, y, idx, keyFrameMaxGradBuf, stats);
        }
}

bool DepthMap::observeDepthCreate(const int &x, const int &y, const int &idx, RunningStats* const &stats)
{
    bool isGood = makeAndCheckEPL(x, y, refFrame, &epx, &epy, stats);
    if(!isGood) return false;

    float error = doLineStereo(
            new_u,new_v,epx,epy,
            0.0f, 1.0f, 1.0f/MIN_DEPTH,
            refFrame, refFrame->image(0),
            result_idepth, result_var, result_eplLength, stats);

    *target = DepthMapPixelHypothesis(
            result_idepth,
            result_var,
            VALIDITY_COUNTER_INITIAL_OBSERVE);

}

bool DepthMap::observeDepthUpdate(const int &x, const int &y, const int &idx, const float* keyFrameMaxGradBuf, RunningStats* const &stats)
{
    bool isGood = makeAndCheckEPL(x, y, refFrame, &epx, &epy, stats);

    // which exact point to track, and where from.
    //mean +- 2 \sigma 深度范围进行搜索
    float sv = sqrt(target->idepth_var_smoothed);
    float min_idepth = target->idepth_smoothed - sv*STEREO_EPL_VAR_FAC;
    float max_idepth = target->idepth_smoothed + sv*STEREO_EPL_VAR_FAC;

    float error = doLineStereo(
            x,y,epx,epy,
            min_idepth, target->idepth_smoothed ,max_idepth,
            refFrame, refFrame->image(0),
            result_idepth, result_var, result_eplLength, stats);


    if(error == -1){
        //out of bounds
    }
    else if(error == -2){
        //not good for stereo(e.g. some inf/nan occured, has inconsistent minimum)
    }
    else if(error  == -3){
        //if not found (error to high)
    }
    .....
    else{
        //do textbook ekf update
        // increase var by a little (prediction-uncertainty)
        float id_var = target->idepth_var*SUCC_VAR_INC_FAC;

        //update var with observation
        float w = result_var / (result_var + id_var);
        float new_idepth = (1-w)*result_idepth + w*target->idepth;
        target->idepth = UNZERO(new_idepth);

        // variance can only decrease from observation; never increase.
        id_var = id_var * w;
        if(id_var < target->idepth_var)
            target->idepth_var = id_var;

        // increase validity!
        target->validity_counter += VALIDITY_COUNTER_INC;
        float absGrad = keyFrameMaxGradBuf[idx];
        if(target->validity_counter > VALIDITY_COUNTER_MAX+absGrad*(VALIDITY_COUNTER_MAX_VARIABLE)/255.0f)
            target->validity_counter = VALIDITY_COUNTER_MAX+absGrad*(VALIDITY_COUNTER_MAX_VARIABLE)/255.0f;
    }
    
}

bool DepthMap::makeAndCheckEPL(const int x, const int y, const Frame* const ref, float* pepx, float* pepy, RunningStats* const stats)
{
//是不是看成 两边同除以z(ref->thisToOther_t[2])，变成x/减去ref->thisToOther_t在图像上投影的位置
    float epx = - fx * ref->thisToOther_t[0] + ref->thisToOther_t[2]*(x - cx);
    float epy = - fy * ref->thisToOther_t[1] + ref->thisToOther_t[2]*(y - cy);
    // ======== check epl length =========
    float eplLengthSquared = epx*epx+epy*epy;
    
    float gx = activeKeyFrameImageData[idx+1] - activeKeyFrameImageData[idx-1];
    float gy = activeKeyFrameImageData[idx+width] - activeKeyFrameImageData[idx-width];
    // ===== check epl-grad magnitude ======
    float eplGradSquared = gx * epx + gy * epy;

    eplGradSquared = eplGradSquared*eplGradSquared / eplLengthSquared;  // square and norm with epl-length

    if(eplGradSquared < MIN_EPL_GRAD_SQUARED)
    {
        if(enablePrintDebugInfo) stats->num_observe_skipped_small_epl_grad++;
        return false;
    }

    // ===== check epl-grad angle ======
    if(eplGradSquared / (gx*gx+gy*gy) < MIN_EPL_ANGLE_SQUARED)
    {
        if(enablePrintDebugInfo) stats->num_observe_skipped_small_epl_angle++;
        return false;
    }
        // ===== DONE - return "normalized" epl =====
    float fac = GRADIENT_SAMPLE_DIST / sqrt(eplLengthSquared);
    *pepx = epx * fac;
    *pepy = epy * fac;

    return true;
}


inline float DepthMap::doLineStereo(
    const float u, const float v, const float epxn, const float epyn,
    const float min_idepth, const float prior_idepth, float max_idepth,
    const Frame* const referenceFrame, const float* referenceFrameImage,
    float &result_idepth, float &result_var, float &result_eplLength,
    RunningStats* stats)
{
    // calculate epipolar line start and end point in old image
    Eigen::Vector3f KinvP = Eigen::Vector3f(fxi*u+cxi,fyi*v+cyi,1.0f);
    Eigen::Vector3f pInf = referenceFrame->K_otherToThis_R * KinvP;
    Eigen::Vector3f pReal = pInf / prior_idepth + referenceFrame->K_otherToThis_t;

    float rescaleFactor = pReal[2] * prior_idepth

    // calculate values to search for
    float realVal_p1 = getInterpolatedElement(activeKeyFrameImageData,u + epxn*rescaleFactor, v + epyn*rescaleFactor, width);
    float realVal_m1 = getInterpolatedElement(activeKeyFrameImageData,u - epxn*rescaleFactor, v - epyn*rescaleFactor, width);
    float realVal = getInterpolatedElement(activeKeyFrameImageData,u, v, width);
    float realVal_m2 = getInterpolatedElement(activeKeyFrameImageData,u - 2*epxn*rescaleFactor, v - 2*epyn*rescaleFactor, width);
    float realVal_p2 = getInterpolatedElement(activeKeyFrameImageData,u + 2*epxn*rescaleFactor, v + 2*epyn*rescaleFactor, width);
    
    Eigen::Vector3f pClose = pInf + referenceFrame->K_otherToThis_t*max_idepth;

    Eigen::Vector3f pFar = pInf + referenceFrame->K_otherToThis_t*min_idepth;
    
    // calculate increments in which we will step through the epipolar line.
    // they are sampleDist (or half sample dist) long
    float incx = pClose[0] - pFar[0];
    float incy = pClose[1] - pFar[1];
    float eplLength = sqrt(incx*incx+incy*incy);

    incx *= GRADIENT_SAMPLE_DIST/eplLength;
    incy *= GRADIENT_SAMPLE_DIST/eplLength;

    // extend one sample_dist to left & right.
    pFar[0] -= incx;
    pFar[1] -= incy;
    pClose[0] += incx;
    pClose[1] += incy;

    // from here on:
    // - pInf: search start-point
    // - p0: search end-point
    // - incx, incy: search steps in pixel
    // - eplLength, min_idepth, max_idepth: determines search-resolution, i.e. the result's variance.
    

    float cpx = pFar[0];
    float cpy =  pFar[1];

    float val_cp_m2 = getInterpolatedElement(referenceFrameImage,cpx-2.0f*incx, cpy-2.0f*incy, width);
    float val_cp_m1 = getInterpolatedElement(referenceFrameImage,cpx-incx, cpy-incy, width);
    float val_cp = getInterpolatedElement(referenceFrameImage,cpx, cpy, width);
    float val_cp_p1 = getInterpolatedElement(referenceFrameImage,cpx+incx, cpy+incy, width);
    float val_cp_p2;
    

    
}

```
## 0.3.5. Map optimization 地图优化
```c
void SlamSystem::constraintSearchThreadLoop()
{
    while(keepRunning)
    {
        if(newKeyFrames.size() == 0)
        {
            if(keyFrameGraph->keyframesForRetrack.size() > 10)
            {
                int found = findConstraintsForNewKeyFrames(toReTrackFrame, false, false, 2.0);

            }

        }
        else{
                findConstraintsForNewKeyFrames(newKF, true, true, 1.0);
        }
    }
}

//对应的是不是sim3，尺度漂移问题 ，添加g2o中的边约束
int SlamSystem::findConstraintsForNewKeyFrames(Frame* newKeyFrame, bool forceParent, bool useFABMAP, float closeCandidatesTH)
{
    
    // =============== get all potential candidates and their initial relative pose. =================
    std::vector<KFConstraintStruct*, Eigen::aligned_allocator<KFConstraintStruct*> > constraints;

    std::unordered_set<Frame*, std::hash<Frame*>, std::equal_to<Frame*>,
        Eigen::aligned_allocator< Frame* > > candidates = trackableKeyFrameSearch->findCandidates(newKeyFrame, fabMapResult, useFABMAP, closeCandidatesTH);
    
    std::map< Frame*, Sim3, std::less<Frame*>, Eigen::aligned_allocator<std::pair<Frame*, Sim3> > > candidateToFrame_initialEstimateMap;

    // erase the ones that are already neighbours.


    // =============== distinguish between close and "far" candidates in Graph =================
    // Do a first check on trackability of close candidates.

    SO3 disturbance = SO3::exp(Sophus::Vector3d(0.05,0,0));
    for (Frame* candidate : candidates)
    {
                SE3 c2f_init = se3FromSim3(candidateToFrame_initialEstimateMap[candidate].inverse()).inverse();
        c2f_init.so3() = c2f_init.so3() * disturbance;
        SE3 c2f = constraintSE3Tracker->trackFrameOnPermaref(candidate, newKeyFrame, c2f_init);
        if(!constraintSE3Tracker->trackingWasGood) {closeFailed++; continue;}


        SE3 f2c_init = se3FromSim3(candidateToFrame_initialEstimateMap[candidate]).inverse();
        f2c_init.so3() = disturbance * f2c_init.so3();
        SE3 f2c = constraintSE3Tracker->trackFrameOnPermaref(newKeyFrame, candidate, f2c_init);
        if(!constraintSE3Tracker->trackingWasGood) {closeFailed++; continue;}

        if((f2c.so3() * c2f.so3()).log().norm() >= 0.09) {closeInconsistent++; continue;}

        closeCandidates.insert(candidate);


        for (Frame* candidate : candidates)
        {

            farCandidates.push_back(candidate);
        }

        // erase the ones that we tried already before (close)

        // erase the ones that are already neighbours (far)

        // =============== limit number of close candidates ===============
        // while too many, remove the one with the highest connectivity.
    }

    for(unsigned int i=0;i<constraints.size();i++)
        keyFrameGraph->insertConstraint(constraints[i]);

}


bool SlamSystem::optimizationIteration(int itsPerTry, float minChange)
{
    // Do the optimization. This can take quite some time!
    int its = keyFrameGraph->optimize(itsPerTry);

    // save the optimization result.
    for(size_t i=0;i<keyFrameGraph->keyframesAll.size(); i++)
    {
        // set change
        keyFrameGraph->keyframesAll[i]->pose->setPoseGraphOptResult(
                keyFrameGraph->keyframesAll[i]->pose->graphVertex->estimate());

        // add error
        for(auto edge : keyFrameGraph->keyframesAll[i]->pose->graphVertex->edges())
        {
            keyFrameGraph->keyframesAll[i]->edgeErrorSum += ((EdgeSim3*)(edge))->chi2();
            keyFrameGraph->keyframesAll[i]->edgesNum++;
        }
    }
    
}

```



