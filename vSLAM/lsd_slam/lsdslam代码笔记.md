# lsdslam代码笔记
# 目录
    0.1. question
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
