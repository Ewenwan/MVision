# MVision　Machine Vision 机器视觉
[学习无人驾驶车，你所必须知道的](https://zhuanlan.zhihu.com/p/27686577)

[强化学习从入门到放弃的资料](https://zhuanlan.zhihu.com/p/34918639?utm_source=wechat_session&utm_medium=social&wechatShare=1&from=singlemessage&isappinstalled=0)

[台大 机器学习深度学习课程](http://speech.ee.ntu.edu.tw/~tlkagk/courses_MLSD15_2.html)

[斯坦福CS231计算机视觉2017](http://www.mooc.ai/course/268/learn?lessonid=1819#lesson/1819)

[2018 MIT 6.S094 麻省理工深度学习和自动驾驶课程 中文](http://www.mooc.ai/course/483/notes)

[MIT  深度学习和自动驾驶课程 英文](https://selfdrivingcars.mit.edu/)
##  公司
[视觉领域的部分国内公司](http://www.ipcv.org/cvcom/)
###  初创公司：
[图普科技](http://www.tuputech.com/)

[Face++](http://www.faceplusplus.com.cn/)

[Linkface](http://www.linkface.cn/index.html)

[Minieye](http://www.minieye.cc/cn/)

[知图Cogtu](http://www.cogtu.com/?lang=zh)

[商汤科技Sensetime](http://www.sensetime.com/cn)

[亮风台Hiscene](http://www.hiscene.com/)

[掌赢科技](http://www.zhangying.mobi/index.html)

[格灵深瞳DeepPG](http://www.deepglint.com/)

[凌感科技usens](http://www.lagou.com/gongsi/j114187.html)

[图森TuSimple](http://www.tusimple.com/)

[中科视拓Seetatech(山世光)](http://www.seetatech.com/)

[第四范式](https://www.4paradigm.com/product/prophet)

### 上市公司：

[百度DL实验室](http://idl.baidu.com/)

[腾讯优图](http://youtu.qq.com/)

[阿里高德](http://www.newsmth.net/nForum/#!article/Career_Upgrade/429476)

[暴风魔镜](http://www.newsmth.net/nForum/#!article/Career_PHD/225254)

[搜狗](http://www.newsmth.net/nForum/#!article/Career_PHD/224449)

[乐视tv](http://www.newsmth.net/nForum/#!article/Career_PHD/222651)

[奇虎360](http://www.newsmth.net/nForum/#!article/Career_PHD/222379)

[京东实验室](http://www.newsmth.net/nForum/#!article/Career_PHD/223133/a>)

[阿里巴巴](http://www.newsmth.net/nForum/#!article/Career_PHD/222007)

[联想研究院](http://www.newsmth.net/nForum/#!article/Career_PHD/220225)

[华为研究院](http://www.newsmth.net/nForum/#!article/Career_PHD/225976)

### 知名外企：
[佳能信息](http://www.newsmth.net/nForum/#!article/Career_PHD/222548)

[索尼研究院](http://www.newsmth.net/nForum/#!article/Career_PHD/223437)

[富士通研发中心](http://www.newsmth.net/nForum/#!article/Career_PHD/220654)

[微软研究院](https://careers.microsoft.com/?rg=cn)

[英特尔研究院](http://www.newsmth.net/nForum/#!article/Career_PHD/221175)

[三星研究院](http://www.yingjiesheng.com/job-001-742-124.html)



## 0 计算摄影　摄影几何
[计算摄影方面的部分课程讲义](http://www.ipcv.org/cp-lecture/)
[相机内部图像处理流程](http://www.comp.nus.edu.sg/~brown/ICIP2013_Brown.html)
[pdf](http://www.comp.nus.edu.sg/~brown/ICIP2013_Tutorial_Brown.pdf)

    相机 = 光测量装置(Camera = light-measuring device)
        照明光源(Illumination source)（辐射(radiance)） --> 
        场景元素(Scene Element)   --->
        成像系统(Imaging System)  --->
        内部图像平面(Internal Image Plane) --->
        输出（数字）图像(Output (digital) image) 
    图像 = 辐射能量测量(Image = radiant-energy measurement)  
    
    
    现代摄影流水线　Modern photography pipeline 
    场景辐射　--->　相机前端(镜头过滤器 镜头Lens 快门Shutter 孔径)　--->　
    相机内部(ccd响应response（RAW） CCD插值Demosaicing （原）)　--->　
    相机后端处理(直方图均衡Hist equalization、空间扭曲Spatial warping)--->　输出
    


    透过棱镜的白光 　“White light” through a prism  ------> 折射光(Refracted light)----> 光谱 Spectral 　
    我们的眼睛有三个受体（锥细胞），它们对可见光作出反应并产生颜色感。
    
[CSC320S: Introduction to Visual Computing 视觉计算导论 ](http://www.cs.toronto.edu/~kyros/courses/320/)

[Facebook surround 360 《全景图拼接》](https://github.com/facebook/Surround360)

        输入：17张raw图像，包括14张side images、2张top images、1张bottom image
        输出：3D立体360度全景图像  
[博客笔记](https://blog.csdn.net/electech6/article/details/53618965)   
        

[深度摄影风格转换 Deep Photo Style Transfer](https://github.com/luanfujun/deep-photo-styletransfer)

### 图像形变 Image warping
[参考](http://www.ipcv.org/image-warping/)
### 色彩增强/转换　Color transfer
[参考](http://www.ipcv.org/colortransfer/)
### 图像修补 Image repair
[参考](http://www.ipcv.org/imagerepair/)
### 图像去噪 Image denoise
[参考](http://www.ipcv.org/imagedenoise/)
### 图像去模糊 Image deblur 
[参考](http://www.ipcv.org/imagedeblur/)
###  图像滤波 Image filter
[参考](http://www.ipcv.org/imagefilter/)

###  超分辨率 Super-resolution
[参考](http://www.ipcv.org/code-superresolution/)  


## 1　三维重建 3D Modeling
[参考](http://www.ipcv.org/category/code-data/3dd/)
### 相机矫正　Camera calibration
[参考](http://www.ipcv.org/poseestimation/)
### 非刚体重建　Non-rigid modeling　
[参考](http://www.ipcv.org/nonnigitreco/)
### 三维重构 3D modeling
[参考](http://www.ipcv.org/3dmodeling/)
[视觉SLAM](http://www.ipcv.org/on-visual-slam/)

[Self-augmented Convolutional Neural Networks](https://github.com/msraig/self-augmented-net)


[运动估计 motion estimation](http://www.ipcv.org/on-motion-estimation/)

[面部变形　face morphing　](http://www.ipcv.org/about-face-morphing/)

[三维重建方面的视觉人物](http://www.ipcv.org/people-3d-modeling/)


## 2  匹配/跟踪 Matching & Tracking
[参考](http://www.ipcv.org/category/code-data/tracking/)

### 2.a 特征提取 Feature extraction
[参考](http://www.ipcv.org/featureextraction/)

### 2.b 特征匹配 Feature matching
[参考](http://www.ipcv.org/code-featmatching/)

### 2.c 时空匹配 Space-time matching
[参考](http://www.ipcv.org/code-spacetimematching/)

### 2.d 区域匹配 Region matching
[参考](http://www.ipcv.org/code-regionmatching/)

### 2.e 轮廓匹配 Contour matching
[参考](http://www.ipcv.org/code-coutourmatching/)

### 2.f 立体匹配 Stereo matching 
[参考](http://www.ipcv.org/code-stereomatching/)

[双目视觉自动驾 场景物体跟踪paper](http://www.cvlibs.net/publications/Menze2015CVPR.pdf)

[kitti双目数据集解决方案](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)

[将深度神经网络用于将2D电影转换为3D电影的转换](https://github.com/piiswrong/deep3d)

[神经网络　双目匹配](https://github.com/jzbontar/mc-cnn)


[中山大学张弛博士](http://chizhang.me/)

    MeshStereo: A Global Stereo Model with Mesh Alignment Regularization for View Interpolation
    1、讨论了立体视觉匹配（Stereo Matching）问题的定义及其与人眼感知深度的关系；
    2、对Matching Cost Volume进行了可视化分析，以期望达到听者对其的直观且本质的理解；
    3、讨论了立体视觉匹配问题中的四个经典方法（
        Graph Cut，Adaptive Support Weight Aggregation, 
        Semi-Global Matching, 
        以及 PatchMatch Stereo）；
    4、讨论了MeshStereo的试图统一disparity求解以及网格生成两个步骤的motivation，
        以及formulate这样一个unified model会遇到的困难；
    5、讨论了MeshStereo引入splitting probability的解决方案及其优化过程。
    
    Webinar最后展示了MeshStereo在深度估计以及新视角渲染两个任务中的结果。


[Stereo Matching Using Tree Filtering non-local算法在双目立体匹配上的应用 ](https://blog.csdn.net/wsj998689aa/article/details/45584725)




### 2.g 深度匹配 depth matching 
[深度匹配 depth matching ](http://www.ipcv.org/on-depth-matching/)

### 2.h 姿态跟踪 Pose tracking 
[参考](http://www.ipcv.org/code-posetracking/)

### 2.i 物体跟踪 Object tracking
[参考](http://www.ipcv.org/code-objtracking/)

### 2.j 群体分析 Crowd analysis
[参考](http://www.ipcv.org/code-crowdanalysis/)
[群体运动度量](https://github.com/metalbubble/collectiveness)

### 2.k 光流场跟踪 Optical flow
[参考](http://www.ipcv.org/code-opticalflow/)


## 3 语义/实例分割&解析　Segmentation & Parsing
[参考](http://www.ipcv.org/category/code-data/parsing/)
### 3.a 视频分割  Video  segmentation
[参考](http://www.ipcv.org/video-segmentation/)

### 3.b 人体解析  Person parsing
[参考](http://www.ipcv.org/code-poseparsing/)

[person parsing](http://www.ipcv.org/about-person-parsing/)

### 3.c 场景解析  Scene  parsing
[scene parsing](http://www.ipcv.org/about-scene-parsing/)
[参考](http://www.ipcv.org/code-sceneparsing/)

### 3.d 边缘检测  Edge   detection
[参考](http://www.ipcv.org/code-edgedetection/)
[边缘检测](http://www.ipcv.org/on-edge-detection/)


### 3.e 图像物体分割 Image object segmentation 
[参考](http://www.ipcv.org/code-imobjseg/)
 
### 3.f 视频物体分割 Video object segmentation
[参考](http://www.ipcv.org/code-viobseg/)
[object segmentation](http://www.ipcv.org/about-video-object-segmentation/)


### 3.g 交互式分割   Interactive segmentation
[参考](http://www.ipcv.org/code-intseg/)

### 3.h 共分割      Co-segmentation
[参考](http://www.ipcv.org/code-cosegmentation/)

### 3.i 背景差      Background subtraction 
[参考](http://www.ipcv.org/code-backsub/)

### 3.j 图像分割方面 Image segmentation
[参考](http://www.ipcv.org/code-imgseg/)
 

## 4 识别/检测　Recognition & Detection
[参考](http://www.ipcv.org/category/code-data/detect/)
###  4.a 其他识别 Other recognition
[参考](http://www.ipcv.org/otherrecog/)

### 4.b 图像检索 Image retrieval
[参考](http://www.ipcv.org/%e5%9b%be%e5%83%8f%e6%a3%80%e7%b4%a2/)

### 4.c 显著检测 Saliency detection
[参考](http://www.ipcv.org/saldetection/)

### 4.d 通用物体检测 Object proposal
[参考](http://www.ipcv.org/code-objproposal/)

### 4.e 行为识别 Action recognition
[参考](http://www.ipcv.org/code-actionrecogntion/)

### 4.f 物体识别 Object recognition
[参考](http://www.ipcv.org/code-objrecogntion/)

### 4.g 行人检测 Human detection
[参考](http://www.ipcv.org/code-humandetection/)

### 4.h 人脸解析 Face Parsing
[参考](http://www.ipcv.org/code-facerecog/)

### 4.i 纹理分析 Texture Analysis
[纹理分析 Texture Analysis](http://www.ipcv.org/on-texture-analysis/)
[相关人物](http://www.ipcv.org/people-reidentity/)

## 5 机器学习 Maching Learning
[参考](http://www.ipcv.org/category/code-data/ml/)

### 5.a 生成对抗网络 GAN Generative Adversarial Networks
[参考](http://www.ipcv.org/adversarial-networks/)

### 5.b 深度学习    Deep learning 
[参考](http://www.ipcv.org/deeplearning/)
[深度学习方面的部分课程讲义](http://www.ipcv.org/lecture-deeplearning/)

[CNN Models 卷积网络模型](http://www.ipcv.org/on-object-detection/)

[Deep Learning Libraries　深度学习软件库](http://www.ipcv.org/deep_learning_libraries/)

[深度学习方面的部分视觉人物](http://www.ipcv.org/dl-researcher/)

### 5.c 能量优化    Energy optimization 
[参考](http://www.ipcv.org/energyopt/)

### 5.d 模型设计    Model design
[参考](http://www.ipcv.org/modelbuilding/)

### 5.e 空间降维    Dimention reduction 
[参考](http://www.ipcv.org/dimention/)

### 5.f 聚类       Clustering 
[参考](http://www.ipcv.org/clustering/)

### 5.g 分类器     Classifier
[参考](http://www.ipcv.org/classifier/)

## 6 开源库　Open library
[参考](http://www.ipcv.org/category/code-data/lib/)
###
###


## 7 数据集　Public dataset
[参考](http://www.ipcv.org/category/code-data/dataset/)
### 7.a 其他方面 Other datasets
[参考](http://www.ipcv.org/otherdb/)
[人脸检测Face tracking and recognition database ](http://seqam.rutgers.edu/site/index.php?option=com_content&view=article&id=65&Itemid=76)

[人脸检测Caltech 10,000 Web Faces](http://vision.caltech.edu/archive.html)

[人脸检测Helen dataset](http://www.ifp.illinois.edu/~vuongle2/helen/)

[深度图 RGB-D dataset](http://mobilerobotics.cs.washington.edu/projects/kdes/)

[视频分割 2010 ECCV Efficient Hierarchical Graph Based Video Segmentation](http://www.cc.gatech.edu/cpl/projects/videosegmentation/)

[手势跟踪 Hand dataset](https://engineering.purdue.edu/RVL/Database.html)

[手势跟踪2](http://www.robots.ox.ac.uk/~vgg/research/hands/index.html)

[车辆检测 2002 ECCV Learning a sparse representation for object detection](http://cogcomp.cs.illinois.edu/Data/Car/)

### 7.b 人体检测 dataset on human annotation
[参考](http://www.ipcv.org/humandetection/)
[Caltech Pedestrian Detection Benchmark](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/)

[Ethz](http://www.vision.ee.ethz.ch/~aess/dataset/)
[bleibe](http://www.vision.ee.ethz.ch/~bleibe/data/datasets.html)

[RGB-D People Dataset](http://www.informatik.uni-freiburg.de/~spinello/RGBD-dataset.html)

[TUD Campus](http://www.d2.mpi-inf.mpg.de/tud-brussels)
[382](https://www.d2.mpi-inf.mpg.de/node/382)

[PSU HUB Dataset](http://www.cse.psu.edu/~rcollins/software.html)

[Pedestrian parsing](http://vision.ics.uci.edu/datasets/)

[Human Eva](http://vision.cs.brown.edu/humaneva/)


### 7.c 物体识别 dataset on object recognition

[参考](http://www.ipcv.org/objectrecognition/)

[ e-Lab Video Data Set](https://engineering.purdue.edu/elab/eVDS/)

[Image Net](http://www.image-net.org/)

[Places2 Database](http://places2.csail.mit.edu)

[Microsoft CoCo: Common Objects in Context](http://mscoco.org/)

[PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)
[MIT’s Place2]( http://places2.csail.mit.edu/)

### 7.d 显著检测方面 dataset on saliency detection
[参考](http://www.ipcv.org/saliencydetection/)

[视觉显著性检测技术发展情况](http://blog.csdn.net/anshan1984/article/details/8657176)

[2012 ECCV Salient Objects Dataset (SOD)](http://elderlab.yorku.ca/SOD/)

[2012 ECCV Neil D. B. Bruce Eye Tracking Data](http://cs.umanitoba.ca/~bruce/datacode.html)

[2012 ECCV DOVES:A database of visual eye movements](http://live.ece.utexas.edu/research/doves/)

[2012 ECCV MSRA:Salient Object Database](http://research.microsoft.com/en-us/um/people/jiansun/SalientObject/salient_object.htm)

[2012 ECCV NUS: Predicting Saliency Beyond Pixels](http://www.ece.nus.edu.sg/stfpage/eleqiz/predicting.html)

[2012 ECCV saliency benchmark](http://people.csail.mit.edu/tjudd/SaliencyBenchmark/index.html)

[2010 ECCV The DUT-OMRON Image Dataset](http://ice.dlut.edu.cn/lu/DUT-OMRON/Homepage.htm)

[2010 ECCV An eye fixation database for saliency detection in images](http://mmas.comp.nus.edu.sg/NUSEF.html)

### 7.e 行为识别 dataset on action recognition
[参考](http://www.ipcv.org/actionrecognition/)

[UCF](http://www.cs.ucf.edu/~liujg/YouTube_Action_dataset.html)
[ChaoticInvariants](http://www.cs.ucf.edu/~sali/Projects/ChaoticInvariants/index.html)
[datasetsActions](http://vision.eecs.ucf.edu/datasetsActions.html)

[Hollywood Human Actions dataset](http://www.di.ens.fr/~laptev/download.html)
[data](http://lear.inrialpes.fr/data)

[Weizmann: Actionsas Space-Time Shapes](http://www.wisdom.weizmann.ac.il/~vision/SpaceTimeActions.html)

[KTH](http://www.nada.kth.se/cvap/actions/)

[UMD](http://www.umiacs.umd.edu/~zhuolin/Keckgesturedataset.html)

[HMDB: A Large Video Database for Human Motion Recognition](http://serre-lab.clps.brown.edu/resources/HMDB/related_data/)

[Collective Activity Dataset](http://www.eecs.umich.edu/vision/activity-dataset.html)

[MSR Action Recognition Datasets and Codes](http://research.microsoft.com/en-us/um/people/zliu/ActionRecoRsrc/default.htm)

[Visual Event Recognition in Videos](http://vc.sce.ntu.edu.sg/index_files/VisualEventRecognition/VisualEventRecognition.html)



### 7.f 物体分割 dataset on object segmentation
[参考](http://www.ipcv.org/objectsegmentation/)
[microsoft MSRC-V2](http://research.microsoft.com/en-us/projects/objectclassrecognition/)

[2010 CVPR iCoseg: Interactive cosegmentation by touch](http://chenlab.ece.cornell.edu/projects/touch-coseg/)

[2010 CVPR Caltech-UCSD Birds 200](http://www.vision.caltech.edu/visipedia/CUB-200.html)

[2010 CVPR Flower Datasets](http://www.robots.ox.ac.uk/~vgg/data/flowers/)

[2009 ICCV An efficient algorithm for co-segmentation](http://www.biostat.wisc.edu/~vsingh/)

[2008 CVPR Unsupervised Learning of Probabilistic Object Models (POMs) for Object Classification, Segmentation and Recognition](http://people.csail.mit.edu/leozhu/)

[2008 CVPR Caltech101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/)

[2004 ECCV The Weizmann Horse Database](http://www.msri.org/people/members/eranb/)


### 7.g 场景解析 dataset on scene parsing
[参考](http://www.ipcv.org/sceneparsing/)

[ImageNet](http://www.image-net.org/)

[ADE 20k](http://sceneparsing.csail.mit.edu/)

[Cityscapes](https://www.cityscapes-dataset.com/)

[COCO](http://cocodataset.org/#home)

[Lab, Koch](http://www.mis.tu-darmstadt.de/tudds)

[uiuc, D hoiem](http://www.cs.illinois.edu/homes/dhoiem/)

[mit, cbcl](http://cbcl.mit.edu/software-datasets/streetscenes/)

[mit LabelMeVideo](http://labelme.csail.mit.edu/LabelMeVideo/)

[2013 BMVC Hierarchical Scene Annotation](http://www.vision.caltech.edu/~mmaire/)

[2010 ECCV SuperParsing: Scalable Nonparametric Image Parsing with Superpixels](http://www.cs.unc.edu/~jtighe/Papers/ECCV10/)

[2009 CVPR Nonparametric Scene Parsing: Label Transfer via Dense Scene Alignment](ttp://people.csail.mit.edu/celiu/CVPR2009/)

[2009 Scene Understanding Datasets](http://dags.stanford.edu/projects/scenedataset.html)

[2008 IJCV 6D-Vision](http://www.6d-vision.com/scene-labeling)

[2008 IJCV The Daimler Urban Segmentation Dataset](http://www.6d-vision.com/scene-labeling)

[2008 ECCV Motion-based Segmentation and Recognition Dataset](http://mi.eng.cam.ac.uk/research/projects/VideoRec))/CamVid/

[2008 York Urban Dataset](http://www.elderlab.yorku.ca/YorkUrbanDB/)

[2008 IJCV LabelMe](http://labelme.csail.mit.edu/LabelMeToolbox/index.html)


## 8 会议　期刊　
### CVPR Computer vision  and  Pattern Reconition 计算机视觉和模式识别
### ECCV European Conference on Computer Vision   欧洲计算机视觉国际会议 
### ICCV IEEE International Conference on Computer Vision  国际计算机视觉大会
### 其他
[其他](http://www.ipcv.org/otherpaper/)
###
###
###
###


## AR&VR
[参考](http://www.ipcv.org/category/top-dir/arvr/)


## 
