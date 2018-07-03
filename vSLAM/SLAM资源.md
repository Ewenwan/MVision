# SLAM 学习与开发经验分享![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)

### 导语
> 毫无疑问，SLAM是当前最酷炫的技术.在这里，我给大家分享一些在学习SLAM的过程中的一些资料与方法（不断更新中...）</br></br>



## 目录
* **[入门](#入门)**

* **[基础](#基础)**

* **[进阶](#进阶)**

* **[优秀文章](#优秀文章)**

* **[技术博客](#技术博客)**

* **[网站与研究组织](#网站与研究组织)**

* **[书籍与资料](#书籍与资料)**

* **[SLAM 方案](#SLAM方案)**

* **[优秀案例](#优秀案例)**

* **[泡泡机器人专栏公开课](#泡泡机器人专栏公开课)**

## 入门

[视觉SLAM的基础知识-高翔](http://www.bilibili.com/video/av5911960/?from=search&seid=375069506917728550)(高翔博士)-----视频，吐血推荐

[关于slam](http://blog.csdn.net/yimingsilence/article/details/51701944)

[SLAM简介](https://zhuanlan.zhihu.com/p/21381490)

[SLAM第一篇：基础知识](http://www.leiphone.com/news/201609/iAe3f8qmRHXavgSl.html?viewType=weixin)

[SLAM_介绍以及浅析](https://wenku.baidu.com/view/905bf05d312b3169a451a495.html)

[SLAM的前世今生 终于有人说清楚了](http://www.leiphone.com/news/201605/5etiwlnkWnx7x0zb.html)(张一茗)

[SLAM for Dummies](https://pan.baidu.com/s/1dFxKLZb)（一本关于实时定位及绘图（SLAM）的入门指导教程）    提取码：k3r3

[STATE ESTIMATION FOR ROBOTICS](https://pan.baidu.com/s/1pLO4Nwv) (吐血推荐)  提取码：y7tc

## 基础
数学基础

- [计算机视觉中的数学方法](https://pan.baidu.com/s/1pLwK4uJ) 提取码：afyg  ----本书着重介绍射影几何学及其在视觉中的应用

- [视觉SLAM中的数学基础 第一篇](http://www.cnblogs.com/gaoxiang12/p/5113334.html)----3D空间的位置表示

- [视觉SLAM中的数学基础 第二篇](http://www.cnblogs.com/gaoxiang12/p/5120175.html)----四元数

- [视觉SLAM中的数学基础 第三篇](http://www.cnblogs.com/gaoxiang12/p/5137454.html)----李群与李代数

- [李群和李代数](https://pan.baidu.com/s/1eRUC3ke) 提取码：92x2

语言编程基础

- [菜鸟教程](http://www.runoob.com/)----学习C++与python基础语法

- [python计算机视觉编程](https://pan.baidu.com/s/1bpcQlvp) 提取码：kyt9

- [OpenCV3编程入门_毛星云编著](https://pan.baidu.com/s/1i4Gtv3B)----C++实现   提取码：qnms

计算机视觉基础

- [计算机视觉算法与应用中文版](https://pan.baidu.com/s/1dFHxDiL) 提取码：b8y1

- [特征提取与图像处理](https://pan.baidu.com/s/1nvseOf3)   提取码：hgy2

- [机器视觉算法与应用](https://pan.baidu.com/s/1i4O0LOp)  提取码：hxgn


泡泡机器人SLAM 优质视频课程----视觉slam十四讲

- [视觉slam十四讲1-2 引言与概述](http://www.bilibili.com/video/av7494417/?from=search&seid=375069506917728550)

- [视觉SLAM十四讲（第三章）](http://www.bilibili.com/video/av7612959/?from=search&seid=375069506917728550)

- [视觉slam第4章](http://www.bilibili.com/video/av7705856/?from=search&seid=375069506917728550)

- [视觉SLAM十四讲-第五章-相机与图像](http://www.bilibili.com/video/av7816357/?from=search&seid=375069506917728550)

- [视觉SLAM十四讲-第六章-非线性优化](http://www.bilibili.com/video/av7921657/?from=search&seid=375069506917728550)

- [视觉SLAM十四讲-第七章-视觉里程计一](http://www.bilibili.com/video/av8061127/?from=search&seid=375069506917728550)

Python + SLAM

- [用python学习slam系列（一）从图像到点云](http://www.rosclub.cn/post-682.html)

- [用python学习slam系列（二）特征提取与配准](http://www.rosclub.cn/post-684.html)





## 进阶
一步步实现SLAM系列教程

- [一步步实现slam1-项目框架搭建](http://fengbing.net/2016/02/03/%E4%B8%80%E6%AD%A5%E6%AD%A5%E5%AE%9E%E7%8E%B0slam1-%E6%A1%86%E6%9E%B6%E6%90%AD%E5%BB%BA/)

- [一步步实现slam2-orb特征检测](http://fengbing.net/2016/04/03/%E4%B8%80%E6%AD%A5%E6%AD%A5%E5%AE%9E%E7%8E%B0slam2-orb%E7%89%B9%E5%BE%81%E6%A3%80%E6%B5%8B/)

- [一步步实现slam3-初始位置估计1](http://fengbing.net/2016/04/23/%E4%B8%80%E6%AD%A5%E6%AD%A5%E5%AE%9E%E7%8E%B0slam3-%E5%88%9D%E5%A7%8B%E4%BD%8D%E5%A7%BF%E4%BC%B0%E8%AE%A11/)

- [一步步实现slam3-初始位置估计2](http://fengbing.net/2016/04/24/%E4%B8%80%E6%AD%A5%E6%AD%A5%E5%AE%9E%E7%8E%B0slam3-%E5%88%9D%E5%A7%8B%E4%BD%8D%E5%A7%BF%E4%BC%B0%E8%AE%A12/)


[SLAM最终话：视觉里程计](http://www.leiphone.com/news/201609/Qj6uJhaywpBD8vdq.html)(高翔博士)

[双目视觉里程计](http://www.bilibili.com/video/av5913124/)(谢晓佳-视频)

[视觉SLAM中的矩阵李群基础](http://www.bilibili.com/video/av6069884/)(王京-视频)

[路径规划](http://www.bilibili.com/video/av6640192/)(王超群-视频)

[优化与求解](http://www.bilibili.com/video/av6298224/)(刘毅-视频)

[直接法的原理与实现](http://www.bilibili.com/video/av6299156/)(高翔-视频)

[Course on SLAM](https://pan.baidu.com/s/1miuOIUW) 提取码：i94s

[LM算法计算单应矩阵](https://mp.weixin.qq.com/s?__biz=MzI5MTM1MTQwMw==&mid=2247484602&idx=1&sn=b6d4b9d24af02a1e8de7f30e7f17f525&chksm=ec10babedb6733a8c5b4ff4877b250394797efff616647778885515991daf25e27f2cd0f9914&mpshare=1&scene=24&srcid=0323y0KxEHYSArWp9ldTTSOi&key=0722bcaff7a71b24fa77c83e37758d99034902520ffb87ae1ceeaed5271181bc7a04c3cab7ed2588574061afc774ff7c19c8538b91f113a45b4f7edb666ccdff2b6a60c60746cff9ef32d749f0f7e87f&ascene=0&uin=NTA0OTM2NzY%3D&devicetype=iMac+MacBookAir7%2C2+OSX+OSX+10.12+build(16A323)&version=12020010&nettype=WIFI&fontScale=100&pass_ticket=XHDonKi50slr29BQ9oY9LM0lAhnHy33o2h%2Fz2ho0874%3D)

[激光SLAM](https://mp.weixin.qq.com/s?__biz=MzI5MTM1MTQwMw==&mid=2247484587&idx=1&sn=82c66613817bd6f50f58cd309dc48a6a&chksm=ec10baafdb6733b9ce69ef19f27087ab8ebfc40d479fc5f0469c301d414c3745d6abb02e2d8b&mpshare=1&scene=24&srcid=0323ANeCiITgcqDmjeIQZ7Li&key=a4c24fb23da90f2ca36bf82f0a48f6a72ddb40abe90030450b2d73544285ef7c297334f0b202b66bddc6aee8ff556fa9c0ac3d4178332056bea0f16171e090921d01980d4007b6102671f14f8b1af8e9&ascene=0&uin=NTA0OTM2NzY%3D&devicetype=iMac+MacBookAir7%2C2+OSX+OSX+10.12+build(16A323)&version=12020010&nettype=WIFI&fontScale=100&pass_ticket=XHDonKi50slr29BQ9oY9LM0lAhnHy33o2h%2Fz2ho0874%3D)(王龙军)

[我们如何定位SLAM？——关于技术创新、产品开发和管理的经验和教训](https://mp.weixin.qq.com/s?__biz=MzI5MTM1MTQwMw==&mid=2247484979&idx=1&sn=3d4af432a50842360f31561dbabfa58b&chksm=ec10b837db673121941d8c3e1c8492238f6922b3223b397b399f0066be03171a9146d6c0c939&mpshare=1&scene=24&srcid=0323oLiJv6WiRX2odtfQDXPe&key=a4c24fb23da90f2ca935bf18db801448a4369496fff7ca8e359883d8b14cc7b482b81a7bc8d50d9165bb8e2a99056a1b79bc3e423cfdd735279a28883133cf50a7ae158b88b39b1a33477e71788b8e5b&ascene=0&uin=NTA0OTM2NzY%3D&devicetype=iMac+MacBookAir7%2C2+OSX+OSX+10.12+build(16A323)&version=12020010&nettype=WIFI&fontScale=100&pass_ticket=XHDonKi50slr29BQ9oY9LM0lAhnHy33o2h%2Fz2ho0874%3D)

[语义SLAM的未来与思考(1)](https://mp.weixin.qq.com/s?__biz=MzI5MTM1MTQwMw==&mid=2247485956&idx=1&sn=3d95d417c7a1446a90276473ec736f0b&chksm=ec10b400db673d168f0c34245c58f9c6fbf339be3366689537468fa5ea84e92a964c92dc8be3&mpshare=1&scene=24&srcid=0323LglzpLJ5ftLN1isUQEc7&key=04a263dc798cf3e2798bd64119c159030a49ead5db2e70090701706793c09e38d26018f0415fe1d986020f20cd29987e51c790ceee046946112b55ded02fc28ccdc2392550a895bf8c1a8515cdc87fb9&ascene=0&uin=NTA0OTM2NzY%3D&devicetype=iMac+MacBookAir7%2C2+OSX+OSX+10.12+build(16A323)&version=12020010&nettype=WIFI&fontScale=100&pass_ticket=XHDonKi50slr29BQ9oY9LM0lAhnHy33o2h%2Fz2ho0874%3D)

[语义SLAM的未来与思考(2)](https://mp.weixin.qq.com/s?__biz=MzI5MTM1MTQwMw==&mid=2247486496&idx=1&sn=672421edb67a7566a8d3629596f12496&chksm=ec10b224db673b3245d7c49ac07b9e7e924b07d93992c0a4392649624178a614eb084eee5cd7&mpshare=1&scene=24&srcid=0323rFVB2RhlBb8YrM7zYgyo&key=405f89b14d07d74b9f658163a8b0fb12e76b79928dfd59dd093a011fc98c93a37d0b489893c3d90a6e2e5b8516f05b0443ac35204d4c2e3c1f28398ff9d48148aa69f818c141f08503c15445b4af2375&ascene=0&uin=NTA0OTM2NzY%3D&devicetype=iMac+MacBookAir7%2C2+OSX+OSX+10.12+build(16A323)&version=12020010&nettype=WIFI&fontScale=100&pass_ticket=XHDonKi50slr29BQ9oY9LM0lAhnHy33o2h%2Fz2ho0874%3D)

## 优秀文章
泡泡机器人SLAM优秀技术文章

- [SLAM: 现在，未来和鲁棒年代（一）](http://weixin.niurenqushi.com/article/2017-02-12/4766381.html)

- [SLAM: 现在，未来和鲁棒年代（二）](http://weixin.niurenqushi.com/article/2017-02-13/4767586.html)

- [SLAM: 现在，未来和鲁棒年代（三）](http://weixin.niurenqushi.com/article/2017-02-14/4768761.html)

- [SLAM: 现在，未来和鲁棒年代（四）](https://mp.weixin.qq.com/s?__biz=MzI5MTM1MTQwMw==&mid=2247488263&idx=1&sn=1c10dbf32bfd90c3587988652df6187d&chksm=ec10ad03db672415dcea09f62f2341fd330895c71c77b04a92d6be3e3f0274eb2771a07354dd&mpshare=1&scene=24&srcid=0323gxrX4cn4zd4zY4dZcIcd&key=3c2da574259c4c3bab55c1c5c0eab459ad4b990e6130aed37d181792a2918d8c8ab73d84eb0872e0631e5c953289d05de5e32588c28a8c899225e45087d67cb4577a806dfce9d5b23eb5f099c7cc8af4&ascene=0&uin=NTA0OTM2NzY%3D&devicetype=iMac+MacBookAir7%2C2+OSX+OSX+10.12+build(16A323)&version=12020010&nettype=WIFI&fontScale=100&pass_ticket=BNJCJrAc4xlKLazxgWI8g7b%2BvcGziaU7%2Bs7XLYj8ASQ%3D)

- [SLAM: 现在，未来和鲁棒年代（五）](http://weixin.niurenqushi.com/article/2017-02-25/4779559.html)

[SLAM刚刚开始的未来](http://weibo.com/ttarticle/p/show?id=2309403994589869514382&mod=zwenzhang)(张哲)

[2D Slam与3D SLam 的区别到底在哪里](http://www.arjiang.com/index.php?m=content&c=index&a=show&catid=11&id=418)

[研究SLAM，对编程的要求有多高？](https://www.zhihu.com/question/51707998)

[SLAM在VR/AR领域重要吗？](https://www.zhihu.com/question/37071486)

[单目SLAM在移动端应用的实现难点有哪些？](https://www.zhihu.com/question/50385799)

[机器人的双眸：视觉SLAM是如何实现的？](http://www.leiphone.com/news/201607/GLQj0wrjKD4eHvq5.html)

[牛逼哄哄的SLAM技术 即将颠覆哪些领域？](http://www.leiphone.com/news/201605/oj1lxZVPulRdNxYt.html)



## 技术博客
[半闲居士](http://www.cnblogs.com/gaoxiang12)----高翔博士的SLAM博客（力推）

- [SLAM拾萃(1)：octomap](http://www.cnblogs.com/gaoxiang12/p/5041142.html)

- [视觉SLAM漫谈（二）:图优化理论与g2o的使用](http://www.cnblogs.com/gaoxiang12/p/3776107.html)

[白巧克力亦唯心](http://blog.csdn.net/heyijia0327)

- [graph slam tutorial : 从推导到应用1](http://blog.csdn.net/heyijia0327/article/details/47686523)

- [graph slam tutorial ：从推导到应用２](http://blog.csdn.net/heyijia0327/article/details/47731631)

[冯兵](http://fengbing.net/)

- [视觉里程计简介](http://fengbing.net/2015/07/25/%E8%A7%86%E8%A7%89%E9%87%8C%E7%A8%8B%E8%AE%A1%E7%AE%80%E4%BB%8B/)

- [视觉里程计总介绍](http://fengbing.net/2015/08/01/%E8%A7%86%E8%A7%89%E9%87%8C%E7%A8%8B%E8%AE%A1%E6%80%BB%E4%BB%8B%E7%BB%8D/)

[hitcm](http://www.cnblogs.com/hitcm/)

- [ROS实时采集Android的图像和IMU数据](http://www.cnblogs.com/hitcm/p/5616364.html)

- [基于点线特征的Kinect2实时环境重建（Tracking and Mapping）](http://www.cnblogs.com/hitcm/p/5245463.html)

[何必浓墨重彩](http://blog.csdn.net/wendox/article/category/6555599)

- [SLAM代码（优化及常用库）](http://blog.csdn.net/wendox/article/details/52507220)

- [SLAM代码（多视几何基础)](http://blog.csdn.net/wendox/article/details/52552286)

- [SLAM代码（三维重建）](http://blog.csdn.net/wendox/article/details/52719252)

- [SLAM代码（设计模式）](http://blog.csdn.net/wendox/article/details/53454768)

- [SLAM代码（设计模式2）](http://blog.csdn.net/wendox/article/details/53489982)

[路游侠](http://www.cnblogs.com/luyb/)

- [AR中的SLAM](http://www.cnblogs.com/luyb/p/6481725.html)

- [SVO原理解析](http://www.cnblogs.com/luyb/p/5773691.html)

[电脑线圈](https://zhuanlan.zhihu.com/computercoil)

[wishchin](http://blog.csdn.net/wishchin/article/category/5723249/2)



## 网站与研究组织
[泡泡机器人](http://space.bilibili.com/38737757/#!/)----泡泡机器人是中国SLAM研究爱好者自发组成的团体，在自愿条件下分享SLAM相关知识，为推动国内SLAM研究做出一点小小的贡献。

![](paopao.png)

[ROSClub机器人俱乐部](http://www.rosclub.cn/cate-9.html)----这是一个完美的ROS机器人开发交流社区平台，在这里你可以找到你所要的！

[SLAMCN](http://www.slamcn.org/index.php/%E9%A6%96%E9%A1%B5)-----SLAM精选国内外学习资源

[openslam.org](http://openslam.org/)--A good collection of open source code and explanations of SLAM.(推荐)

[易科机器人实验室](http://blog.exbot.net/)

[电子发烧友--SLAM](http://www.elecfans.com/tags/slam/)

[电子工程世界--SLAM](http://www.eeworld.com.cn/tags/SLAM)

[Jianxiong Xiao (Professor X)](http:/vision.princeton.edu/people/xj/)---从事cv dl与slam相结合的多项研究.

[Robot Mapping](http://ais.informatik.uni-freiburg.de/teaching/ws13/mapping/)

[Chuck Palumbo](http://www.discovery.com/tv-shows/rusted-development/hosts/chuck-palumbo/)----需要翻墙

[Alexander Grau's blog](http://grauonline.de/wordpress/)----博客里有很多关于机器人, SLAM, 传感器等技术

[Andrew Davison: Research](https://www.doc.ic.ac.uk/~ajd/index.html)

[Autonome and Perceptive Systemen](http://www.ai.rug.nl/~gert/as/)---research page at University of Groningen about visual SLAM.

[SLAM Tutorial@ICRA 2016](http://www.dis.uniroma1.it/~labrococo/tutorial_icra_2016/)

[SLAM Summer School](http://www.acfr.usyd.edu.au/education/summerschool.shtml)----https://github.com/kanster/awesome-slam#courses-lectures-and-workshops

[autoloc](http://webdiis.unizar.es/~neira/slam.html)

## 书籍与资料
[工业机器人视觉系统组成及介绍](https://pan.baidu.com/s/1gf1KEdX)(fu78)

[深度学习](https://pan.baidu.com/s/1jIKBwSu)（zyn2）

[svo_lsd](https://pan.baidu.com/s/1kV5apsN)(uyce)

[MEMS IMU的入门与应用 - 胡佳兴](https://pan.baidu.com/s/1o8M5fSY)(vt6m)

[双目视觉里程计](https://pan.baidu.com/s/1nvv2M6l)(27jk)

[高精度实时视觉定位的关键技术研究](https://pan.baidu.com/s/1dEJj1Ln)(yrct)

[ORB-SLAM2源码详解](https://pan.baidu.com/s/1ceoq78)(su4r)

[ORB-SLAM2源码详解-补充H矩阵分解](https://pan.baidu.com/s/1mhVg2ta)(vjik)

[激光slam](https://pan.baidu.com/s/1i5QJeYx)(9dqd)

[图像特征的非刚性匹配](https://pan.baidu.com/s/1eRNRf1o)(w85h)



英文版（有些需翻墙）

- [Current trends in SLAM](http://webdiis.unizar.es/~neira/SLAM/SLAM_5_Trends.pdf)---关于DTAM,PTAM,SLAM++等系统的对比

- [The scaling problem](http://webdiis.unizar.es/~neira/SLAM/SLAM_4_Scaling.pptx.pdf)----针对SLAM计算量过大的问题进行讲解

- [slamtute1--The Essential Algorithms](http://www-personal.acfr.usyd.edu.au/tbailey/papers/slamtute1.pdf)

- [A random-finite-set approach to Bayesian SLAM](http://staffhome.ecm.uwa.edu.au/~00053612/vo/MVAV_SLAM11.pdf)

- [On the Representation and Estimation of Spatial Uncertainty](http://www.frc.ri.cmu.edu/~hpm/project.archive/reference.file/Smith&Cheeseman.pdf)

- [Past, Present, and Future of Simultaneous Localization And Mapping: Towards the Robust-Perception Age](https://arxiv.org/abs/1606.05830)(2016)

- [Direct Sparse Odometry](https://arxiv.org/abs/1607.02565)

- [Modelling Uncertainty in Deep Learning for Camera Relocalization](https://arxiv.org/abs/1509.05909)

- [Tree-connectivity: Evaluating the graphical structure of SLAM](http://ieeexplore.ieee.org/document/7487264/)

- [Multi-Level Mapping: Real-time Dense Monocular SLAM](https://groups.csail.mit.edu/rrg/papers/greene_icra16.pdf)

- [State Estimation for Robotic -- A Matrix Lie Group Approach ](http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser15.pdf)

- [Probabilistic Robotics](http://www.probabilistic-robotics.org/)----Dieter Fox, Sebastian Thrun, and Wolfram Burgard, 2005

- [Simultaneous Localization and Mapping for Mobile Robots: Introduction and Methods](http://www.igi-global.com/book/simultaneous-localization-mapping-mobile-robots/66380)

- [An Invitation to 3-D Vision -- from Images to Geometric Models](http://vision.ucla.edu/MASKS/)----Yi Ma, Stefano Soatto, Jana Kosecka and Shankar S. Sastry, 2005


## SLAM 方案
ORB-SLAM

- [ORB-SLAM（一）简介](http://www.cnblogs.com/luyb/p/5215168.html)

- [ORB-SLAM（二）性能](http://www.cnblogs.com/luyb/p/5240168.html)

- [ORB-SLAM（三）地图初始化](http://www.cnblogs.com/luyb/p/5260785.html)

- [ORB-SLAM（四）追踪](http://www.cnblogs.com/luyb/p/5357790.html)

- [ORB-SLAM（五）优化](http://www.cnblogs.com/luyb/p/5447497.html)

- [ORB-SLAM（六）回环检测](http://www.cnblogs.com/luyb/p/5599042.html)

- [ORB_SLAM 初接触](https://zhuanlan.zhihu.com/p/20589372)

- [ORB_SLAM 初接触2](https://zhuanlan.zhihu.com/p/20596486)

- [ORB_SLAM - 3：和markerless AR的结合](https://zhuanlan.zhihu.com/p/20599601)

- [ORB_SLAM - 4：卡片地图预先创建](https://zhuanlan.zhihu.com/p/20602540)

- [ORB_SLAM - 5：SLAM多目标添加](https://zhuanlan.zhihu.com/p/20608728)

- [运行ORB-SLAM笔记_编译篇（一）](http://www.cnblogs.com/li-yao7758258/p/5906447.html)----Being_young

- [运行ORB-SLAM笔记_使用篇（二）](http://www.cnblogs.com/li-yao7758258/p/5912663.html)----Being_young

- [ORB-SLAM](http://webdiis.unizar.es/~raulmur/orbslam/)----ORB-SLAM 官方网站

- [ORB-SLAM：精确多功能单目SLAM系统](http://qiqitek.com/blog/?p=13)----中文翻译

- [视觉SLAM实战：ORB-SLAM2 with Kinect2](http://www.cnblogs.com/gaoxiang12/p/5161223.html)

- [ORB-SLAM--- 让程序飞起来](http://blog.csdn.net/dourenyin/article/details/48055441)

- [ORB-SLAM 笔记](http://blog.csdn.net/fuxingyin/article/details/53511439)



RGB-SLAM

- [视觉SLAM实战（一）：RGB-D SLAM V2](http://www.cnblogs.com/gaoxiang12/p/4462518.html)

- [一起做RGB-D SLAM (2)](http://www.cnblogs.com/gaoxiang12/p/4652478.html)

- [一起做RGB-D SLAM (3)](http://www.cnblogs.com/gaoxiang12/p/4659805.html)

- [一起做RGB-D SLAM (4)](http://www.cnblogs.com/gaoxiang12/p/4669490.html)

- [一起做RGB-D SLAM (5)](http://www.cnblogs.com/gaoxiang12/p/4719156.html)

- [一起做RGB-D SLAM (6)](http://www.cnblogs.com/gaoxiang12/p/4739934.html)

- [一起做RGB-D SLAM(7) （完结篇）](http://www.cnblogs.com/gaoxiang12/p/4754948.html)

- [一起做RGB-D SLAM(8) （关于调试与补充内容）](http://www.cnblogs.com/gaoxiang12/p/4770813.html)

- [一起做RGB-D SLAM (9)--问题总结](http://www.rosclub.cn/post-85.html)

PTAM单目

- [PTAM算法流程介绍](http://blog.csdn.net/zzzblog/article/details/14455463)

- [Parallel Tracking and Mapping for Small AR Workspaces](http://www.robots.ox.ac.uk/~gk/PTAM/)

- [PTAM跟踪过程中的旋转预测方法](https://zhuanlan.zhihu.com/p/20302059?refer=computercoil)

- [PTAM跟踪失败后的重定位](https://zhuanlan.zhihu.com/p/20308700)

LSD-SLAM

- [LSD-SLAM深入学习（1）-基本介绍与ros下的安装](http://www.cnblogs.com/hitcm/p/4907465.html)

- [LSD-SLAM深入学习（2）-算法解析](http://www.cnblogs.com/hitcm/p/4907536.html)

- [LSD-SLAM深入学习（3）-代码解析](http://www.cnblogs.com/hitcm/p/4887345.html)

- [LSD-SLAM: Large-Scale Direct Monocular SLAM](http://vision.in.tum.de/research/vslam/lsdslam)----Computer Vision Group

- [lsd-slam源码解读第一篇:Sophus/sophus](http://blog.csdn.net/lancelot_vim/article/details/51706832)

- [lsd-slam源码解读第二篇:DataStructures](http://blog.csdn.net/lancelot_vim/article/details/51708412)

- [lsd-slam源码解读第三篇:算法解析](http://blog.csdn.net/lancelot_vim/article/details/51730676)

- [lsd-slam源码解读第四篇:tracking](http://blog.csdn.net/lancelot_vim/article/details/51758870)

- [lsd-slam源码解读第五篇:DepthEstimation](http://blog.csdn.net/lancelot_vim/article/details/51789318)

- [sd-slam源码解读第六篇:GlobalMapping](http://blog.csdn.net/lancelot_vim/article/details/51812484)

DSO单目

- [DSO: Direct Sparse Odometry](http://vision.in.tum.de/research/vslam/dso)

- [DSO论文速递（一）----泡泡机器人](http://www.zglwfww.com/a/dajiadouzaikan/2016/1216/5775.html)

- [DSO论文速递（二）----泡泡机器人](http://diyitui.com/content-1482249289.66463991.html)

- [DSO论文速递（三）----泡泡机器人](http://diyitui.com/content-1482851742.66460941.html)

- [DSO 初探](http://blog.csdn.net/heyijia0327/article/details/53173146)----白巧克力亦唯心

- [基于视觉+惯性传感器的空间定位方法](https://zhuanlan.zhihu.com/p/24072804)

SVO单目

- [svo： semi-direct visual odometry 论文解析](http://blog.csdn.net/heyijia0327/article/details/51083398)

- [安装说明](https://github.com/uzh-rpg/rpg_svo/wiki)

- [安装SVO](http://blog.sina.com.cn/s/blog_7b83134b0102wfu4.html)

- [SLAM代码之svo代码分析](http://blog.csdn.net/wendox/article/details/52536706)

DTAM

- [OpenDTAM](https://github.com/anuranbaka/OpenDTAM)----An open source implementation of DTAM


## 优秀案例

[ORB_SLAM](https://github.com/raulmur/ORB_SLAM)----多功能和单目的SLAM框架

[ORB_SLAM2](https://github.com/raulmur/ORB_SLAM2)----Real-Time SLAM for Monocular, Stereo and RGB-D Cameras, with Loop Detection and Relocalization Capabilities

[LSD-SLAM](https://github.com/tum-vision/lsd_slam)----很有名的，不多解释了

[DVO-SLAM](https://github.com/tum-vision/dvo_slam)----Dense Visual Odometry and SLAM

[RGBD-SLAM2](https://github.com/felixendres/rgbdslam_v2)----RGB-D SLAM for ROS

[SVO](https://github.com/uzh-rpg/rpg_svo)----Semi-Direct Monocular Visual Odometry

[G2O](https://github.com/RainerKuemmerle/g2o)----A General Framework for Graph Optimization


[cartographer](https://github.com/googlecartographer/cartographer)----这是一个提供实时SLAM在2D和3D的跨多个平台和传感器配置。

[slambook](https://github.com/gaoxiang12/slambook)----高翔博士的SLAM书籍中的code

[slamhound](https://github.com/technomancy/slamhound)----Slamhound rips your namespace form apart and reconstructs it.

[ElasticFusion](https://github.com/mp3guy/ElasticFusion)----Real-time dense visual SLAM system

[ORB_SLAM_iOS](https://github.com/egoist-sx/ORB_SLAM_iOS)----ORB_SLAM for iOS

[ORB_SLAM2_Android](https://github.com/FangGet/ORB_SLAM2_Android)----a repository for ORB_SLAM2 in Android

## 泡泡机器人公开课
[【泡泡机器人公开课】第二课：深度学习及应用](http://www.rosclub.cn/post-212.html)

[【泡泡机器人公开课】第三课 SVO 和 LSD_SLAM解析](http://www.rosclub.cn/post-213.html)

[【泡泡机器人公开课】第四课：Caffe入门与应用 by 高翔](http://www.rosclub.cn/post-216.html)

[【泡泡机器人公开课】第五课：双目视觉里程计](http://www.rosclub.cn/post-217.html)

[【泡泡机器人公开课】第六课：比特币介绍 by 李其乐](http://www.rosclub.cn/post-218.html)

[【泡泡机器人公开课】第七课：增强现实及其应用](http://www.rosclub.cn/post-220.html)

[【泡泡机器人公开课】第八课：MEMS IMU的入门与应用](http://www.rosclub.cn/post-221.html)

[【泡泡机器人公开课】第九课 双目校正及视差图的计算](http://www.rosclub.cn/post-222.html)

[【泡泡机器人公开课】第十课 IMU+动态背景消除](http://www.rosclub.cn/post-223.html)

[【泡泡机器人公开课】第十一课：COP-SLAM by 杨俊](http://www.rosclub.cn/post-224.html)

[【泡泡机器人公开课】第十二课：SLAM综述ORB-LSD-SVO by 刘浩敏](http://www.rosclub.cn/post-225.html)

[【泡泡机器人公开课】第十三课：CUDA 优化代码 by 张也冬](http://www.rosclub.cn/post-226.html)

[【泡泡机器人公开课】第十四课：KinectFusion、ElasticFusion 论文和代码解析](http://www.rosclub.cn/post-227.html)

[【泡泡机器人公开课】第十五课：视觉SLAM中的矩阵李群基础](http://www.rosclub.cn/post-228.html)

[【泡泡机器人公开课】第十六课：rosbridge原理及应用](http://www.rosclub.cn/post-229.html)

[【泡泡机器人公开课】第十七课：SLAM 优化与求解](http://www.rosclub.cn/post-230.html)

[【泡泡机器人公开课】第十八课：Direct方法的原理与实现](http://www.rosclub.cn/post-231.html)

[【泡泡机器人公开课】第十九课：图像技术在AR中的实践](http://www.rosclub.cn/post-232.html)

[【泡泡机器人公开课】第二十课：路径规划](http://www.rosclub.cn/post-233.html)

[【泡泡机器人公开课】第二十一课：ORB-SLAM简单重构](http://www.rosclub.cn/post-234.html)

[【泡泡机器人公开课】第二十二课：LeastSquare_and_gps_fusion](http://www.rosclub.cn/post-235.html)

[【泡泡机器人公开课】第二十三课：Scan Matching in 2D SLAM](http://www.rosclub.cn/post-236.html)

[【泡泡机器人公开课】第二十四课：LSD-SLAM深度解析](http://www.rosclub.cn/post-237.html)

[【泡泡机器人公开课】第二十五课：激光SLAM](http://www.rosclub.cn/post-238.html)

[【泡泡机器人公开课】第二十六课：TSL安全网络传输协议简介](http://www.rosclub.cn/post-240.html)

[【泡泡机器人公开课】第二十七课：Textureless Object Tracking](http://www.rosclub.cn/post-242.html)

[【泡泡机器人公开课】第二十八课：基于光流的视觉控制](http://www.rosclub.cn/post-243.html)

[【泡泡机器人公开课】第二十九课：Robust Camera Location Estimation](http://www.rosclub.cn/post-244.html)

[【泡泡机器人公开课】第三十课:非线性优化与g2o](http://www.rosclub.cn/post-245.html)

[【泡泡机器人公开课】第三十一课：G2O简介](http://www.rosclub.cn/post-247.html)

[【泡泡机器人公开课】第三十二课：我们如何定位SLAM？](http://www.rosclub.cn/post-499.html)

[【泡泡机器人公开课】第三十三课：矩阵流形上的优化介绍](http://www.rosclub.cn/post-500.html)

[【泡泡机器人公开课】第三十四课：里程计-视觉融合SLAM](http://www.rosclub.cn/post-502.html)

[【泡泡机器人公开课】第三十五课：Visualization in SLAM](http://www.rosclub.cn/post-504.html)

[【泡泡机器人公开课】第三十六课：ORB-SLAM2源码详解](http://www.rosclub.cn/post-505.html)

[【泡泡机器人公开课】第三十七课：Absolute Scale Estimation and Correction](http://www.rosclub.cn/post-517.html)

[【泡泡机器人公开课】第三十八课：Structure Light Based3D Surface Imaging](http://www.rosclub.cn/post-564.html)

[【泡泡机器人公开课】第三十九课：PnP)算法简介与代码解析](http://www.rosclub.cn/post-566.html)


#不断更新中...

AR开发者社区：

![](WeChat.jpg)

