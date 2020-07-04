# 无人驾驶
![](https://github.com/Ewenwan/MVision/blob/master/UMCar/img/sdc.PNG)

[百度apollo课程](https://www.bilibili.com/video/BV1yJ411d7xu/?spm_id_from=333.788.videocard.16)

当今，自动驾驶技术已经成为整个汽车产业的最新发展方向。应用自动驾驶技术可以全面提升汽车驾驶的安全性、舒适性，满足更高层次的市场需求等。自动驾驶技术得益于人工智能技术的应用及推广，在环境感知、精准定位、决策与规划、控制与执行、高精地图与车联网V2X 等方面实现了全面提升。科研院校、汽车制造厂商、科技公司、自动驾驶汽车创业公司以及汽车零部件供应商在自动驾驶技术领域进行不断地探索，寻求通过人工智能技术来获得技术上的新突破。

自动驾驶汽车（Automated Vehicle；Intelligent Vehicle；Autonomous Vehicle；Self-drivingCar；Driverless Car）又称智能汽车、自主汽车、自动驾驶汽车或轮式移动机器人，是一种通过计算机实现自动驾驶的智能汽车。

自动驾驶汽车等级标准,SAE J3016 标准(Level0~Level 5 共6 个级别)、

* Level 0：无自动化，由人类驾驶员全程操控汽车，但可以得到示警式或须干预的辅助信息。
* Level 1：辅助驾驶，利用环境感知信息对转向或纵向加减速进行闭环控制，其余工作由人类驾驶员完成。
* Level 2：部分自动化，利用环境感知信息同时对转向和纵向加减速进行闭环控制，其余工作由人类驾驶员完成。
* Level 3：有条件自动化，由自动驾驶系统完成所有驾驶操作，人类驾驶员根据系统请求进行干预。
* Level 4：高度自动化，由自动驾驶系统完成所有驾驶操作，无需人类驾驶员进行任何干预，但须限定道路和功能。
* Level 5：完全自动化，由自动驾驶系统完成所有的驾驶操作，人类驾驶员能够应付的所有道路和环境，系统也能完全自动完成。

目前对于自动驾驶汽车的研究有两条不同的技术路线：一条是渐进提高汽车驾驶的自动化水平；另一条是“一步到位”的无人驾驶技术发展路线。由SAE J3016 标准可以看出，通常大家谈论的无人驾驶汽车对应该标准的Level 4 和Level 5 级。无人驾驶汽车是自动驾驶的一种表现形式，它具有整个道路环境中所有与车辆安全性相关的控制功能，不需要驾驶员对车辆实施控制。


〉 自动驾驶技术的价值

* 1. 改善交通安全。驾驶员的过失责任是交通事故的主要因素。无人驾驶汽车不受人的心理和情绪干扰，保证遵守交通法规，按照规划路线行驶，可以有效地减少人为疏失所造成的交通事故。

* 2. 实现节能减排。由于通过合理调度实现共享享出行，减少了私家车购买数量，车辆绝对量的减少，将使温室气体排量大幅降低。

* 3. 消除交通拥堵，提升社会效率。自动驾驶汽车可以通过提高车速、缩小车距以及选择更有效路线来减少通勤所耗时间。 
* 4. 个人移动能力更加便利，不再需要找停车场。
* 5. 拉动汽车、电子、通信、服务、社会管理等协同发展，对促进我国产业转型升级具有重大战略意义。

# 实战

自动驾驶汽车关键技术包括环境感知、精准定位、决策与规划、控制与执行、高精地图与车联网V2X 以及自动驾驶汽车测试与验证技术；人工智能在自动驾驶汽车中的应用包括人工智能在环境感知中的应用、人工智能在决策规划中的应用、人工智能在车辆控制中的应用。


	计算机视觉（处理摄像头，分割、检测、识别）
				定位（算法+HD MAP）   路径规划  控制
	传感器融合fusion（激光雷达等）


    以百度apollo 无人驾驶平台介绍相关的技术
    
   1. apollo 源码分析
   2. 感知
   3. 规划  
   
[comma.ai（无人驾驶公司）的这两千行Python/tf代码 Learning a Driving Simulator](https://github.com/Ewenwan/research)

[openpilot 一个开源的自动驾驶（驾驶代理），它实行 Hondas 和 Acuras 的自适应巡航控制（ACC）和车道保持辅助系统（LKAS）的功能。 ](https://github.com/Ewenwan/openpilot)

[Autoware](https://github.com/Ewenwan/Autoware)

[udacity/self-driving-car](https://github.com/Ewenwan/self-driving-car)

[第六十八篇：从ADAS到自动驾驶（一）：自动驾驶发展及分级](https://blog.csdn.net/liaojiacai/article/details/55062873)

> 1.环境感知，起着人类驾驶员“眼睛”“耳朵”的作用

* 摄像机可以识别车辆行驶环境中的车辆、行人、车道线、路标、交通标志、交通信号灯等。它具有较高的图像稳定性、抗干扰能力和传输能力等特点。
* 激光雷达是以发射激光束来探测目标空间位置的主动测量设备。
* 毫米波雷达是指工作在毫米波波段、频率在30—300GHz 之间的雷达。根据测量原理的不同，毫米波雷达可分为脉冲方式毫米波雷达和调频连续波方式毫米波雷达两种。
* 超声波雷达的数据处理简单快速，检测距离较短，多用于近距离障碍物检测。

目前，环境感知技术有两种技术路线，一种是以摄像机为主导的多传感器融合方案，典型代表是特斯拉。另一种是以激光雷达为主导，其他传感器为辅助的技术方案，典型企业代表如谷歌、百度等。

> 2.精准定位

* 惯性导航系统由陀螺仪和加速度计构成，通过测量运动载体的线加速度和角速率数据，并将这些数据对时间进行积分运算，从而得到速度、位置和姿态。

车辆速度   时间 初始位置  和 初始速度   车辆加速度。


短时间内准确，长时间内，由于IMU数据飘逸，变得的不准确，可以和GPS结合



* 轮速编码器与航迹推算.可以通过轮速编码器推算出自动驾驶汽车的位置。通常轮速编码器安装在汽车的前轮，分别记录左轮与右轮的总转数。通过分析每个时间段里左右轮的转数，可以推算出车辆向前走了多远，向左右转了多少度等。由于在不同地面材质（如冰面与水泥地）上转数对距离转换存在偏差，随着时间推进，测量偏差会越来越大，因此单靠轮测距器并不能精准估计自动驾驶汽车的位姿。

* 卫星导航系统.目前全球卫星导航系统包括美国的GPS、俄罗斯的GLONASS、中国的北斗卫星导航系统。

卫星   <---> 地面控制站    gps接收器。

     时间*光速    距离 ，时间有误差


差分GPS:  使用两个GPS定位装置，两者都有误差，通过两个数据差分，来消除误差

实时运动定位 RTK， 通过地面基站计算GPS定位误差，gps接收器该误差来进行校正。

缺点: 大型建筑物阻挡信号


* SLAM 自主导航系统.目前主流有两种SLAM 策略。

第一种是基于激光雷达的SLAM，以谷歌汽车为例。车辆携带有GPS，通过GPS 对位置进行判断，并以激光雷达SLAM 点云图像与高精度地图进行坐标配准，匹配后确认自身位姿。


高精度地图定位，利用雷达数据和地图数据进行匹配，滤波算法定位

gps不能正常使用时：

车辆将其传感器识别的地标，通过坐标变换，与高精度地图数据进行匹配。

距离三个地标的信息，来定位，三角定位，三个圆圈的交汇点。

传感器 雷达数据获取的点云数据 和 MAP  利用点云匹配算法(ICP,迭代最近点)

滤波算法定位，直方图滤波，kaman滤波



第二种是基于视觉的SLAM，以Mobileye 为例。Mobileye 提出一种SLAM 的变种定位方法——REM。车辆通过采集包括信号灯、指示牌、车道线等标识，得到了一个简单的三维坐标数据，再通过视觉识别车道线等信息，获取一个一维数据。摄像机中的图像 与 高精度地图数据 进行配准，即可完成定位。

粒子滤波定位， 使用检测出的地图点 匹配定位，多中粒子点可能性，最后真实位置，保留了下来。


> 3.决策与规划

自动驾驶汽车的行为决策与路径规划是指依据环境感知和导航子系统输出信息，通过一些特定的约束条件如无碰撞、安全到达终点等，规划出给定起止点之间多条可选安全路径，并在这些路径中选取一条最优的路径作为车辆行驶轨迹。

* 路径规划：即路径局部规划时，自动驾驶车辆中的路径规划算法会在行驶任务设定之后将完成任务的最佳路径选取出来，避免碰撞和保持安全距离。在此过程中，会对路径的曲率和弧长等进行综合考量，从而实现路径选择的最优化。

* 驾驶任务规划：即全局路径规划，主要的规划内容是指行驶路径范围的规划。当自动驾驶汽车上路行驶时，驾驶任务规划会为汽车的自主驾驶提供方向引导方面的行为决策方案，通过GPS 技术进行即将需要前进行驶的路段和途径区域的规划与顺序排列。


**自动驾驶汽车主要使用的行为决策算法有以下3 种**
* 基于神经网络：自动驾驶汽车的决策系统主要采用神经网络确定具体的场景并做出适当的行为决策。
* 基于规则：工程师想出所有可能的“if-then 规则”的组合，然后再用基于规则的技术路线对汽车的决策系统进行编程。
* 混合路线：结合了以上两种决策方式，通过集中性神经网络优化，通过“if-then 规则”完善。混合路线是最流行的技术路线。

感知与决策技术的核心是人工智能算法与芯片。人工智能算法的实现需要强大的计算能力做支撑，特别是深度学习算法的大规模使用，对计算能力提出了更高的要求。随着人工智能业界对于计算能力要求的快速提升，进入2015 年后，业界开始研发针对人工智能的专用芯片，通过更好的硬件和芯片架构，在计算效率上进一步带来大幅的提升。


〉4.控制与执行

自动驾驶汽车的车辆控制系统是自动驾驶汽车行驶的基础，包括车辆的纵向控制和横向控制。纵向控制，即车辆的驱动与制动控制，是指通过对油门和制动的协调，实现对期望车速的精确跟随。横向控制，即通过方向盘角度的调整以及轮胎力的控制，实现自动驾驶汽车的路径跟踪。

* 纵向控制.自动驾驶汽车采用油门和制动综合控制的方法来实现对预定车速的跟踪，各种电机-发动机-传动模型、汽车运行模型和刹车过程模型与不同的控制算法相结合，构成了各种各样的纵向控制模式。

* 横向控制.车辆横向控制主要有两种基本设计方法：基于驾驶员模拟的方法和基于车辆动力学模型的控制方法。

基于驾驶员模拟的方法：一种是使用较简单的动力学模型和驾驶员操纵规则设计控制器；另一种是用驾驶员操纵过程的数据训练控制器获取控制算法。

基于车辆动力学模型的方法：需要建立较精确的汽车横向运动模型。典型模型如单轨模型，该模型认为汽车左右两侧特性相同。

* 车辆控制平台.车辆控制平台是无人车的核心部件，控制着车辆的各种控制系统。其主要包括电子控制单元（ECU）和通信总线两部分。ECU 主要用来实现控制算法，通信总线主要用来实现ECU与机械部件间的通信功能.

* 通信总线：目前，车用总线技术被国际自动机工程师学会（SEA）下的汽车网络委员会按照协议特性分为A、B、C、D 共4类.

〉5.高精地图与车联网V2X

* 高精地图拥有精确的车辆位置信息和丰富的道路元素数据信息，起到构建类似于人脑对于空间的整体记忆与认知的功能，可以帮助汽车预知路面复杂信息，如坡度、曲率、航向等，更好地规避潜在的风险，是自动驾驶汽车的核心技术之一。

高精地图相比服务于GPS 导航系统的传统地图而言，最显著的特征是其表征路面特征的精准性。传统地图只需要做到米量级的精度就可以实现基于GPS 的导航，而高精地图需要至少十倍以上的精度，即达到厘米级的精度才能保证自动驾驶汽车行驶的安全。

同时，高精地图还需要有比传统地图更高的实时性。由于道路路网经常会发生变化，如道路整修、标识线磨损或重漆、交通标识改变等。这些改变都要及时反映在高精地图上，以确保自动驾驶汽车的行车安全。


百度高精度地图 openDrive

数据收集(实车带传感器实地跑) -> 数据处理（整理、分类、清洗） -> 目标检测分类(车道线、路标、交通标示、电线杆) -> 人工验证  -> 地图发布


* 车联网V2X

V2X 表示Vehicle to X，其中X 表示基础设施（Infrastructure）、车辆（Vehicle）、行人（Pedestrian）、道路（Road）等。V2X 网联通信集成了V2N、V2V、V2I 和V2P 共四类关健技术。

V2N（Vehicle to Network，车-互联网），通过网络将车辆连接到云服务器，能够使用云服务器上的娱乐、导航等功能。

V2V（Vehicle to Vehicle，车-车），指不同车辆之间的信息互通。

V2I（Vehicle to Infrastructure，车-基础设施），包括车辆与路障、道路、交通灯等设施之间的通信，用于获取路障位置、交通灯信号时序等道路管理信息。

V2P（Vehicle to Pedestrian，车-行人），指车辆与行人或非机动车之间的交互，主要是提供安全警告。

V2X 技术的实现一般基于RFID、拍照设备、车载传感器等硬件平台。V2X 网联通信产业分为DSRC 和LTE-V2X 两个标准和产业阵营。


〉6.自动驾驶汽车测试与验证技术

* 实测.让车辆行驶数百万公里，以确定设计的系统是否安全并按照预期运行。该方法的困难在于必须累积的测试里程数，这可能要花费大量的时间。

* 软件在环或模型在环仿真.另一种更可行的方法是将现实世界的测试与仿真相结合。在仿真软件所构建的各种场景中，通过算法控制车辆进行相应的应对操作，来证明所设计的系统确实可以在各种场景下做出正确的决定，这可以大大减少必须完成的测试里程数。

* 硬件在环仿真.为了验证真实硬件的运行情况，硬件在环仿真可以对其进行测试，并将预先记录的传感器数据提供给系统，此种技术路线可以降低车辆测试和验证的成本。

## 人工智能在自动驾驶汽车中的应用
### 工智能在环境感知中的应用
环境感知包括：可行驶路面检测、车道线检测、路缘检测、护栏检测、行人检测、机动车检测、非机动车检测、路标检测、交通标志检测、交通信号灯检测等。

对于如此复杂的路况检测，深度学习能够满足视觉感知的高精度需求。基于深度学习的计算机视觉，可获得较接近于人的感知能力。有研究报告指出深度学习在算法和样本量足够的情况下，视觉感知的准确率可以达到99.9%以上，而传统视觉算法的检测精度极限在93%左右，人感知的准确率一般是95%左右。

### 人工智能在决策与规划中的应用

行为决策与路径规划是人工智能在自动驾驶汽车领域中的另一个重要应用。前期决策树、贝叶斯网络等人工智能方法已有大量应用。近年来兴起的深度卷积神经网络与深度强化学习，能通过大量学习实现对复杂工况的决策，并能进行在线学习优化，由于需要较多的计算资源，当前是计算机与互联网领域研究自动驾驶汽车的决策与规划处理的热门技术。随着深度强化学习的兴起，越来越多的公司和研究者把强化学习应用到无人车的行为与决策中，并取得了不错的效果.

可学习部分是将无人车所处的环境映射成一系列抽象策略的过程。他们设计了一张策略选项图，主要包含无人车的加减速、转向以及对周围车辆的反应，并利用策略网络来选择合适的应对选项。其中，策略网络在给定的车辆环境下，评估每一种应对的可能影响，从而选择最合适的策略。不可学习部分则是将学习到的抽象策略转化成对车辆的实际控制动作，该部分主要对车辆动作进行具体规划，检查抽象策略是否可执行，或者执行满足策略的动作，从而充分保证系统的安全性。

### 人工智能在车辆控制中的应用

相对于传统的车辆控制方法，智能控制方法主要体现在对控制对象模型的运用和综合信息学习运用上，包括神经网络控制和深度学习方法等，这些算法已逐步在车辆控制中广泛应用。

* **神经控制**，是研究和利用人脑的某些结构机理以及人的知识和经验对系统的控制。利用神经网络，可以把控制问题看成模式识别问题，被识别的模式映射成“行为”信号的“变化”信号。神经控制最显著的特点是具有学习能力。它是通过不断修正神经元之间的连接权值，并离散存储在连接网络中来实现的。它对非线性系统和难以建模的系统的控制具有良好效果。

* **深度神经网络学习**，源于神经网络的研究，可理解为深层的神经网络。通过它可以获得深层次的特征表示，免除人工选取特征的繁复冗杂和高维数据的维度灾难问题。深度学习在特征提取与模型拟合方面显示了其潜力和优势。对于存在高维数据的控制系统，引入深度学习具有一定的意义。自动驾驶系统需要尽量减少人的参与或者没有人的参与，深度学习自动学习状态特征的能力使得深度学习在自动驾驶系统的研究中具有先天的优势。

* **深度强化学习**，强化学习的灵感来源于生物学中的动物行为训练，训练员通过奖励与惩罚的方式让动物学会一种行为与状态之间的某种联系规则。强化学习就是要解决这类问题：一个能够感知环境的智能体怎样通过学习选择达到其目标的最优动作。



## 应用篇 

业界普遍认为，自动驾驶技术在公共交通领域和特定场所的使用将早于在个人乘用车市场的普及。自动驾驶汽车将最先应用的行业包括公共交通、快递运输、服务于老年人和残疾人.

自动驾驶巴士、无人驾驶出租车、物流运输、服务于残疾人

自动驾驶巴士被认为是解决城市“最后一公里”难题的有效方案，大多用于机场、旅游景区和办公园区等封闭的场所。

自动驾驶汽车在公共交通领域的另一个重要应用是出租车。

快递用车和“列队”卡车将是另外一个较快采用自动驾驶汽车的领域。
随着全球老龄化问题的加剧，自动驾驶技术在快递等行业的应用将极大地弥补劳动力不足的问题，并且随着自动驾驶技术的成熟与市场普及程度的提高，无人配送将成为必然的趋势。

自动驾驶汽车已经开始在老年人和残疾人这两个消费群体中有所应用。自动驾驶汽车不仅可增强老年人的移动能力，也能帮助残疾人旅行。


## 困难和挑战

自动驾驶的一个很重要的用途是用于某些特殊的环境下，由于在某些特殊的环境下，人员生存困难，自动驾驶能克服这些问题，但是其也要解决如极寒、道路条件复杂等各种极端环境的影响，这同样也是自动驾驶未来发展所应面临的困难。

由于人工智能的大量应用，自动驾驶技术更依赖于网络，如通过云端获取的高精地图、精准导航等的数据，其安全性显得尤为突出。如何打造安全可靠的数据链路，不被黑客侵扰等也将是需要长期面临的困难与挑战。


# Apollo 相关介绍



```
We choose to go to the moon in this decade and do the other things,
not because they are easy, but because they are hard.
-- John F. Kennedy, 1962
```

Welcome to the Apollo GitHub.

[Apollo](http://apollo.auto) 开源自动驾驶平台. 
It is a high performance flexible architecture which supports fully autonomous driving capabilities.
For business contact, please visit http://apollo.auto

**Apollo Team now proudly presents to you the latest [version 2.5](https://github.com/ApolloAuto/apollo/releases/tag/v2.5.0).**

## 安装

推荐在 Docker environment 中安装

The steps are:
 - 1. Run a machine that runs linux (tested on Ubuntu 16.04 with and without an nVidia GPU)
 - 2. Create a docker environment
 - 3. Build Apollo from source
 - 4. Bootstrap start Apollo
 - 5. Download the demonstration loop and run it
 - 6. Start a browser session and see the Dreamview user interface

More instructions are below

###  docker environment 安装

First, you need to [install docker-ce properly](https://github.com/ApolloAuto/apollo/blob/master/docker/scripts/README.md#install-docker).
The following scripts will get you into the container

```
docker ps  # to verify docker works without sudo
bash docker/scripts/dev_start.sh
# if in China, you had better use:bash docker/scripts/dev_start.sh -C to download from the server of docker in china.
bash docker/scripts/dev_into.sh

```

### 源码编译 apollo
```
# To get a list of build commands
./apollo.sh
# To make sure you start clean
./apollo.sh clean
# This will build the full system and requires that you have an nVidia GPU with nVidia drivers loaded
bash apollo.sh build
```

If you do not have an nVidia GPU, the system will run but with the CUDA-based perception and other modules. 

You mustspecify either `dbg` for debug mode or `opt` for optimized code

```
./apollo.sh build_no_perception dbg
```

If you make modifications to the Dreamview frontend, then you must run `./apollo.sh build_fe`  before you run the
full build.


## 运行 Apollo

Follow the steps below to launch Apollo. Note that you must build the system first before you run it.
Note that the bootstrap.sh will actually succeed but the user interface will not come up if you skip the build step.

### Start Apollo

Running Apollo will start the ROS core and then startup a web user interface called Dreamview, 
this is handled by the bootstrap script, so from within the docker container, you should run:

```
# start module monitor
bash scripts/bootstrap.sh
```

### Access Dreamview
    Access Dreamview by opening your favorite browser, e.g. Chrome, go to http://localhost:8888 
    and you should see this screenHowever, there will be nothing running in the system.

![Access Dreamview](https://github.com/Ewenwan/apollo/docs/demo_guide/images/apollo_bootstrap_screen.png)

### Select Drive Mode
From the dropdown box selet "Navigation" mode.

![Navigation Mode](https://github.com/Ewenwan/apollo/docs/demo_guide/images/dreamview_2_5_setup_profile.png)


### Replay demo rosbag

To see if the system works, use the demo 'bag' which feeds the system.

```
# get rosbag note that the command download is required
python ./docs/demo_guide/rosbag_helper.py demo_2.5.bag

# You can now replay this demo "bag" in a loop with the '-l' flag
rosbag play -l demo_2.5.bag
```

Dreamview should show a running vehicle now. (The following image might be different due to changes in frontend.)

![Dreamview with Trajectory](docs/demo_guide/images/dv_trajectory_2.5.png)

## Documents

Apollo documents can be found under the [docs](https://github.com/ApolloAuto/apollo/blob/master/docs/) repository.
   * [quickstart](https://github.com/ApolloAuto/apollo/blob/master/docs/quickstart/): the quickstart tutorial.
   * [demo_guide](https://github.com/ApolloAuto/apollo/blob/master/docs/demo_guide/): the guide for demonstration.
   * [![Apollo Offline Demo](https://img.youtube.com/vi/Q4BawiLWl8c/0.jpg)](https://www.youtube.com/watch?v=Q4BawiLWl8c)
   * [how to contribute code](https://github.com/ApolloAuto/apollo/blob/master/CONTRIBUTING.md): the guide for contributing code to Apollo.
   * [howto](https://github.com/ApolloAuto/apollo/blob/master/docs/howto/): tutorials on how to build, run and modify codes.
   * [specs](https://github.com/ApolloAuto/apollo/blob/master/docs/specs/): Specification documents of Apollo.
   * [Doxygen APIs](https://apolloauto.github.io/doxygen/apollo/): Apollo Doxygen pages

## Ask Questions

You are welcome to submit questions and bug reports as [Github Issues](https://github.com/ApolloAuto/apollo/issues).

## Copyright and License

Apollo is provided under the [Apache-2.0 license](LICENSE).

## Disclaimer
Please refer the Disclaimer of Apollo in [Apollo official website](http://apollo.auto/docs/disclaimer.html).
# ===========================
# Apollo 3.0 技术指南

## 概况
> 了解Apollo3.0基础概念和Apollo3.0快速入门指南

  * [Apollo 3.0快速入门指南](https://github.com/ApolloAuto/apollo/blob/master/docs/quickstart/apollo_3_0_quick_start_cn.md)
  
## 硬件和系统安装
> 了解Apollo3.0硬件和系统安装过程

  * [Apollo 3.0硬件和系统安装指南](https://github.com/ApolloAuto/apollo/blob/master/docs/quickstart/apollo_3_0_hardware_system_installation_guide_cn.md)

## 校准
> 了解校准的过程

  * [Apollo激光雷达校准指南](https://github.com/ApolloAuto/apollo/blob/master/docs/quickstart/apollo_1_5_lidar_calibration_guide_cn.md)
  * [Apollo 2.0传感器校准指南](https://github.com/ApolloAuto/apollo/blob/master/docs/quickstart/apollo_2_0_sensor_calibration_guide_cn.md)
  * [多激光雷达全球导航卫星系统(Multiple-LiDAR GNSS)校准指南](https://github.com/ApolloAuto/apollo/blob/master/docs/quickstart/multiple_lidar_gnss_calibration_guide_cn.md)
  * [Apollo坐标系统](https://github.com/ApolloAuto/apollo/blob/master/docs/specs/coordination_cn.md)

## 软件安装
> 了解Apollo3.0的软件安装过程

  * [Apollo软件安装指南](https://github.com/ApolloAuto/apollo/blob/master/docs/quickstart/apollo_software_installation_guide_cn.md)
  * [如何调试Dreamview启动问题](https://github.com/ApolloAuto/apollo/blob/master/docs/howto/how_to_debug_dreamview_start_problem_cn.md)
  * [运行线下演示](https://github.com/ApolloAuto/apollo/blob/master/docs/demo_guide/README_cn.md)
  
## Apollo系统架构和原理
> 了解核心模块的架构和原理

  * [Apollo 3.0 软件架构](https://github.com/ApolloAuto/apollo/blob/master/docs/specs/Apollo_3.0_Software_Architecture_cn.md "Apollo software architecture")
  * [3D 障碍物感知](https://github.com/ApolloAuto/apollo/blob/master/docs/specs/3d_obstacle_perception_cn.md)
  * [Apollo 3.0感知](https://github.com/ApolloAuto/apollo/blob/master/modules/perception/README.md)
  * [二次规划（QP）样条路径优化](https://github.com/ApolloAuto/apollo/blob/master/docs/specs/qp_spline_path_optimizer_cn.md)
  * [二次规划（QP）样条ST速度优化](https://github.com/ApolloAuto/apollo/blob/master/docs/specs/qp_spline_st_speed_optimizer_cn.md)
  * [参考线平滑设定](https://github.com/ApolloAuto/apollo/blob/master/docs/specs/reference_line_smoother_cn.md)
  * [交通信号灯感知](https://github.com/ApolloAuto/apollo/blob/master/docs/specs/traffic_light_cn.md)
  
## 功能模块和相关扩展知识
> 了解Apollo功能模块和相关扩展知识

  * [控制总线模块](https://github.com/ApolloAuto/apollo/blob/master/modules/canbus/README.md)
  * [通用模块](https://github.com/ApolloAuto/apollo/blob/master/modules/common/README.md)
  * [控制模块](https://github.com/ApolloAuto/apollo/blob/master/modules/control/README.md)
  * [数据模块](https://github.com/ApolloAuto/apollo/blob/master/modules/data/README.md)
  * [定位模块](https://github.com/ApolloAuto/apollo/blob/master/modules/localization/README.md)
  * [感知模块](https://github.com/ApolloAuto/apollo/blob/master/modules/perception/README.md)
  * [Planning模块](https://github.com/ApolloAuto/apollo/blob/master/modules/planning/README.md)
  * [预测模块](https://github.com/ApolloAuto/apollo/blob/master/modules/prediction/README.md)
  * [寻路模块](https://github.com/ApolloAuto/apollo/blob/master/modules/routing/README.md)

  * [如何添加新的GPS接收器](https://github.com/ApolloAuto/apollo/blob/master/docs/howto/how_to_add_a_gps_receiver_cn.md)
  * [如何添加新的CAN卡](https://github.com/ApolloAuto/apollo/blob/master/docs/howto/how_to_add_a_new_can_card_cn.md )
  * [如何添加新的控制算法](https://github.com/ApolloAuto/apollo/blob/master/docs/howto/how_to_add_a_new_control_algorithm_cn.md)
  * [如何在预测模块中添加新评估器](https://github.com/ApolloAuto/apollo/blob/master/docs/howto/how_to_add_a_new_evaluator_in_prediction_module_cn.md)
  * [如何在预测模块中添加一个预测器](https://github.com/ApolloAuto/apollo/blob/master/docs/howto/how_to_add_a_new_predictor_in_prediction_module_cn.md)
  * [如何在Apollo中添加新的车辆](https://github.com/ApolloAuto/apollo/blob/master/docs/howto/how_to_add_a_new_vehicle_cn.md)
  * [如何添加新的外部依赖项](https://github.com/ApolloAuto/apollo/blob/master/docs/howto/how_to_add_an_external_dependency_cn.md)
  
  ## 开发者工具
> 了解开发者工具

  * [使用VSCode构建、调试Apollo项目](https://github.com/ApolloAuto/apollo/blob/master/docs/howto/how_to_build_and_debug_apollo_in_vs
