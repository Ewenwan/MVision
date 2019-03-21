# 无人驾驶
![](https://github.com/Ewenwan/MVision/blob/master/UMCar/img/sdc.PNG)

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

* 轮速编码器与航迹推算.可以通过轮速编码器推算出自动驾驶汽车的位置。通常轮速编码器安装在汽车的前轮，分别记录左轮与右轮的总转数。通过分析每个时间段里左右轮的转数，可以推算出车辆向前走了多远，向左右转了多少度等。由于在不同地面材质（如冰面与水泥地）上转数对距离转换存在偏差，随着时间推进，测量偏差会越来越大，因此单靠轮测距器并不能精准估计自动驾驶汽车的位姿。

* 卫星导航系统.目前全球卫星导航系统包括美国的GPS、俄罗斯的GLONASS、中国的北斗卫星导航系统。

* SLAM 自主导航系统.目前主流有两种SLAM 策略。

第一种是基于激光雷达的SLAM，以谷歌汽车为例。车辆携带有GPS，通过GPS 对位置进行判断，并以激光雷达SLAM 点云图像与高精度地图进行坐标配准，匹配后确认自身位姿。

第二种是基于视觉的SLAM，以Mobileye 为例。Mobileye 提出一种SLAM 的变种定位方法——REM。车辆通过采集包括信号灯、指示牌等标识，得到了一个简单的三维坐标数据，再通过视觉识别车道线等信息，获取一个一维数据。摄像机中的图像与REM 地图中进行配准，即可完成定位。


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

  * [使用VSCode构建、调试Apollo项目](https://github.com/ApolloAuto/apollo/blob/master/docs/howto/how_to_build_and_debug_apollo_in_vscode_cn.md "How  to build and debug Apollo in VSCode")
  * [DreamView用法介绍](https://github.com/ApolloAuto/apollo/blob/master/docs/specs/dreamview_usage_table_cn.md)



# ===============================
# Apollo具体内容说明

## 软件
- [Apollo 2.0软件系统架构](Apollo_2.0_Software_Architecture.md)
- [Apollo 3.0软件系统架构](Apollo_3.0_Software_Architecture_cn.md)
- [Planning模块架构概述](Class_Architecture_Planning_cn.md)

## Apollo硬件开发平台

我们强烈建议使用者在阅读硬件开发平台文档前浏览我们的免责声明。

- [Apollo传感器单元（ASU）](https://github.com/ApolloAuto/apollo/tree/master/docs/specs/Apollo_Sensor_Unit/Apollo_Sensor_Unit_Installation_Guide_cn.md)
- [摄像机](https://github.com/ApolloAuto/apollo/tree/master/docs/specs/Camera/README.md)
- [激光雷达](https://github.com/ApolloAuto/apollo/tree/master/docs/specs/Lidar/README.md)
- [雷达](https://github.com/ApolloAuto/apollo/tree/master/docs/specs/Radar/README.md)
- [导航](https://github.com/ApolloAuto/apollo/tree/master/docs/specs/Navigation/README_cn.md)
- [IPC](https://github.com/ApolloAuto/apollo/tree/master/docs/specs/IPC/Nuvo-6108GC_Installation_Guide_cn.md)
- [软件系统和内核安装指南](https://github.com/ApolloAuto/apollo/tree/master/docs/specs/Software_and_Kernel_Installation_guide_cn.md)

## 感知

- [Apollo 2.5感知系统](https://github.com/ApolloAuto/apollo/tree/master/docs/specs/perception_apollo_2.5.md)
- [Apollo 2.5传感器安装指南](https://github.com/ApolloAuto/apollo/tree/master/docs/specs/Guideline_sensor_Installation_apollo_2.5.md)
- [Apollo 3.0感知系统]https://github.com/ApolloAuto/apollo/tree/master/docs/specs/(perception_apollo_3.0_cn.md)
- [Apollo 3.0传感器安装指南](https://github.com/ApolloAuto/apollo/tree/master/docs/specs/Guideline_sensor_Installation_apollo_3.0_cn.md)
- [激光雷达校准英文版](https://github.com/ApolloAuto/apollo/tree/master/docs/specs/lidar_calibration.pdf)
- [激光雷达校准中文版](https://github.com/ApolloAuto/apollo/tree/master/docs/specs/lidar_calibration_cn.pdf)

## HMI
- [Dreamview使用方法](https://github.com/ApolloAuto/apollo/tree/master/docs/specs/dreamview_usage_table_cn.md)

## 算法
- [三维障碍物感知英文版](https://github.com/ApolloAuto/apollo/tree/master/docs/specs/3d_obstacle_perception.md)
- [三维障碍物感知中文版](https://github.com/ApolloAuto/apollo/tree/master/docs/specs/3d_obstacle_perception_cn.md)
- [二次规划路径样条优化](https://github.com/ApolloAuto/apollo/tree/master/docs/specs/qp_spline_path_optimizer_cn.md)
- [二次规划st速度样条优化](https://github.com/ApolloAuto/apollo/tree/master/docs/specs/qp_spline_st_speed_optimizer_cn.md)
- [参考线平滑设定](https://github.com/ApolloAuto/apollo/tree/master/docs/specs/reference_line_smoother_cn.md)
- [交通信号灯感知](https://github.com/ApolloAuto/apollo/tree/master/docs/specs/traffic_light_cn.md)

## 其他通用知识
- [坐标系统](https://github.com/ApolloAuto/apollo/tree/master/docs/specs/coordination_cn.md)
- [Apollo安全更新SDK用户指南](https://github.com/ApolloAuto/apollo/tree/master/docs/specs/apollo_secure_upgrade_user_guide-CN.md)


# ===============================
# 工业级PC（IPC）软件安装指南

本文档介绍下述软件的安装步骤：

- Ubuntu Linux
- Apollo Kernel
- Nvidia GPU Driver

![tip_icon](images/tip_icon.png)成功完成本文档中介绍的软件安装需要使用者有使用Linux系统的经验。

## 安装Unbuntu Linux

按照如下步骤执行：

1. 创建一个可引导的Ubuntu Linux USB启动盘：

   下载Ubuntu 14.04（或其他的变种系统如Xubuntu）并[创建一个可引导的USB启动盘](https://tutorials.ubuntu.com/tutorial/tutorial-create-a-usb-stick-on-ubuntu#0)。我们推荐使用Ubuntu 14.04。在系统启动时按下F2（或其他按键，请参考系统文档）进入BIOS设置，我们建议禁用Quick Boot和Quiet Boot设置以更容易的在启动时捕获错误信息。
   
2. 安装Ubuntu Linux：

   a.   将安装Unbuntu的USB启动盘插入USB接口中并启动系统
   
   b.   按照屏幕提示执行安装
   
3. 执行软件更新：

   a.   安装结束后重启并进入系统
   
   b.   启动Software Update并更新到最新软件包，或在终端程序如GNOME Terminal中执行下述指令完成更新：

   ```shell
   sudo apt-get update; sudo apt-get upgrade
   ```
   
   c. 启动终端程序如GNOME Terminal，执行下述指令安装Linux 4.4内核
   
   ```shell
   sudo apt-get install linux-generic-lts-xenial
   ```
   
   ![tip_icon](images/tip_icon.png)IPC必须有网络连接以更新和安装软件。确保IPC的以太网线接入了有互联网访问权限的网络。如果接入的网络没有使用动态主机配置协议（DHCP），使用者可能需要对IPC的网络进行配置。

## 安装Apollo内核

Apollo在车辆上的运行需要[Apollo内核](https://github.com/ApolloAuto/apollo-kernel)。我们强烈推荐安装预先构建的内核版本。

##  使用预先构建的内核版本

使用者使用下述指令获取和安装预先构建的内核版本。

1. 从GitHub下载发布版本包：

```
https://github.com/ApolloAuto/apollo-kernel/releases
```

2. 成功下载发布版本包后安装内核：

```
tar zxvf linux-4.4.32-apollo-1.5.0.tar.gz
cd install
sudo bash install_kernel.sh
```

3. 使用 `reboot` 指令重启系统
4. 【可选步骤-如果使用者使用了CAN卡】参考CAN卡供应商提供的指令构建CAN卡驱动程序

##  构建个人的内核版本

如果使用者修改了内核，或者预先构建的版本对使用者的工作平台不是最好的选择，使用者可以使用下述指令构建个人的内核版本：

1. 从资源库中clone源代码

```
git clone https://github.com/ApolloAuto/apollo-kernel.git
cd apollo-kernel
```

2. 参考CAN卡供应商提供的指令加入CAN卡驱动的源代码
3. 使用下述指令构建内核：

```
bash build.sh
```

4. 参考上面章节中介绍的如何安装预先构建内核版本的步骤进行内核的安装

## 安装NVIDIA GPU驱动

Apollo在车辆上的运行需要[NVIDIA GPU驱动](http://www.nvidia.com/download/driverResults.aspx/114708/en-us)。使用者必须使用指定的参数选项安装NVIDIA GPU驱动。

1. 下载安装文件

```
wget http://us.download.nvidia.com/XFree86/Linux-x86_64/375.39/NVIDIA-Linux-x86_64-375.39.run
```

2. 执行驱动安装

```
sudo bash ./NVIDIA-Linux-x86_64-375.39.run --no-x-check -a -s
```

##  参考资料

1. [Ubuntu官方网站](https://www.ubuntu.com/desktop)


# =============================
# Apollo 3.0 软件架构

自动驾驶Apollo3.0核心软件模块包括：

- **感知** — 感知模块识别自动驾驶车辆周围的世界。感知中有两个重要的子模块：障碍物检测和交通灯检测。
- **预测** — 预测模块预测感知障碍物的未来运动轨迹。
- **路由** — 路由模块告诉自动驾驶车辆如何通过一系列车道或道路到达其目的地。
- **规划** — 规划模块规划自动驾驶车辆的时间和空间轨迹。
- **控制** — 控制模块通过产生诸如油门，制动和转向的控制命令来执行规划模块产生的轨迹。
- **CanBus** — CanBus是将控制命令传递给车辆硬件的接口。它还将底盘信息传递给软件系统。
- **高精地图** — 该模块类似于库。它不是发布和订阅消息，而是经常用作查询引擎支持，以提供关于道路的特定结构化信息。
- **定位** — 定位模块利用GPS，LiDAR和IMU的各种信息源来定位自动驾驶车辆的位置。
- **HMI** — Apollo中的HMI和DreamView是一个用于查看车辆状态，测试其他模块以及实时控制车辆功能的模块.
- **监控** — 车辆中所有模块的监控系统包括硬件。
- **Guardian** — 新的安全模块，用于干预监控检测到的失败和action center相应的功能。
执行操作中心功能并进行干预的新安全模块应监控检测故障。

```
注意：下面列出了每个模块的详细信息。
```

这些模块的交互如下图所示。

![img](https://github.com/ApolloAuto/apollo/blob/master/docs/specs/images/Apollo_3.0_SW.png)

每个模块都作为单独的基于CarOS的ROS节点运行。每个模块节点都发布和订阅特定topic。订阅的topic用作数据输入，而发布的topic用作数据输出。以下各节详细介绍了各模块情况。

## 感知

感知依赖LiDAR点云数据和相机原始数据。除了这些传感器数据输入之外，交通灯检测依赖定位以及HD-Map。由于实时ad-hoc交通灯检测在计算上是不可行的，因此交通灯检测需要依赖定位确定何时何地开始通过相机捕获的图像检测交通灯。
对Apollo 3.0的更改：
  - CIPV检测/尾随 - 在单个车道内移动。
  - 全线支持 - 粗线支持，可实现远程精确度。相机安装有高低两种不同的安装方式。
  - 异步传感器融合 – 因为不同传感器的帧速率差异——雷达为10ms，相机为33s，LiDAR为100ms，所以异步融合LiDAR，雷达和相机数据，并获取所有信息并得到数据点的功能非常重要。
  - 在线姿态估计 - 在出现颠簸或斜坡时确定与估算角度变化，以确保传感器随汽车移动且角度/姿态相应地变化。
  - 视觉定位 – 基于相机的视觉定位方案正在测试中。
  - 超声波传感器 – 作为安全保障传感器，与Guardian一起用于自动紧急制动和停车。

## 预测

预测模块负责预测所有感知障碍物的未来运动轨迹。输出预测消息封装了感知信息。预测订阅定位和感知障碍物消息，如下所示。

![img](https://github.com/ApolloAuto/apollo/blob/master/docs/specs/images/prediction.png)

当接收到定位更新时，预测模块更新其内部状态。当感知发出其发布感知障碍物消息时，触发预测实际执行。

## 定位

定位模块聚合各种数据以定位自动驾驶车辆。有两种类型的定位模式：OnTimer和多传感器融合。

第一种基于RTK的定位方法，通过计时器的回调函数“OnTimer”实现，如下所示。
![img](https://github.com/ApolloAuto/apollo/blob/master/docs/specs/images/localization.png)

另一种定位方法是多传感器融合（MSF）方法，其中注册了一些事件触发的回调函数，如下所示。

![img](https://github.com/ApolloAuto/apollo/blob/master/docs/specs/images/localization_2.png)

## 路由
为了计算可通行车道和道路，路由模块需要知道起点和终点。通常，路由起点是自动驾驶车辆位置。重要的数据接口是一个名为`OnRoutingRequest`的事件触发函数，其中`RoutingResponse`的计算和发布如下所示。

![img](https://github.com/ApolloAuto/apollo/blob/master/docs/specs/images/routing.png)

## 规划
Apollo 2.0需要使用多个信息源来规划安全无碰撞的行驶轨迹，因此规划模块几乎与其他所有模块进行交互。

首先，规划模块获得预测模块的输出。预测输出封装了原始感知障碍物，规划模块订阅交通灯检测输出而不是感知障碍物输出。
然后，规划模块获取路由输出。在某些情况下，如果当前路由结果不可执行，则规划模块还可以通过发送路由请求来触发新的路由计算。

最后，规划模块需要知道定位信息（定位：我在哪里）以及当前的自动驾驶车辆信息（底盘：我的状态是什么）。规划模块由固定频率触发，主数据接口是调用`RunOnce`函数的`OnTimer`回调函数。

![img](https://github.com/ApolloAuto/apollo/blob/master/docs/specs/images/planning_1.png)

底盘，定位，交通灯和预测等数据依赖关系通过`AdapterManager`类进行管理。核心软件模块同样也由`AdapterManager`类管理。例如，定位通过`AdapterManager :: GetLocalization()`管理，如下所示。

![img](https://github.com/ApolloAuto/apollo/blob/master/docs/specs/images/planning_2.png)

## 控制
如规划模块中所述，控制将规划轨迹作为输入，并生成控制命令传递给CanBus。它有三个主要的数据接口：OnPad，OnMonitor和OnTimer。

![img](https://github.com/ApolloAuto/apollo/blob/master/docs/specs/images/control_1.png)

`OnPad`和`OnMonitor`是仿真和HMI的交互接口。 主要数据接口是`OnTimer`，它定期产生实际的控制命令，如下所示。

![img](https://github.com/ApolloAuto/apollo/blob/master/docs/specs/images/control_2.png)

## CanBus

CanBus有两个数据接口。

![img](https://github.com/ApolloAuto/apollo/blob/master/docs/specs/images/canbus_1.png)

第一个数据接口是基于计时器的发布者，回调函数为“OnTimer”。如果启用，此数据接口会定期发布底盘信息。

![img](https://github.com/ApolloAuto/apollo/blob/master/docs/specs/images/canbus_2.png)

第二个数据接口是一个基于事件的发布者，回调函数为“OnControlCommand”，当CanBus模块接收到控制命令时会触发该函数。


## HMI
Apollo中的HMI或DreamView是一个Web应用程序：
     - 可视化自动驾驶模块的输出，例如，规划轨迹，汽车定位，底盘状态等。
     - 为用户提供人机交互界面，以查看硬件状态，打开/关闭模块，以及启动自动驾驶汽车。
     - 提供调试工具，如PnC Monitor，以有效跟踪模块问题。

## 监控
包括硬件在内的，车辆中所有模块的监控系统。监控模块从其他模块接收数据并传递给HMI，以便司机查看并确保所有模块都正常工作。如果模块或硬件发生故障，监控会向Guardian（新的操作中心模块）发送警报，然后决定需要采取哪些操作来防止系统崩溃。

## Guardian
这个新模块根据Monitor发送的数据做出相应决定。Guardian有两个主要功能：
     - 所有模块都正常工作 - Guardian允许控制模块正常工作。控制信号被发送到CANBus，就像Guardian不存在一样。
     - 监控检测到模块崩溃 - 如果监控检测到故障，Guardian将阻止控制信号到达CANBus并使汽车停止。 Guardian有三种方式决定如何停车并会依赖最终的Gatekeeper——超声波传感器，
         - 如果超声波传感器运行正常而未检测到障碍物，Guardian将使汽车缓慢停止
         - 如果传感器没有响应，Guardian会硬制动，使车马上停止。
         - 这是一种特殊情况，如果HMI通知驾驶员即将发生碰撞并且驾驶员在10秒内没有干预，Guardian会使用硬制动使汽车立即停止。

```
注意: 
1.在上述任何一种情况下，如果Monitor检测到任何模块或硬件出现故障，Guardian将始终停止该车。
2.监控器和Guardian解耦以确保没有单点故障，并且可以为Guardian模块添加其他行为且不影响监控系统，监控还与HMI通信。
```

# =======================
# 感知
Apollo 3.0
June 27, 2018

## 简介
    Apollo 3.0 主要针对采用低成本传感器的L2级别自动驾驶车辆。
    在车道中的自动驾驶车辆通过一个前置摄像头和前置雷达要与关键车辆（在路径上最近的车辆）保持一定的距离。
    Apollo 3.0 支持在高速公路上不依赖地图的高速自动驾驶。
    深度网路学习处理图像数据，随着搜集更多的数据，深度网络的性能随着时间的推移将得到改善。


***安全警告***
    Apollo 3.0 不支持没有包含本地道路和说明标示的急转弯道路。
    感知模块是基于采用深度网络并结合有限数据的可视化检测技术。
    因此，在我们发布更好的网络之前，驾驶员应该小心驾驶并控制好车辆方向而不能依赖与自动驾驶。
    请在安全和限制区域进行试驾。

- ***推荐道路***
	- ***道路两侧有清晰的白色车道线***

- ***禁止***
	- ***急转弯道路***
	- ***没有车道线标记的道路***
	- ***路口***
	- ***对接点或虚线车道线***
	- ***公共道路***

## 感知模块
每个模块的流程图如下所示。

![Image](https://github.com/ApolloAuto/apollo/blob/master/docs/specs/images/perception_flow_chart_apollo_3.0.png)

**图 1: Apollo 3.0的流程图**

### 深度网络
    深度网络摄取图像并为Apollo 3.0提供两个检测输出，车道线和对象。
    目前，对深度学习中使用单一任务还是协同训练任务还存在一些争议。
    诸如车道检测网络或物体检测网络的单一网络通常比一个协同训练的多任务网络执行得更好。
    然而，在给定有限资源的情况下，多个单独的网络将是昂贵的并且在处理中消耗更多时间。
    因此，对于经济设计而言，协同训练是不可避免的，并且在性能上会有一些妥协。
    在 Apollo 3.0, YOLO [1][2] 被用作对象和车道线检测的基础网络。
    该对象具有车辆、卡车、骑车人和行人类别，并由表示成具有方向信息的2-D边界框。
    通过使用具有一些修改的相同网络进行分段来检测车道线。
    对于整条车道线，我们有一个单独的网络，
    以提供更长的车道线，无论是车道线是离散的还是连续的。


### 物体识别/跟踪
    在交通场景中，有两类物体: 静态物体和动态物体。
    静态物体包括车道线、交通信号灯以及数以千计的以各种语言写成的交通标示。
    除了驾驶之外，道路上还有多个地标，主要用于视觉定位，包括路灯，障碍物，道路上的桥梁或任何天际线。
    对于静态物体，Apollo 3.0将仅检测车道线.

    在动态物体中，Apollo在路上关心乘用车，卡车，骑自行车者，行人或任何其他物体，包括动物或身体部位。
    Apollo还可以根据物体所在的车道对物体进行分类。
    最重要的物体是CIPV（路径中最近的物体）。下一个重要对象将是相邻车道中的物体。


#### 2D-to-3D 边界框
    给定一个2D盒子，其3D大小和相机方向，该模块搜索相机坐标系统中的3D位置，
    并使用该2D盒子的宽度，高度或2D区域估计精确的3D距离。
    该模块可在没有准确的外部相机参数的情况下工作。

#### 对象跟踪
    对象跟踪模块利用多种信息，例如3D位置，2D图像补丁，2D框或深度学习ROI特征。
    跟踪问题通过有效地组合线索来表达为多个假设数据关联，
    以提供路径和检测到的对象之间的最正确关联，从而获得每个对象的正确ID关联。

### 车道检测/追踪
    在静态对象中，我们在Apollo 3.0中将仅处理通道线。
    该车道用于纵向和横向控制。
    车道本身引导横向控制，并且在车道内的对象引导纵向控制。

#### 车道线
    我们有两种类型的车道线，车道标记段和整个车道线。
    车道标记段用于视觉定位，整个车道线用于使车辆保持在车道内。
    该通道可以由多组折线表示，例如下一个左侧车道线，左侧线，右侧线和下一个右侧线。
    给定来自深度网络的车道线热图，通过阈值化生成分段的二进制图像。
    该方法首先找到连接的组件并检测内部轮廓。
    然后，它基于自我车辆坐标系的地面空间中的轮廓边缘生成车道标记点。
    之后，它将这些车道标记与具有相应的相对空间（例如，左（L0），右（R0），下左（L1），下（右）（L2）等）标签的若干车道线对象相关联。

### CIPV (最近路径车辆)
    CIPV是当前车道中最接近的车辆。
    对象由3D边界框表示，其从上到下视图的2D投影将对象定位在地面上。
    然后，检查每个对象是否在当前车道中。
    在当前车道的对象中，最接近的一个将被选为CIPV。

### 跟车
    跟车是跟随前车的一种策略。
    从跟踪对象和当前车辆运动中，估计对象的轨迹。
    该轨迹将指导对象如何在道路上作为一组移动并且可以预测未来的轨迹。
    有两种跟车尾随，一种是跟随特定车辆的纯尾随，
    另一种是CIPV引导的尾随，当检测到无车道线时，当前车辆遵循CIPV的轨迹。 

输出可视化的快照如图2所示。 
![Image](https://github.com/ApolloAuto/apollo/blob/master/docs/specs/images/perception_visualization_apollo_3.0.png)

**图 2: Apollo 3.0中感知输出的可视化。左上角是基于图像的输出。左下角显示了对象的3D边界框。左图显示了车道线和物体的三维俯视图。CIPV标有红色边框。黄线表示每辆车的轨迹**

### 雷达 + 摄像头融合
    给定多个传感器，它们的输出应以协同方式组合。
    Apollo 3.0，介绍了一套带雷达和摄像头的传感器。
    对于此过程，需要校准两个传感器。每个传感器都将使用Apollo 2.0中介绍的相同方法进行校准。
    校准后，输出将以3-D世界坐标表示，每个输出将通过它们在位置，大小，时间和每个传感器的效用方面的相似性进行融合。
    在学习了每个传感器的效用函数后，摄像机对横向距离的贡献更大，雷达对纵向距离测量的贡献更大。
    异步传感器融合算法也作为选项提供。

### 伪车道
    所有车道检测结果将在空间上临时组合以诱导伪车道，
    该车道将被反馈到规划和控制模块。
    某些车道线在某帧中不正确或缺失。
    为了提供平滑的车道线输出，使用车辆里程测量的历史车道线。
    当车辆移动时，保存每个帧的里程表，并且先前帧中的车道线也将保存在历史缓冲器中。
    检测到的与历史车道线不匹配的车道线将被移除，历史输出将替换车道线并提供给规划模块。

### 超声波传感器
    Apollo 3.0支持超声波传感器。每个超声波传感器通过CAN总线提供被检测对象的距离。
    来自每个超声波传感器的测量数据被收集并作为ROS主题广播。
    将来，在融合超声波传感器后，物体和边界的地图将作为ROS的输出发布。

## 感知输出
PnC的输入将与之前基于激光雷达的系统的输入完全不同。

- 车道线输出
	- 折线和/或多项式曲线
	- 车道类型按位置：L1（左下车道线），L0（左车道线），R0（右车道线），R1（右下车道线

- 对象输出
	- 3D长方体
	- 相对速度和方向
	- 类型：CIPV，PIHP，其他
	- 分类：汽车，卡车，自行车，行人
	- Drops：物体的轨迹

世界坐标是3D中的当前车辆坐标，其中后中心轴是原点。

## 参考
[1] J Redmon, S Divvala, R Girshick, A Farhadi, "你只看一次：统一的实时物体检测" CVPR 2016

[2] J Redmon, A Farhadi, "YOLO9000: 更好, 更快, 更强," arXiv preprint

# ===================================
# 交通信号灯感知

本文档详细的介绍了Apollo2.0中交通信号感知模块的工作原理。

## 简介

交通信号灯感知模块通过使用摄像头提供精确全面的路面交通信号灯状态。

通常情况下，交通信号灯有3种状态：

- 红
- 黄
- 绿

然而当信号灯不能正常工作时，它可能是黑色的或者闪烁着红灯或黄灯。有时候在摄像头的视野内找不到信号灯，从而导致无法正确检测信号灯状态。

为了覆盖全部的情况，交通信号灯感知模块提供了5种信号灯状态输出：

- 红
- 黄
- 绿
- 黑
- 未知

该模块的高精地图功能反复的检测车辆前方是否有信号灯出现。在给定车辆的位置后，可以通过查询高精地图获取信号灯的边界，并用边界上的4个点来表示信号灯。如果存在信号灯，则信号灯位置信息将从世界坐标系投射到图片坐标系。

Apollo已经证明了仅仅使用一个固定视野的摄像头无法识别所有的信号灯。存在这种限制的原因是：

- 感知范围应该大于100米
- 信号灯的高度和路口的宽度变化范围很大

结果是Apollo2.0使用了2个摄像头来扩大感知范围。

-  一个**远距摄像头**，焦距是25毫米，被用来观察前方远距离的信号灯。远距摄像头捕获的信号灯在图片上展现的非常大而且容易被检测。但是远距摄像头的视野有限制，如果路线不够直或者车辆太过于靠近信号灯，经常无法拍摄到信号灯。


- 一个**广角摄像头**。焦距是6毫米，是对远距摄像头视野不足的补充。

该模块会根据当前信号灯的投射状态决定使用哪个摄像头。虽然只有两个摄像头，但是该模块的算法被设计的可以控制多个摄像头。

下述图片展示了使用远距摄像头（上图）和广角摄像头（下图）检测到信号灯的图片。

![telephoto camera](https://github.com/ApolloAuto/apollo/blob/master/docs/specs/images/traffic_light/long.jpg)


![wide angle camera](https://github.com/ApolloAuto/apollo/blob/master/docs/specs/images/traffic_light/short.jpg)


# 数据管道

数据管道有两个主要的部分，会在下面章节中介绍
- 预处理阶段
  - 信号灯投射
  - 摄像头选择
  - 图像和信号灯缓存同步
- 处理阶段
  - 调整—提供精确的信号灯边界盒
  - 识别—提供每个边界盒的颜色
  - 修正—根据时间顺序关系修正颜色

## 预处理阶段

没有必要在每一帧的图像中去检测信号灯。信号灯的变化频率是很低的而且计算机的资源也有限。通常，从不同摄像头输入的图像信息会几乎同时的到达，但是只有一个会进入管道的处理阶段。因此图像的遴选和匹配是很必要的。

### 输入输出

本章节介绍了预处理阶段的输入输出数据。输入数据可以通过订阅Apollo相关模块数据来获得，或者直接读取本地的存储文件。输出数据被传输到下一层的处理阶段。

#### 输入数据

- 可以通过订阅以下topic来获取不同摄像头的图像数据：

    - `/apollo/sensor/camera/traffic/image_long`
    - `/apollo/sensor/camera/traffic/image_short`

- 定位信息，通过查询以下topic获得：
    - `/tf`

- 高精地图

- 校准结果

#### 输出数据

  - 被选择的摄像头输出的的图像信息
  - 从世界坐标系投射到图像坐标系的信号灯边界盒
  
### 摄像头选择
  
使用一个唯一的ID和其边界上的4个点来表示信号灯，每个点都是世界坐标系中的3维坐标点。

下例展示了一个典型的信号灯记录信息`signal info`。给出车辆位置后，4个边界点可以通过查询高精地图获得。

```protobuf
signal info:
id {
  id: "xxx"
}
boundary {
  point { x: ...  y: ...  z: ...  }
  point { x: ...  y: ...  z: ...  }
  point { x: ...  y: ...  z: ...  }
  point { x: ...  y: ...  z: ...  }
}
```

3维世界坐标系中的边界点随后被投射到每个摄像头图像的2维坐标系。对每个信号灯而言，远距摄像头图像上展示的4个投射点区域更大，这比广角摄像头更容易检测信号灯。最后会选择具有最长的焦距且能够看到所有信号灯的摄像头图片作为输出图像。投射到该图像上的信号边界盒将作为输出的边界盒。

被选择的摄像头的ID和时间戳缓存在队列中：


 ``` C++
struct ImageLights {
  CarPose pose;
  CameraId camera_id;
  double timestamp;
  size_t num_signal;
  ... other ...
};
 ```
 
 至此，我们需要的所有信息包括定位信息、校准结果和高精地图。因为投射不依赖于图像的内容，所以选择可以在任何时间完成。在图像信息到达时进行选择仅仅是为了简单。而且，并不是图像信息一到达就要进行选择，通常会设置选择的时间间隔。
 
 
### 图像同步

图像信息包含了摄像头ID和时间戳。摄像头ID和时间戳的组合用来找到可能存在的缓存信息。如果能在缓存区找到和该图像的摄像头ID一样且时间戳相差很小的缓存信息，则该图像会被传输到处理阶段。所有不合适的缓存信息会被丢弃。

## 处理阶段

该阶段分为3个步骤，每个步骤重点执行一个任务：

- 调整 — 在ROI中检测信号灯边界盒
- 识别 — 鉴别边界盒的颜色
- 修正 — 根据信号灯颜色的时间顺序关系修正颜色

### 输入输出

本章节介绍处理阶段的输入和输出数据。输入数据从预处理阶段获得，输出数据作为鉴别信号灯的结果。

#### 输入数据

- 被选择的摄像头图像信息
- 一组边界盒信息

#### 输出数据

  - 一组带有颜色标签的边界盒信息


### 调整

被定位信息、校准信息和高精地图信息影响的投射点 ***不是完全可靠的*** 。通过投射的信号灯位置计算的一个大的兴趣区域（Region of Interest ROI）被用来确定信号灯精确的边界盒。

在下述图片中，蓝色的长方形表示被投射的信号灯的边界盒，实际上和信号灯的准确位置有一定的偏差。大的黄色长方形是ROI。

![example](https://github.com/ApolloAuto/apollo/blob/master/docs/specs/images/traffic_light/example.jpg)

信号灯检测是一个常规的卷积神经网络检测任务，它接收带有ROI信息的图像作为输入数据，顺序输出边界盒。输出结果中的信号灯数量可能多于输入数据。

Apollo会根据输入信号灯的位置、形状及检测的评分选择合适的信号灯。如果CNN在ROI内找不到任何的信号灯，则输入数据中的信号灯将被标记为未知，且跳过剩下的两个步骤。

### 识别

信号灯识别是一个常规的卷积神经网络鉴别任务，它接收带有ROI信息的图像和一组边界盒信息作为输入数据。输出数据是一个`$4\times n$ vector`， 表示每个边界盒是黑色、红色、黄色和绿色的概率。
当且仅当概率足够大时，有最大概率的类别会被识别为信号灯的状态。否则信号灯状态被设置为未知，表示状态未确定。

### 修正

因为信号灯可能会闪烁或者被遮挡，并且识别阶段也 ***并不是*** 完美的，输出的信号灯状态可能不是真正的状态。修正信号灯状态是很有必要的。

如果修正器接收到一个确定的信号灯状态例如红色或者绿色，则修正器保存该状态并直接输出。如果接收到黑色或者未知，修正器会检测状态保存列表。如果信号灯状态已经确定持续了一段时间，那么将保存的状态输出。否则将黑色或者未知输出。

因为时间顺序关系的存在，黄色只会在绿色之后红色之前出现，所以为了安全的考虑，在绿色出现之前任何红色之后的黄色都会被设置为红色。



3D 障碍物感知
====================

Apollo解决的障碍物感知问题：

- 高精地图ROI过滤器（HDMap ROI Filter）
- 基于卷积神经网络分割（CNN Segmentation）
- MinBox 障碍物边框构建（MinBox Builder）
- HM对象跟踪（HM Object Tracker）

高精地图ROI过滤器
-------------------------------------

ROI（The Region of Interest）指定从高精地图检索到包含路面、路口的可驾驶区域。高精地图 ROI 过滤器（往下简称“过滤器”）处理在ROI之外的激光雷达点，去除背景对象，如路边建筑物和树木等，剩余的点云留待后续处理。

给定一个高精地图，每个激光雷达点的关系意味着它在ROI内部还是外部。
每个激光雷达点可以查询一个车辆周围区域的2D量化的查找表（LUT）。过滤器模块的输入和输出汇总于下表。

  |输入                                                                     |输出                                                                     |
  |------------------------------------------------------------------------- |---------------------------------------------------------------------------|
  |点云: 激光雷达捕捉的3D点数据集           | 由高精地图定义的ROI内的输入点索引。      |
  |高精地图: 多边形集合，每个多边形均含有一个有序的点集。     |                                       |
 
一般来说，Apollo 高精地图 ROI过滤器有以下三步：

1. 坐标转换
2. ROI LUT构造
3. ROI LUT点查询

### 坐标转换

对于（高精地图ROI）过滤器来说，高精地图数据接口被定义为一系列多边形集合，每个集合由世界坐标系点组成有序点集。高精地图ROI点查询需要点云和多边形处在相同的坐标系，为此，Apollo将输入点云和HDMap多边形变换为来自激光雷达传感器位置的地方坐标系。

### ROI LUT构造

Apollo采用网格显示查找表（LUT），将ROI量化为俯视图2D网格，以此决定输入点是在ROI之内还是之外。

如图1所示，该LUT覆盖了一个矩形区域，该区域位于高精地图边界上方，以普通视图周围的预定义空间范围为边界。它代表了与ROI关联网格的每个单元格（如用1/0表示在ROI的内部/外部）。 为了计算效率，Apollo使用 **扫描线算法**和 **位图编码**来构建ROI LUT。

<img src="https://raw.githubusercontent.com/ApolloAuto/apollo/master/docs/specs/images/3d_obstacle_perception/roi_lookup_table.png">
<div align=center>图 1 ROI显示查找表（LUT）</div>

蓝色线条标出了高精地图ROI的边界，包含路表与路口。红色加粗点表示对应于激光雷达传感器位置的地方坐标系原始位置。2D网格由8*8个绿色正方形组成，在ROI中的单元格，为蓝色填充的正方形，而之外的是黄色填充的正方形。

### ROI LUT点查询

基于ROI LUT，查询每个输入点的关系使用两步认证。对于点查询过程，Apollo数据编译输出如下，:

1. 检查点在ROI LUT矩形区域之内还是之外。
2. 查询LUT中相对于ROI关联点的相应单元格。
3. 收集属于ROI的所有点，并输出其相对于输入点云的索引。

用户定义的参数可在配置文件`modules/perception/model/hdmap_roi_filter.config`中设置，HDMap ROI Filter 参数使用参考如下表格：

  |参数名称      |使用                                                                          |默认     |
  |------------------- |------------------------------------------------------------------------------ |------------|
  |range           | 基于LiDAR传感器点的2D网格ROI LUT的图层范围），如(-70, 70)*(-70, 70) |70.0 米 |
  |cell_size           | 用于量化2D网格的单元格的大小。                                   |0.25 米  |
  |extend_dist         | 从多边形边界扩展ROI的距离。                 |0.0 米   |

基于CNN的障碍物分割
------------------------------------------------
高精地图 ROI过滤之后，Apollo得到已过滤、只包含属于ROI内的点云，大部分背景障碍物，如路侧的建筑物、树木等均被移除，ROI内的点云被传递到分割模块。分割模块检测和划分前景障碍物，例如汽车，卡车，自行车和行人。

  |输入                                                                        |输出                             |
  |----------------------------------------------------------------------------|---------------------------------------------------------------|
  |点云（3D数据集）                                         |对应于ROI中的障碍物对象数据集    |
  |表示在HDMap中定义的ROI内的点的点索引       |                               |                                                                              
Apollo 使用深度卷积神经网络提高障碍物识别与分割的精度，障碍物分割包含以下四步：
- 通道特征提取
- 基于卷积神经网络的障碍物预测
- 障碍物集群
- 后期处理

卷积神经网络详细介绍如下：

### 通道特征提取

给定一个点云框架，Apollo在地方坐标系中构建俯视图（即投影到X-Y平面）2D网格。 基于点的X、Y坐标，相对于LiDAR传感器原点的预定范围内，每个点被量化为2D网格的一个单元。 量化后，Apollo计算网格内每个单元格中点的8个统计测量，这将是下一步中传递给CNN的输入通道特征。 

计算的8个统计测量：

1. 单元格中点的最大高度
2. 单元格中最高点的强度
3. 单元格中点的平均高度
4. 单元格中点的平均强度
5. 单元格中的点数
6. 单元格中心相对于原点的角度
7. 单元格中心与原点之间的距离
8. 二进制值标示单元格是空还是被占用

### 基于卷积神经网络的障碍物预测

基于上述通道特征，Apollo使用深度完全卷积神经网络（FCNN）来预测单元格障碍物属性，包括潜在物体中心的偏移位移（称为中心偏移）、对象性
积极性和物体高度。如图2所示，网络的输入为 *W* x *H* x *C* 通道图像，其中：

- *W* 代表网格中的列数
- *H* 代表网格中的行数
- *C* 代表通道特征数

完全卷积神经网络由三层构成：
- 下游编码层（特征编码器）
- 上游解码层（特征解码器）
- 障碍物属性预测层（预测器）

特征编码器将通道特征图像作为输入，并且随着特征抽取的增加而连续**下采样**其空间分辨率。 然后特征解码器逐渐对特征图像 **上采样**到输入2D网格的空间分辨率，可以恢复特征图像的空间细节，以促进单元格方向的障碍物位置、速度属性预测。 根据具有非线性激活（即ReLu）层的堆叠卷积/分散层来实现 **下采样**和 **上采样**操作。

<div align=center><img src="https://raw.githubusercontent.com/ApolloAuto/apollo/master/docs/specs/images/3d_obstacle_perception/FCNN.png" width="99%"></div>

<div align=center>图 2 FCNN在单元格方向上的障碍物预测</div>

### 障碍物聚类
在基于CNN的预测之后，Apollo获取单个单元格的预测信息。利用四个单元对象属性图像，其中包含：

- 中心偏移
- 对象性
- 积极性
- 对象高度

为生成障碍物，Apollo基于单元格中心偏移，预测构建有向图，并搜索连接的组件作为候选对象集群。

如图3所示，每个单元格是图的一个节点，并且基于单元格的中心偏移预测构建有向边，其指向对应于另一单元的父节点。

如图3，Apollo采用压缩的联合查找算法（Union Find algorithm ）有效
