# Apollo源码分析
[源码](https://github.com/ApolloAuto/apollo)

# 框架

```
             地图 MAP (HP MAP 高精度地图)
                  |                         |                              |
目的地 goal  ->  导航 routing ---path---> 规划planning ---path---> 感知perception（交通灯/障碍物/）
                  |                                                        |
                  |                                   <-----------预测（障碍物/交通灯/）
                  |
                定位localization             |
                                          控制 control
                                            |  控制命令/传感器反馈
                                          canbus 汽车总线  


```

# 控制 control 模块

介绍

    本模块基于规划和当前的汽车状态，使用不同的控制算法来生成舒适的驾驶体验。控制模块可以在正常模式和导航模式下工作。

输入

    规划轨迹
    车辆状态
    定位
    Dreamview自动模式更改请求
  
输出

    给底盘的控制指令（转向，节流，刹车）。
    
    
    
    
