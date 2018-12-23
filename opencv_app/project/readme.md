# opencv项目

道路标线检测

交通灯检测

级联回归人脸笑脸检测 

[车道线和交通标志检测](https://github.com/ZhangChaoZhong/LaneMarkings_TrafficSigns_Detection)

基于 特征匹配单应变换的实时 3D物体跟踪

# 纹理对象的实时姿态估计

## 【1】 ply格式数据 储存立体扫描结果的三维数值

      ply文件不支持中文格式的文件名字,所以在使用过程中避免使用中文来命名。

      https://www.cnblogs.com/liangliangdetianxia/p/4000295.html
      PLY是一种电脑档案格式，全名为多边形档案（Polygon File Format）或 
      斯坦福三角形档案（Stanford Triangle Format）。 

      该格式主要用以储存立体扫描结果的三维数值，透过多边形片面的集合描述三维物体，
      与其他格式相较之下这是较为简单的方法。它可以储存的资讯包含颜色、
      透明度、表面法向量、材质座标与资料可信度，并能对多边形的正反两面设定不同的属性。

      在档案内容的储存上PLY有两种版本，分别是纯文字（ASCII）版本与二元码（binary）版本，
      其差异在储存时是否以ASCII编码表示元素资讯。

      每个PLY档都包含档头（header），用以设定网格模型的“元素”与“属性”，
      以及在档头下方接着一连串的元素“数值资料”。
      一般而言，网格模型的“元素”
      就是顶点（vertices）、
      面（faces），另外还可能包含有
      边（edges）、
      深度图样本（samples of range maps）与
      三角带（triangle strips）等元素。
      无论是纯文字与二元码的PLY档，档头资讯都是以ASCII编码编写，
      接续其后的数值资料才有编码之分。PLY档案以此行：

      ply     开头作为PLY格式的识别。接着第二行是版本资讯，目前有三种写法：

      format ascii 1.0
      format binary_little_endian 1.0
      format binary_big_endian 1.0
             // 其中ascii, binary_little_endian, binary_big_endian是档案储存的编码方式，
             // 而1.0是遵循的标准版本（现阶段仅有PLY 1.0版）。

      comment This is a comment!   // 使用'comment'作为一行的开头以编写注解
      comment made by anonymous
      comment this file is a cube

      element vertex 8    //  描述元素  element <element name> <number in file>   8个顶点
                          //  以下 接续的6行property描述构成vertex元素的数值字段顺序代表的意义，及其资料形态。
      property float32 x  //  描述属性  property <data_type> <property name 1>
      property float32 y  //  每个顶点使用3个 float32 类型浮点数（x，y，z）代表点的坐标
      property float32 z

      property uchar blue // 使用3个unsigned char代表顶点颜色，颜色顺序为 (B, G, R)
      property uchar green
      property uchar red

      element face 12       
      property list uint8 int32 vertex_index
            // 12 个面(6*2)   另一个常使用的元素是面。
            // 由于一个面是由3个以上的顶点所组成，因此使用一个“顶点列表”即可描述一个面, 
            // PLY格式使用一个特殊关键字'property list'定义之。 
      end_header              // 最后，标头必须以此行结尾：

      // 档头后接着的是元素资料（端点座标、拓朴连结等）。在ASCII格式中各个端点与面的资讯都是以独立的一行描述

      0 0 0                   // 8个顶点 索引 0~7
      0 25.8 0
      18.9 0 0
      18.9 25.8 0
      0 0 7.5
      0 25.8 7.5
      18.9 0 7.5
      18.9 25.8 7.5

      3 5 1 0            // 前面的3表示3点表示的面   有的一个面 它用其中的三个点 表示了两次  6*2=12
      3 5 4 0            // 后面是上面定点的 索引 0~7
      3 4 0 2
      3 4 6 2
      3 7 5 4
      3 7 6 4
      3 3 2 1
      3 1 2 0
      3 5 7 1
      3 7 1 3
      3 7 6 3
      3 6 3 2



## 【2】由简单的 长方体 顶点  面 描述的ply文件 和 物体的彩色图像 生产 物体的三维纹理模型文件


        【a】手动指定 图像中 物体顶点的位置（得到二维像素值位置）
          ply文件 中有物体定点的三维坐标

          由对应的 2d-3d点对关系
          u
          v  =  K × [R t] X
          1               Y
              Z
              1
          K 为图像拍摄时 相机的内参数

          世界坐标中的三维点(以文件中坐标为(0,0,0)某个定点为世界坐标系原点)
          经过 旋转矩阵R  和平移向量t 变换到相机坐标系下
          在通过相机内参数 变换到 相机的图像平面上

        【b】由 PnP 算法可解的 旋转矩阵R  和平移向量t 

        【c】把从图像中得到的纹理信息 加入到 物体的三维纹理模型中

          在图像中提取特征点 和对应的描述子
          利用 内参数K 、 旋转矩阵R  和平移向量t  反向投影到三维空间
             标记 该反投影的3d点 是否在三维物体的 某个平面上

        【d】将 2d-3d点对 、关键点 以及 关键点描述子 存入物体的三维纹理模型中





