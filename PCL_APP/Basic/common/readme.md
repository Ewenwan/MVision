# pcl常规用法
## PDC文件格式
    点云数据可以用ASCII码的形式存储在PCD文件中（
    关于该格式的描述可以参考链接：The PCD (Point Cloud Data) file format）。
    为了生成三维点云数据，在excel中用rand()函数生成200行0-1的小数，ABC三列分别代表空间点的xyz坐标。

    PDC文件格式
    # .PCD v.7 - Point Cloud Data file format
    VERSION .7        
    FIELDS x y z     
    SIZE 4 4 4         
    TYPE F F F         
    COUNT 1 1 1     
    WIDTH 200        
    HEIGHT 1        
    VIEWPOINT 0 0 0 1 0 0 0
    POINTS 200        
    DATA ascii        
    0.88071666    0.369209703    0.062937221
    0.06418104    0.579762553    0.221359779
    ...
    ...
    0.640053058    0.480279041    0.843647334
    0.245554712    0.825770496    0.626442137

    内容格式
    FIELDS x y z                                # XYZ data
    FIELDS x y z rgb                            # XYZ + colors
    FIELDS x y z normal_x normal_y normal_z     # XYZ + surface normals
    FIELDS j1 j2 j3                             # moment invariants
    存储大小
    SIZE - specifies the size of each dimension in bytes. Examples:

        unsigned char/char has 1 byte
        unsigned short/short has 2 bytes
        unsigned int/int/float has 4 bytes
        double has 8 bytes
    数据类型
    TYPE - specifies the type of each dimension as a char. The current accepted types are:

        I - represents signed types int8 (char), int16 (short), and int32 (int)
        U - represents unsigned types uint8 (unsigned char), uint16 (unsigned short), uint32 (unsigned int)
        F - represents float types

    视角
    The viewpoint information is specified as a translation (tx ty tz) + quaternion (qw qx qy qz). The default value is:
    VIEWPOINT 0 0 0 1 0 0 0


## CSV格式的ply文件类

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

[pcd2ply](pcd2ply.cpp)

[ply2pcd](https://github.com/PointCloudLibrary/pcl/blob/master/tools/ply2pcd.cpp)

## 点云变换
    点云坐标变换
    变换矩阵的其次表示
    [R t]
    绕Z轴旋转 45度 （顺时针） 沿着x轴平移 2.5个单位

    x' = x * cos(cet) + y * sin(cet) + 0 * z
    y' = y * cos(cet) - x * sin(cet) + 0 *z
    z' = 0*x +0*y + 1*z
    所以旋转矩阵为
    cos(cet) -sin(cet) 0
    sin(cet)  cos(cet) 0
    0           0      1
    平移矩阵为
    2.5
    0
    0

    点云数据可以用ASCII码的形式存储在PCD文件中（
    关于该格式的描述可以参考链接：The PCD (Point Cloud Data) file format）。
    为了生成三维点云数据，在excel中用rand()函数生成200行0-1的小数，ABC三列分别代表空间点的xyz坐标。

[点云变换](transformPointC.cpp)

## 连接两个点云中的字段或数据形成新点云
    pcl::concatenateFields 链接字段 
    如何连接两个不同点云为一个点云，进行操作前要确保两个数据集中字段的类型相同和维度相等，
    同时了解如何连接两个不同点云的字段（例如颜色 法线）这种操作的强制约束条件是两个数据集中点的数目必须一样，
    例如：点云A是N个点XYZ点，点云B是N个点的RGB颜色点，则连接两个字段形成点云C是N个点xyzrgb类型
    新建文件concatenate_clouds.cpp  CMakeLists.txt

    字段间连接是在行的基础后连接 （增添点的其他信息），
    而点云连接是在列的下方连接（扩展点的数量），
    最重要的就是要考虑维度问题，同时每个点云都有XYZ三个数据值

[点云连接](concatenate_clouds.cpp)

## 计算点云重心
    点云的重心是一个点坐标，计算出云中所有点的平均值。
    你可以说它是“质量中心”，它对于某些算法有多种用途。
    如果你想计算一个聚集的物体的实际重心，
    记住，传感器没有检索到从相机中相反的一面，
    就像被前面板遮挡的背面，或者里面的。
    只有面对相机表面的一部分。
    
[计算点云重心](compute_centroid.cpp)

## 提取索引
## 点是否合法
## 复制点云 
## 最大最小的索引
## ...
