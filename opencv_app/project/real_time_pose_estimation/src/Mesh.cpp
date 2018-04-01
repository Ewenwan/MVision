/*
网格数据Mesh类(class) 
实际　CsvReader　类读取了文件

读取 ply 顶点（vertices） 面（faces） Mesh.cpp
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

 */

#include "Mesh.h"
#include "CsvReader.h"


// --------------------------------------------------- //
//                   TRIANGLE CLASS 三角形类            //
// --------------------------------------------------- //

// 构造函数
Triangle::Triangle(int id, cv::Point3f V0, cv::Point3f V1, cv::Point3f V2)
{
  id_ = id; v0_ = V0; v1_ = V1; v2_ = V2;// 三维空间点 三个点 组成　一个　三角形　 
}

// 析构函数
Triangle::~Triangle()
{
  // TODO Auto-generated destructor stub
}


// --------------------------------------------------- //
//                     RAY CLASS　线段类　Ray类          //
// --------------------------------------------------- //

// 构造函数
Ray::Ray(cv::Point3f P0, cv::Point3f P1) {
  p0_ = P0; p1_ = P1;
}
// 两个三维空间点　组成一条线段　三个线段组成一个三角形面
// 析构函数
Ray::~Ray()
{
  // TODO Auto-generated destructor stub
}


// --------------------------------------------------- //
//                 OBJECT MESH CLASS                   //
// --------------------------------------------------- //

// 物体网格类　模型　Mesh构造函数
Mesh::Mesh() : list_vertex_(0) , list_triangles_(0)
{
  id_ = 0;
  num_vertexs_ = 0;//顶点数量
  num_triangles_ = 0;//三角形面　数量
}

// 物体网格类　模型　Mesh　　析构函数
Mesh::~Mesh()
{
  // TODO Auto-generated destructor stub
}


// CSV文件　载入物体网格数据模型　顶点坐标　面　Load a CSV with *.ply format 
void Mesh::load(const std::string path)
{

  //  创建　CSV　文件读取器
  CsvReader csvReader(path);

  // 清空之前的数据
  list_vertex_.clear();//顶点
  list_triangles_.clear();//三角形面　

  // 读取 .ply 文件
  csvReader.readPLY(list_vertex_, list_triangles_);

  // 更新　物体网格数据模型　数据记录　参数
  num_vertexs_ = (int)list_vertex_.size();
  num_triangles_ = (int)list_triangles_.size();

}

