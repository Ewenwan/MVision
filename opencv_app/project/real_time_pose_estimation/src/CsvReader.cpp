/*
读取
CSV格式的ply文件类
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
#include "CsvReader.h"

// 默认构造函数
CsvReader::CsvReader(const string &path, const char &separator){
    _file.open(path.c_str(), ifstream::in);//打开文件
    _separator = separator;
}

// 读取ply文件　　得到　顶点列表　　和三角形面　列表
void CsvReader::readPLY(vector<Point3f> &list_vertex, vector<vector<int> > &list_triangles)
{
    std::string line, tmp_str, n;
    int num_vertex = 0, num_triangles = 0;
    int count = 0;
    bool end_header = false;
    bool end_vertex = false;

    // Read the whole *.ply file
    while (getline(_file, line)) {//读取文件的每一行
    stringstream liness(line);

    // 读取ply文件头　read header
    if(!end_header)
    {
        getline(liness, tmp_str, _separator);//分割每一行
        if( tmp_str == "element" )//  描述元素  element <element name> <number in file>  
        {
            getline(liness, tmp_str, _separator);//
            getline(liness, n);
            if(tmp_str == "vertex") num_vertex = StringToInt(n);// 顶点vertices　　8个顶点 element vertex 8 
            if(tmp_str == "face") num_triangles = StringToInt(n);// 面face　　　　　　　12个三角形面　element face 12 
        }
        if(tmp_str == "end_header") end_header = true;// 标头必须以此 end_header 行结尾
    }

    // read file content
    else if(end_header)
    {
	 // 顶点vertices
         // read vertex and add into 'list_vertex'
         if(!end_vertex && count < num_vertex)
         {
             string x, y, z;//  0 25.8 0 
             getline(liness, x, _separator);
             getline(liness, y, _separator);
             getline(liness, z);

             cv::Point3f tmp_p;// 三维空间点
             tmp_p.x = (float)StringToFloat(x);// Utils.h 
             tmp_p.y = (float)StringToFloat(y);
             tmp_p.z = (float)StringToFloat(z);
             list_vertex.push_back(tmp_p);

             count++;
             if(count == num_vertex)
             {
                 count = 0;
                 end_vertex = !end_vertex;
             }
         }
 	 // 面face
         // read faces and add into 'list_triangles'
         else if(end_vertex  && count < num_triangles)
         {
             string num_pts_per_face, id0, id1, id2;
             getline(liness, num_pts_per_face, _separator);
             getline(liness, id0, _separator);// 索引 
             getline(liness, id1, _separator);
             getline(liness, id2);

             std::vector<int> tmp_triangle(3);
             tmp_triangle[0] = StringToInt(id0);
             tmp_triangle[1] = StringToInt(id1);
             tmp_triangle[2] = StringToInt(id2);
             list_triangles.push_back(tmp_triangle);
             count++;
      }
    }
  }
}
