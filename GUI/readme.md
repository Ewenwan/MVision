# 可视化界面软件
    PCL Visialization,  Pangolin,  ros下的rviz，OPEN-GL，QT
    
[freetype-gl-cpp](https://github.com/Ewenwan/freetype-gl-cpp)
       
[Mastering Qt 5 GUI Programming](https://github.com/PacktPublishing/Mastering-Qt-5-GUI-Programming)

[End-to-End-GUI-development-with-Qt5](https://github.com/PacktPublishing/End-to-End-GUI-development-with-Qt5)
    
[Learn QT 5](https://github.com/PacktPublishing/Learn-Qt-5)
    
#  Pangolin 用于可视化和用户接口 基于opengl

## 安装，
	是一款开源的OPENGL显示库，可以用来视频显示、而且开发容易。
	是对OpenGL进行封装的轻量级的OpenGL输入/输出和视频显示的库。
	可以用于3D视觉和3D导航的视觉图，
	可以输入各种类型的视频、并且可以保留视频和输入数据用于debug。

[github](https://github.com/Ewenwan/Pangolin)

[官方样例demo](https://github.com/stevenlovegrove/Pangolin/tree/master/examples)

[Pangolin函数的使用](http://docs.ros.org/fuerte/api/pangolin_wrapper/html/namespacepangolin.html)


[Pangolin安装问题 安装依赖项](http://www.cnblogs.com/liufuqiang/p/5618335.html)

> 安装Pangolin的依赖项

	1. glew
	   sudo apt-get install libglew-dev
	2. CMake：
	   sudo apt-get install cmake
	3. Boost：
	   sudo apt-get install libboost-dev libboost-thread-dev libboost-filesystem-dev
	4. Python2 / Python3：
	   sudo apt-get install libpython2.7-dev
	5. build-essential
	   sudo apt-get install build-essential
> 安装Pangolin

	git clone https://github.com/stevenlovegrove/Pangolin.git
	cd Pangolin
	mkdir build
	cd build
	cmake ..
	make -j
	sudo make install
# OpenGL
[中文教程地址](https://learnopengl-cn.github.io/)

[github 代码](https://github.com/JoeyDeVries/LearnOpenGL/tree/master/src)

[OpenGL-Examples](https://github.com/progschj/OpenGL-Examples)

```c
#include <iostream>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

namespace {

void Init(void)
{
    glClearColor(1.0, 0.0, 1.0, 0.0);//设置背景颜色为洋红
    glColor3f(0.0f, 1.0f, 0.0f);//设置绘图颜色为绿色
    glPointSize(4.0);//设置点的大小为4*4像素
    glMatrixMode(GL_PROJECTION);//设置合适的矩阵
    glLoadIdentity();
    gluOrtho2D(0.0, 640.0, 0.0, 480.0);
}

void Display(void)
{
    glClear(GL_COLOR_BUFFER_BIT);//清屏
    glBegin(GL_POINTS);
    glVertex2i(289, 190);
    glVertex2i(320, 128);
    glVertex2i(239, 67);
    glVertex2i(194, 101);
    glVertex2i(129, 83);
    glVertex2i(75, 73);
    glVertex2i(74, 74);
    glVertex2i(20, 10);
    glEnd();
    glFlush();
}

} // namespace

int main(int argc, char* argv[])
{
    glutInit(&argc, argv);//初始化工具包
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);//设置显示模式
    glutInitWindowSize(640, 480);//设置窗口大小
    glutInitWindowPosition(100, 150);//设置屏幕上窗口位置
    glutCreateWindow("my first attempt");//打开带标题的窗口
    glutDisplayFunc(&Display);//注册重画回调函数
    Init();
    glutMainLoop();//进入循环

    return 0;
}

# cmakelists.txt
PROJECT(multi_executable_file)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

# 查找OpenGL
FIND_PACKAGE(OpenGL REQUIRED)
IF(OPENGL_FOUND)
	MESSAGE("===== support OpenGL =====")
	MESSAGE(STATUS "OpenGL library status:")
	MESSAGE(STATUS "	include path: ${OPENGL_INCLUDE_DIR}")
	MESSAGE(STATUS "	libraries: ${OPENGL_LIBRARIES}")
	INCLUDE_DIRECTORIES(${OPENGL_INCLUDE_DIR})
ELSE()
	MESSAGE("##### not support OpenGL #####")
ENDIF()

ADD_EXECUTABLE(test_${sample_basename} ${sample})
TARGET_LINK_LIBRARIES(test_${sample_basename} ${OPENGL_LIBRARIES})

```
	
	
	
# QT

	安装命令：
	sudo apt-get install qt4-dev-tools qt4-doc qt4-qtconfig qt4-demos qt4-designer

	关于集成开发环境我觉得QDevelop很不错，它跟Qt Designer结合的很好，而且有提示类成员函数的功能。
	这样，使用Qdevelop编写代码和编译、调试，使用Qt Designer设计界面，开发效率较高。
	运行以下命令安装QDevelop：
	sudo apt-get install qdevelop

	为了连接MySQL数据库，需要安装连接MySQL的驱动程序：
	sudo apt-get install libqt4-sql-mysql

	如果还需要其它的没有默认安装的Qt库，可以在命令行输入
	sudo apt-get install libqt4-
	然后按tab键自动补全，就会列出所有以libqt4- 


	如果还需要画一些数据曲线和统计图表等，而第三方的QWT库提供了这些功能。同样，只需要一个命令即可完成安装：
	sudo apt-get install libqwt5-qt4 libqwt5-qt4-dev 


