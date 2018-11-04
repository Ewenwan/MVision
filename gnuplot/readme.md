# gnuplot 一个通用、强大的科学绘图软件
    可以跨平台调用，便携式命令行驱动图形工具。
    
    Gnuplot是一个命令行的交互式绘图工具（command-driven interactive function plotting program）。
    用户通过输入命令，可以逐步设置或修改绘图环境，并以图形描述数据或函数，使我们可以借由图形做更进一步的分析。

    gnuplot是由Colin Kelly和Thomas Williams于1986年开始开发的科学绘图工具，支持二维和三维图形。
    它的功能是把数据资料和数学函数转换为容易观察的平面或立体的图形，它有两种工作方式，
    交互式方式和批处理方式，它可以让使用者很容易地读入外部的数据结果，在屏幕上显示图形，
    并且可以选择和修改图形的画法，明显地表现出数据的特性。
      
[主页](http://www.gnuplot.info/)

[使用手册 英文](http://www.gnuplot.info/docs_5.2/Gnuplot_5.2.pdf)

[中文手册](https://github.com/Ewenwan/MVision/blob/master/gnuplot/gnuplot%E4%B8%AD%E6%96%87%E6%89%8B%E5%86%8C.pdf)

[cpp 接口](https://github.com/Ewenwan/gnuplot-cpp)

# Linux安装编辑
    终端输入命令 $ sudo apt-get install gnuplot 系统自动获取包信息、处理依赖关系，完成安装
    安装完毕后，在终端运行命令 $ gnuplot 进入gnuplot
    系统出现：gnuplot>是提示符，所有gnuplot命令在此输入

    
# 用法
## （1）数据文件格式
    可以通过在plot或splot命令行上指定数据文件的名称（用引号括起来）来显示文件中包含的离散数据。
    数据文件应该以数字列排列数据。列应该只由空白（标签或空格）分隔，（没有逗号）。
    以一个字符开始的行被视为注释，被GNUTRAP忽略。
    数据文件中的空行导致连接数据点的行中断。

## （2）风格定制

    一般需要指明：

    ranges of the axes

    labels of the x and y axes

    style of data point

    style of the lines connecting the data points

    title of the entire plot



    Plots may be displayed in one of styles:

    lines

    points

    linespoints

    impulses

    dots

    steps

    fsteps

    histeps

    errorbars

    xerrorbars

    yerrorbars

    xyerrorbars

    boxes

    boxerrorbars

    boxxyerrorbars

    financebars

    candlesticks

    vector

 

## （1）基础

    set title "Some math functions" // 图片标题

    set xrange [-10:10] // 横坐标范围

    set yrange [-2:2] // 纵坐标范围

    set zeroaxis // 零坐标线

    plot (x/4)*2, sin(x), 1/x // 画函数



## （2）三维曲线

    splot sin(x), tan(x) // 画sin(x)和tan(x)的三维曲线



## （3）多条曲线

    plot sin(x) title 'Sine', tan(x) title 'Tangent' // 画多条曲线，并且分别指定名称



    从不同数据文件中提取信息来画图：

    plot "fileA.dat" using 1:2 title 'data A', \

            "fileB.dat" using 1:3 title 'data B'



## （4）点和线风格

    plot "fileA.dat" using 1:2 title 'data A' with lines, \

            "fileB.dat" using 1:3 title 'data B' with linespoints

    可以使用缩写：

    using title and with can be abbreviated as u t and w.



    plot sin(x) with line linetype 3 linewidth 2

    或plot sin(x) w l lt 3 lw 2

    plot sin(x) with point pointtype 3 pointsize 2

    或plot sin(x) w p pt 3 ps 2



    颜色列表大全：

    http://www.uni-hamburg.de/Wiss/FB/15/Sustainability/schneider/gnuplot/colors.htm

    用法：lc grb "greenyellow"



    线条颜色，点的为pointtype

    linetype 1 // 红色

    linetype 2 // 绿色

    linetype 3 // 蓝色

    linetype 4 // 粉色

    linetype 5 // 比较浅的蓝色

    linetype 6 // 褐色

    linetype 7 // 橘黄色

    linetype 8 // 浅红色



    线条粗细，点的为大小pointsize

    linewidth 1 // 普通的线

    linewidth 2 // 比较粗

    linewidth 3 // 很粗



## （5）常用

    replot // 重绘

    set autoscale // scale axes automatically

    unset label // remove any previous labels

    set xtic auto // set xtics automatically

    set ytic auto // set ytics automatically

    set xtics 2 // x轴每隔2设一个标点

    set xlabel "x axe label"

    set ylabel 'y axe label"

    set label 'Yield Point" at 0.003,260 // 对一个点进行注释

    set arrow from 0.0028,250 to 0.003,280 // 两点之间添加箭头

    set grid // 添加网格

    reset // gnuplot没有缩小，放大后只能reset后重绘，remove all customization

    set key box // 曲线名称加框

    set key top left // 改变曲线名称位置

    set format xy "%3.2f" // x轴和y轴坐标格式，至少有3位，精确到小数点后两位

    quit、q、exit // 退出程序



## （6）绘制多个图

    set size 1,1 // 总的大小

    set origin 0,0 // 总的起点

    set multiplot // 进入多图模式

    set size 0.5,0.5 // 第一幅图大小

    set origin 0,0.5 // 第一幅图起点

    plot sin(x)

    set size 0.5,0.5

    set origin 0,0

    plot 1/sin(x)

    set size 0.5,0.5

    set orgin 0.5,0.5

    plot cos(x)

    set size 0.5,0.5

    set origin 0.5,0

    plot 1/cos(x)

    unset multiplot



## （7）输出文件

    set terminal png size 1024, 768 // 保存文件格式和大小

    set output "file.png" // 保存文件名称

    set terminal X11 // 重新输出到屏幕



## （8）脚本

    a.plt //后缀名为plt

    gnuplot> load 'a.plt'

    或gnuplot a.plt

    save "graph.gp"

    或save "graph.plt"

    传入参数

    call "a.plt" param1 param2

    param1、param2在a.plt中对应于$0、$1

 

## （9）字体设置

    输出错误：

    Could not find/open font when opening font "arial", using internal non-scalable font

    下载RPM包：

    wget http://www.my-guides.net/en/images/stories/fedora12/msttcore-fonts-2.0-3.noarch.rpm

    rpm -ivh msttcore-fonts-2.0-3.noarch.rpm

    设置环境变量：

    修改/etc/profile或~/.bashrc，这样设置可以固定下来。

    export GDFONTPATH="/usr/share/fonts/msttcore"

    export GNUPLOT_DEFAULT_GDFONT="arial"

    . /profile/etc

    OK，现在可以使用arial字体了。


# 命令大全

在linux命令提示符下运行gnuplot命令启动，输入quit或q或exit退出。

 

## 1. plot命令
    gnuplot> plot sin(x) with line linetype 3 linewidth 2 或 
    gnuplot> plot sin(x) w l lt 3 lw 2 %用线画，线的类型（包括颜色与虚线的类型）是3，线的宽度是2，对函数sin(x)作图 
    gnuplot> plot sin(x) with point pointtype 3 pointsize 2 或 
    gnuplot> plot sin(x) w p pt 3 ps 2 %用点画，点的类型（包括颜色与点的类型）是3，点的大小是2 
    gnuplot> plot sin(x) title ‘f(x)’ w lp lt 3 lw 2 pt 3 ps 2 %同时用点和线画，这里title ‘f(x)’表示图例上标’f(x)’，如果不用则用默认选项 
    gnuplot> plot sin(x) %此时所有选项均用默认值。如果缺某一项则将用默认值 
    gnuplot> plot ‘a.dat’ u 2:3 w l lt 3 lw 2 %利用数据文件a.dat中的第二和第三列作图 
    顺便提一下，如这里最前面的两个例子所示，在gnuplot中，如果某两个词，按字母先后顺序，前面某几个字母相同，后面的不同，那么只要写到第一个不同的字母就可以了。如with，由于没有其它以w开头的词，因此可以用 w 代替，line也可以用l 代替。 
## 2、同时画多条曲线 
    gnuplot> plot sin(x) title ‘sin(x)’ w l lt 1 lw 2, cos(x) title ‘cos(x)’ w l lt 2 lw 2 
    ％两条曲线是用逗号隔开的。画多条曲线时，各曲线间均用逗号隔开就可以了。 
    以上例子中是对函数作图，如果对数据文件作图，将函数名称换为数据文件名即可，但要用单引号引起来。 
## 3、关于图例的位置 
    默认位置在右上方。 
    gnuplot> set key left %放在左边，有left 和right两个选项 
    gnuplot> set key bottom %放在下边，只有这一个选项；默认在上边 
    gnuplot> set key outside %放在外边，但只能在右面的外边，以上三个选项可以进行组合。如： 
    gnuplot> set key left bottom %表示左下边，还可以直接用坐标精确表示图例的位置，如 
    gnuplot> set key 0.5,0.6 %将图例放在0.5,0.6的位置处 
## 4、关于坐标轴 
    gnuplot> set xlabel ‘x’ %x轴标为‘x’ 
    gnuplot> set ylabel ‘y’ %y轴标为’y’ 
    gnuplot> set ylabel ‘DOS’ tc lt 3 %其中的tc lt 3表示’DOS’的颜色用第三种颜色。 
    gnuplot> set xtics 1.0 %x轴的主刻度的宽度为1.0，同样可以为y轴定义ytics 
    gnuplot> set mxtics 3 %x轴上每个主刻度中画3个分刻度，同样可以为y轴定义mytics 
    gnuplot> set border 3 lt 3 lw 2 %设为第三种边界，颜色类型为3，线宽为2 
    同样可以为上边的x轴（称为x2）和右边y（称为y2）轴进行设置，即x2tics，mx2tics，y2tics，my2tics。 
    gnuplot> set xtics nomirror 
    gnuplot> unset x2tics %以上两条命令去掉上边x2轴的刻度 
    gnuplot> set ytics nomirror 
    gnuplot> unset y2tics %以上两条命令去掉右边y轴的刻度 
## 5、在图中插入文字 
    gnuplot> set label ‘sin(x)’ at 0.5,0.5 %在坐标（0.5,0.5）处加入字符串’sin(x)’。 
    在输出为.ps或.eps文件时，如果在set term 的语句中加入了enhanced选现，
    则可以插入上下标、希腊字母和特殊符号。
    上下标的插入和latex中的方法是一样的。 
## 6、在图中添加直线和箭头 
    gnuplot> set arrow from 0.0,0.0 to 0.6,0.8 %从（0.0,0.0）到（0.6,0.8）画一个箭头 
    gnuplot> set arrow from 0.0,0.0 to 0.6,0.8 lt 3 lw 2 %这个箭头颜色类型为3，线宽类型为2 
    gnuplot> set arrow from 0.0,0.0 to 0.6,0.8 nohead lt 3 lw 2 %利用nohead可以去掉箭头的头部，这就是添加直线的方法。 
    注意，在gnuplot中，对于插入多个的label和arrow等等，
    系统会默认按先后顺序分别对各个label或arrow进行编号，从1开始。
    如果以后要去掉某个label或arrow，那么只要用unset命令将相应的去掉即可。如： 
    gnuplot> unset arrow 2，将去掉第二个箭头。 
## 7、图的大小和位置 
    gnuplot>set size 0.5,0.5 %长宽均为默认宽度的一半，建议用这个取值，尤其是画成ps或eps图形的时候 
    gnuplot>set origin 0.0,0.5 %设定图的最左下角的那一点在图形面板中的位置。这里图将出现在左上角。 
## 8、画三维图 
    gnuplot>splot ‘文件名’ u 2:4:5 %以第二和第四列作为x和y坐标，第五列为z坐标。
# 二、提高篇： 
## 1、如何在同一张图里同时画多个图 
    gnuplot>set multiplot %设置为多图模式 
    gnuplot>set origin 0.0,0.0 %设置第一个图的原点的位置 
    gnuplot>set size 0.5,0.5 %设置第一个图的大小 
    gnuplot>plot “a1.dat” 
    gnuplot>set origin 0.0,0.5 %设置第二个图的原点的位置 
    gnuplot>set size 0.5,0.5 %设置第二个图的大小 
    gnuplot>plot “a2.dat” 
    gnuplot>set origin 0.0,0.0 %设置第三个图的原点的位置 
    gnuplot>set size 0.5,0.5 %设置第三个图的大小 
    gnuplot>plot “a3.dat” 
    gnuplot>set origin 0.0,0.0 %设置第四个图的原点的位置 
    gnuplot>set size 0.5,0.5 %设置第四个图的大小 
    gnuplot>plot “a4.dat” 
    当然，如果后一个图中的某个量的设置和前一个的相同，那么后一个中的这个量的设置可以省略。
    例如上面对第二、第三和第四个图的大小的设置。前一个图中对某个量的设置也会在后一个图中起作用。
    如果要取消在后面图中的作用，必须用如下命令，如取消label，用 
    gnuplot>unset label 
## 2、作二维图时，如何使两边坐标轴的单位长度等长 
    gnuplot> set size square %使图形是方的 
    gnuplot> set size 0.5,0.5 %使图形是你要的大小 
    gnuplot> set xrange[-a:a] 
    gnuplot> set yrange[-a:a] %两坐标轴刻度范围一样 
    gnuplot> plot ‘a.dat’ 
## 3、如何在同一张图里利用左右两边的y轴分别画图 
    gnuplot> set xtics nomirror %去掉上面坐标轴x2的刻度 
    gnuplot> set ytics nomirror %去掉右边坐标轴y2的刻度 
    gnuplot> set x2tics %让上面坐标轴x2刻度自动产生 
    gnuplot> set y2tics %让右边坐标轴y2的刻度自动产生 
    gnuplot> plot sin(x),cos(x) axes x1y2 %cos(x)用x1y2坐标，axes x1y2表示用x1y2坐标轴 
    gnuplot> plot sin(x),cos(x) axes x2y2 %cos(x)用x2y2坐标，axes x2y2表示用x2y2坐标轴 
    gnuplot> set x2range[-20:20] %设定x2坐标的范围 
    gnuplot> replot 
    gnuplot> set xrange[-5:5] %设定x坐标的范围 
    gnuplot> replot 
    gnuplot> set xlabel ‘x’ 
    gnuplot> set x2label ‘t’ 
    gnuplot> set ylabel ‘y’ 
    gnuplot> set y2label ’s’ 
    gnuplot> replot 
    gnuplot> set title ‘The figure’ 
    gnuplot> replot 
    gnuplot> set x2label ‘t’ textcolor lt 3 %textcolor lt 3或tc lt 3设置坐标轴名称的颜色 
## 4、如何插入希腊字母和特殊符号 
    一般只能在ps和eps图中，且必须指定enhanced选项。在X11终端（即显示器）中无法显示。 
    gnuplot> set terminal postscript enhanced 
    然后希腊字母就可以通过{/Symbol a}输入。例如 
    gnuplot> set label ‘{/Symbol a}’ 
    各种希腊字母与特殊符号的输入方法请见安装包中gnuplot-4.0.0/docs/psdoc目录下的ps_guide.ps文件。 
    另外还可参见： 
    http://t16web.lanl.gov/Kawano/gnuplot/label-e.html#4.3 
## 5、gnuplot中如何插入Angstrom（埃）这个符号(A上面一个小圆圈) 
    脚本中在插入前先加入gnuplot>set encoding iso_8859_1这个命令，
    然后就可以通过“{\305}”加入了。如横坐标要标上“k(1/?)”： 
    gnuplot>set xlabel ‘k(1/{\305}) 
    如果是multiplot模式，则这个命令必须放在gnuplot>set multiplot的前面。 
    如果后面还要插入别的转义字符，那么还要在插入字符后加入如下命令： 
    set encoding default 
    安装包中gnuplot-4.0.0/docs/psdoc/ps_guide.ps文件中的表中的‘E’代表那一列的所有符号都用这个方法输入。 
## 6、gnuplot画等高线图 
    gnuplot>splot ‘文件名.dat’ u 1:2:3 w l %做三维图 
    gnuplot>set dgrid3d 100,100 %设置三维图表面的网格的数目 
    gnuplot>replot 
    gnuplot>set contour %设置画等高线 
    gnuplot>set cntrparam levels incremental -0.2,0.01,0.2 %设置等高线的疏密和范围，
    数据从 -0.2到0.2中间每隔0.01画一条线 
    gnuplot>unset surface 去掉上面的三维图形，最后用鼠标拽动图形，选择合理的角度即可。
    或者直接设置(0,0)的视角也可以： 
    gnuplot>set view 0,0 
    gnuplot>replot 
    这里注意，画三维图的数据文件必须是分块的，也就是x每变换一个值，
    y在其变化范围内变化一周，这样作为一块，然后再取一个x值，y再变化一周，作为下一数据块，等等。块与块之间用一空行格开。 
## 7、输出为ps或eps图时，以下几个选项值得特别注意 
    gnuplot>set term postscript eps enh solid color 
    其中eps选项表示输出为eps格式，去掉则表示用默认的ps格式；
    enh选项表示图中可以插入上下标、希腊字母及其它特殊符号，
    如果去掉则不能插入；solid选项表示图中所有的曲线都用实线，去掉则将用不同的虚线；
    color选项表示在图中全部曲线用彩色，去掉则将用黑白。 
## 8、如何画漂亮的pm3d图 
    gnuplot> set pm3d %设置pm3d模式 
    gnuplot> set isosamples 50,50 %设置网格点 
    gnuplot> splot x**2+y**2 ％画三维图 
    gnuplot> splot x**2+y**2 w pm3d ％画成pm3d模式，注意比较变化 
    gnuplot> set view 0,0 ％设置视角，（0，0）将投影到底面上去 
    gnuplot> splot x**2+y**2 w pm3d ％重画，注意看变化 
    gnuplot> unset ztics %把z轴上的数字给去掉 
    gnuplot> set isosamples 200,200 ％使网格变细 
    gnuplot> replot ％重画，注意看变化，主要是过渡更光滑 
## 9、利用脚本文件避免重复输入 
    有时候对某个数据文件做好一张图后，下次可能还要利用这个数据文件作图，但某个或某些设置要作些细微变化。
    这时候，可以把第一次作图时的命令全部写到一个文件里，如a.plt，下次只要将相应的设置做修改后，
    用下面的命令就会自动运行文件所有的命令而最后得到你要的图： gnuplot>load ‘a.plt’ 
    作为一个例子，假设文件名为a.plt，里面的内容为： 
    set pm3d 
    set view 0,0 
    unset ztics 
    set isosamples 200,200 
    splot x**2+y**2 w pm3d 
    set term post color 
    set output ‘a.ps’ 
    replot 
    那么启动gnuplot后，只要运行如下命令就可以了： 
    gnuplot>load ‘a.plt’ 
    如果我们要得到的仅仅是.ps或.eps图，那也可以在linux命令提示符下直接运行如下命令： 
    [zxh@theory zxh]$gnuplot a.plt 
## 10、在gnuplot模式下运行linux命令 
    在gnuplot提示符下也可以运行linux命令，但必须在相应的命令前面加上 ! 号。
    例如，假设很多参量都已经设置好了，但需要对某个数据文件a.dat进行修改后再画图，则可以用如下方式 
    gnuplot>!vi a.dat 
    通过这种方式，所有的linux命令都可以在gnuplot环境里运行。 
    另外，也可以在gnuplot的提示符后输入shell，暂时性退出gnuplot，进入linux环境，
    做完要做的事情后，运行exit命令，又回到gnuplot环境下。 
    gnuplot>shell 
    [zxh@theory zxh]$vi a.f 
    [zxh@theory zxh]$f77 a.f 
    [zxh@theory zxh]$a.out (假设生成a.dat数据文件) 
    [zxh@theory zxh]$exit 
    gnuplot>plot ‘a.dat’ w l


    
