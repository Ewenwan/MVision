# 1. eigen c++矩阵运算库

[eigen c++矩阵运算库 代码实验](https://github.com/Ewenwan/Eigen_Test)

## linux下安装

      sudo apt-get install libeigen3-dev
      定位安装位置
      locate eigen3
      sudo updatedb


      * 当调用 Eigen库 成员 时，一下情况需要注意
           Eigen库中的数据结构作为自定义的结构体或者类中的成员;
           STL容器含有Eigen的数据结构
           Eigen数据结构作为函数的参数


           1:数据结构使用 Eigen库 成员
        class Foo
          {
            ...
            Eigen::Vector2d v;//
            ...
          public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW //不加  会提示 对其错误
          }

          2.STL Containers 标准容器vector<> 中使用 Eigen库 成员
          vector<Eigen::Matrix4d>;//会提示出错
          vector<Eigen::Matrix4d,Eigen::aligned_allocator<Eigen::Matrix4d>>;//aligned_allocator管理C++中的各种数据类型的内存方法是一样的,但是在Eigen中不一样

           3.函数参数 调用  Eigen库 成员
           FramedTransformation( int id, Eigen::Matrix4d t );//出错  error C2719: 't': formal parameter with __declspec(align('16')) won't be aligned
           FramedTransformation( int id, const Eigen::Matrix4d& t );// 把参数t的类型稍加变化即可


      头文件包含
      #include <Eigen/Dense>
      Eigen 矩阵定义
      Eigen中关于矩阵类的模板函数中，共有六个模板参数，常用的只有前三个。
      其前三个参数分别表示矩阵元素的类型、行数和列数。
      矩阵定义时可以使用Dynamic来表示矩阵的行列数为未知。

       矩阵类型：
       Eigen中的矩阵类型一般都是用类似MatrixXXX来表示，可以根据该名字来判断其数据类型，
       比如”d”表示double类型，
       ”f”表示float类型，
       ”i”表示整数，
       ”c”表示复数；
       Matrix2f，表示的是一个2*2维的，其每个元素都是float类型。

         数据存储：
         Matrix<double, 3, 3, RowMajor> E;       //  行排列(存储)的矩阵
       Matrix<int,3, 4, ColMajor> Acolmajor;   //   默认为列排列(存储)的矩阵  

        动态矩阵和静态矩阵：
               动态矩阵是指其大小在运行时确定，静态矩阵是指其大小在编译时确定。
               MatrixXd：表示任意大小的元素类型为double的矩阵变量，其大小只有在运行时被赋值之后才能知道。
               Matrix3d：表示元素类型为double大小为3*3的矩阵变量，其大小在编译时就知道。
               在Eigen中行优先的矩阵会在其名字中包含有row，否则就是列优先。
               Eigen中的向量只是一个特殊的矩阵，其维度为1而已。

        矩阵元素的访问：
            在矩阵的访问中，行索引总是作为第一个参数，Eigen中矩阵、数组、向量的下标都是从0开始。
            矩阵元素的访问可以通过”()”操作符完成。
            例如m(2, 3)既是获取矩阵m的第2行第3列元素。
            针对向量还提供”[]”操作符，注意矩阵则不可如此使用。

         设置矩阵的元素：
         在Eigen中重载了”<<”操作符，
         通过该操作符即可以一个一个元素的进行赋值，
         也可以一块一块的赋值。
         另外也可以使用下标进行赋值。
         A << 1, 2, 3,  // 按列填充A
           4, 5, 6,         //  
           7, 8, 9;         //  
        B << A, A, A;   //  块赋值
        A.fill(10);        //  函数赋值

        重置矩阵大小：
        当前矩阵的行数、列数、大小可以通过rows()、cols()和size()来获取，
        对于动态矩阵可以通过resize()函数来动态修改矩阵的大小。
        注意：(1)、固定大小的矩阵是不能使用resize()来修改矩阵的大小；
        (2)、resize()函数会析构掉原来的数据，因此调用resize()函数之后将不能保证元素的值不改变；
        (3)、使用”=”操作符操作动态矩阵时，如果左右两边的矩阵大小不等，
               则左边的动态矩阵的大小会被修改为右边的大小。

        如何选择动态矩阵和静态矩阵：对于小矩阵(一般大小小于16)使用固定大小的静态矩阵，它可以带来比较高的效率；
        对于大矩阵(一般大小大于32)建议使用动态矩阵。
        注意：如果特别大的矩阵使用了固定大小的静态矩阵则可能会造成栈溢出的问题。





      Matrix<double, 3, 3> A;                           // 固定行和列的矩阵 Fixed   Matrix3d.
      Matrix<double, 3, Dynamic> B;              // 固定行，列数不定的矩阵  Matrix<double, Dynamic，3> BB;  
      Matrix<double, Dynamic, Dynamic> C; // 行列数不定的矩阵              MatrixXd.
      Matrix<double, 3, 3, RowMajor> E;       //  行排列(存储)的矩阵， ，，默认为列排列(存储)的矩阵  Matrix<int,3, 4, ColMajor> Acolmajor;
      Matrix3f P, Q, R;                                      // 3x3 float matrix. 浮点型 3x3矩阵
      Vector3f x, y, z;                                        // 3x1 float matrix. 浮点型 3x1向量
      RowVector3f a, b, c;                  // 行向量 1x3 float matrix.
      VectorXd v;                               //  动态 doubles 类型向量


       Eigen 基础使用
      // Eigen          // Matlab           // comments
      x.size()          // length(x)        // vector size
      C.rows()          // size(C,1)        // 行数量
      C.cols()          // size(C,2)         // 列数量
      x(i)              // x(i+1)                //按列排列顺序(或行排列顺序)取元素  Matlab is 1-based
      C(i,j)            // C(i+1,j+1)         //对应行和列的元素


      A.resize(4, 4);   // Runtime error if assertions are on.
      B.resize(4, 9);   // Runtime error if assertions are on.
      A.resize(3, 3);   // Ok; size didn't change.
      B.resize(3, 9);   // Ok; only dynamic cols changed.

      A << 1, 2, 3,     // Initialize A. The elements can also be
           4, 5, 6,     // matrices, which are stacked along cols
           7, 8, 9;     // and then the rows are stacked.
      B << A, A, A;     // B is three horizontally stacked A's.
      A.fill(10);       // Fill A with all 10's.

      Eigen 特殊矩阵生成


      // Eigen                            // Matlab
      MatrixXd::Identity(rows,cols)       // eye(rows,cols)
      C.setIdentity(rows,cols)            // C = eye(rows,cols)
      MatrixXd::Zero(rows,cols)           // zeros(rows,cols)
      C.setZero(rows,cols)                // C = ones(rows,cols)
      MatrixXd::Ones(rows,cols)           // ones(rows,cols)
      C.setOnes(rows,cols)                // C = ones(rows,cols)
      MatrixXd::Random(rows,cols)         // rand(rows,cols)*2-1        // MatrixXd::Random returns uniform random numbers in (-1, 1).
      C.setRandom(rows,cols)              // C = rand(rows,cols)*2-1
      VectorXd::LinSpaced(size,low,high)  // linspace(low,high,size)'
      v.setLinSpaced(size,low,high)       // v = linspace(low,high,size)'

      矩阵的块操作：有两种使用方法：

               matrix.block(i,j, p, q) : 表示返回从矩阵(i, j)开始，每行取p个元素，每列取q个元素所组成的临时新矩阵对象，原矩阵的元素不变；
               matrix.block<p,q>(i, j) :<p, q>可理解为一个p行q列的子矩阵，该定义表示从原矩阵中第(i, j)开始，
               获取一个p行q列的子矩阵，返回该子矩阵组成的临时矩阵对象，原矩阵的元素不变；

        向量的块操作：

               获取向量的前n个元素：vector.head(n);
               获取向量尾部的n个元素：vector.tail(n);
               获取从向量的第i个元素开始的n个元素：vector.segment(i,n);
               Map类：在已经存在的矩阵或向量中，不必拷贝对象，而是直接在该对象的内存上进行运算操作。

      Eigen 矩阵分块

      // vector is x(1)...x(N)).
      // Eigen                           // Matlab
      x.head(n)                          // x(1:n)     向量快操作
      x.head<n>()                        // x(1:n)
      x.tail(n)                          // x(end - n + 1: end)
      x.tail<n>()                        // x(end - n + 1: end)
      x.segment(i, n)                    // x(i+1 : i+n)
      x.segment<n>(i)                    // x(i+1 : i+n)
      P.block(i, j, rows, cols)          // P(i+1 : i+rows, j+1 : j+cols)    矩阵块操作
      P.block<rows, cols>(i, j)          // P(i+1 : i+rows, j+1 : j+cols)
      P.row(i)                           // P(i+1, :)
      P.col(j)                           // P(:, j+1)
      P.leftCols<cols>()                 // P(:, 1:cols)
      P.leftCols(cols)                   // P(:, 1:cols)
      P.middleCols<cols>(j)              // P(:, j+1:j+cols)
      P.middleCols(j, cols)              // P(:, j+1:j+cols)
      P.rightCols<cols>()                // P(:, end-cols+1:end)
      P.rightCols(cols)                  // P(:, end-cols+1:end)
      P.topRows<rows>()                  // P(1:rows, :)
      P.topRows(rows)                    // P(1:rows, :)
      P.middleRows<rows>(i)              // P(i+1:i+rows, :)
      P.middleRows(i, rows)              // P(i+1:i+rows, :)
      P.bottomRows<rows>()               // P(end-rows+1:end, :)
      P.bottomRows(rows)                 // P(end-rows+1:end, :)
      P.topLeftCorner(rows, cols)        // P(1:rows, 1:cols)
      P.topRightCorner(rows, cols)       // P(1:rows, end-cols+1:end)
      P.bottomLeftCorner(rows, cols)     // P(end-rows+1:end, 1:cols)
      P.bottomRightCorner(rows, cols)    // P(end-rows+1:end, end-cols+1:end)
      P.topLeftCorner<rows,cols>()       // P(1:rows, 1:cols)
      P.topRightCorner<rows,cols>()      // P(1:rows, end-cols+1:end)
      P.bottomLeftCorner<rows,cols>()    // P(end-rows+1:end, 1:cols)
      P.bottomRightCorner<rows,cols>()   // P(end-rows+1:end, end-cols+1:end)

      复制代码
      // Of particular note is Eigen's swap function which is highly optimized.
      // Eigen                           // Matlab
      R.row(i) = P.col(j);               // R(i, :) = P(:, i)
      R.col(j1).swap(mat1.col(j2));      // R(:, [j1 j2]) = R(:, [j2, j1])


      求矩阵的转置、共轭矩阵、伴随矩阵：可以通过成员函数transpose()、conjugate()、adjoint()来完成。
      注意：这些函数返回操作后的结果，而不会对原矩阵的元素进行直接操作，
      如果要让原矩阵进行转换，则需要使用响应的InPlace函数，
      如transpoceInPlace()等；

      Eigen 矩阵转置
      // Views, transpose, etc; all read-write except for .adjoint().
      // Eigen                           // Matlab
      R.adjoint()                        // R'
      R.transpose()                      // R.' or conj(R')
      R.diagonal()                       // diag(R)
      x.asDiagonal()                     // diag(x)
      R.transpose().colwise().reverse(); // rot90(R)
      R.conjugate()                      // conj(R)


      矩阵和向量的算术运算：在Eigen中算术运算重载了C++的+、-、*

               (1)、矩阵的运算：
               提供+、-、一元操作符
               ”-”、+=、-=；二元操作符
               +/-，
               表示两矩阵相加(矩阵中对应元素相加/减，返回一个临时矩阵)；
               一元操作符-表示对矩阵取负(矩阵中对应元素取负，返回一个临时矩阵)；
               组合操作法+=或者-=表示(对应每个元素都做相应操作)；
               矩阵还提供与标量(单一数字)的乘除操作，表示每个元素都与该标量进行乘除操作；

      Eigen 矩阵乘积
      // All the same as Matlab, but matlab doesn't have *= style operators.
      // Matrix-vector.  Matrix-matrix.   Matrix-scalar.
      y  = M*x;          R  = P*Q;        R  = P*s;
      a  = b*M;          R  = P - Q;      R  = s*P;
      a *= M;            R  = P + Q;      R  = P/s;
                         R *= Q;          R  = s*P;
                         R += Q;          R *= s;
                         R -= Q;          R /= s;

      Eigen 矩阵单个元素操作

      // Vectorized operations on each element independently
      // Eigen                            // Matlab
      R = P.cwiseProduct(Q);    // R = P .* Q
      R = P.array() * s.array();  // R = P .* s
      R = P.cwiseQuotient(Q);   // R = P ./ Q
      R = P.array() / Q.array();  // R = P ./ Q
      R = P.array() + s.array();  // R = P + s
      R = P.array() - s.array();   // R = P - s
      R.array() += s;           // R = R + s
      R.array() -= s;           // R = R - s
      R.array() < Q.array();    // R < Q
      R.array() <= Q.array();   // R <= Q
      R.cwiseInverse();         // 1 ./ P
      R.array().inverse();      // 1 ./ P
      R.array().sin()           // sin(P)
      R.array().cos()           // cos(P)
      R.array().pow(s)          // P .^ s
      R.array().square()        // P .^ 2
      R.array().cube()          // P .^ 3
      R.cwiseSqrt()             // sqrt(P)
      R.array().sqrt()          // sqrt(P)
      R.array().exp()           // exp(P)
      R.array().log()           // log(P)
      R.cwiseMax(P)             // max(R, P)
      R.array().max(P.array())  // max(R, P)
      R.cwiseMin(P)             // min(R, P)
      R.array().min(P.array())  // min(R, P)
      R.cwiseAbs()              // abs(P)
      R.array().abs()           // abs(P)
      R.cwiseAbs2()             // abs(P.^2)
      R.array().abs2()          // abs(P.^2)
      (R.array() < s).select(P,Q);  // (R < s ? P : Q)

      复制代码
      Eigen 矩阵化简
      复制代码

      // Reductions.
      int r, c;
      // Eigen                  // Matlab
      R.minCoeff()              // min(R(:))
      R.maxCoeff()              // max(R(:))
      s = R.minCoeff(&r, &c)    // [s, i] = min(R(:)); [r, c] = ind2sub(size(R), i);
      s = R.maxCoeff(&r, &c)    // [s, i] = max(R(:)); [r, c] = ind2sub(size(R), i);
      R.sum()                   // sum(R(:))
      R.colwise().sum()         // sum(R)
      R.rowwise().sum()         // sum(R, 2) or sum(R')'
      R.prod()                  // prod(R(:))
      R.colwise().prod()        // prod(R)
      R.rowwise().prod()        // prod(R, 2) or prod(R')'
      R.trace()                 // trace(R)
      R.all()                   // all(R(:))
      R.colwise().all()         // all(R)
      R.rowwise().all()         // all(R, 2)
      R.any()                   // any(R(:))
      R.colwise().any()         // any(R)
      R.rowwise().any()         // any(R, 2)

      复制代码
      Eigen 矩阵点乘

      // Dot products, norms, etc.
      // Eigen                  // Matlab
      x.norm()                  // norm(x).    Note that norm(R) doesn't work in Eigen.
      x.squaredNorm()           // dot(x, x)   Note the equivalence is not true for complex
      x.dot(y)                  // dot(x, y)
      x.cross(y)                // cross(x, y) Requires #include <Eigen/Geometry>

      Eigen 矩阵类型转换
      复制代码

      //// Type conversion
      // Eigen                           // Matlab
      A.cast<double>();                  // double(A)
      A.cast<float>();                   // single(A)
      A.cast<int>();                     // int32(A)
      A.real();                          // real(A)
      A.imag();                          // imag(A)
      // if the original type equals destination type, no work is done

      复制代码
      Eigen 求解线性方程组 Ax = b
      复制代码

      // Solve Ax = b. Result stored in x. Matlab: x = A \ b.
      x = A.ldlt().solve(b));  // A sym. p.s.d.    #include <Eigen/Cholesky>
      x = A.llt() .solve(b));  // A sym. p.d.      #include <Eigen/Cholesky>
      x = A.lu()  .solve(b));  // Stable and fast. #include <Eigen/LU>
      x = A.qr()  .solve(b));  // No pivoting.     #include <Eigen/QR>
      x = A.svd() .solve(b));  // Stable, slowest. #include <Eigen/SVD>
      // .ldlt() -> .matrixL() and .matrixD()
      // .llt()  -> .matrixL()
      // .lu()   -> .matrixL() and .matrixU()
      // .qr()   -> .matrixQ() and .matrixR()
      // .svd()  -> .matrixU(), .singularValues(), and .matrixV()

      复制代码
      Eigen 矩阵特征值
      复制代码

      // Eigenvalue problems
      // Eigen                          // Matlab
      A.eigenvalues();                  // eig(A);
      EigenSolver<Matrix3d> eig(A);     // [vec val] = eig(A)
      eig.eigenvalues();                // diag(val)
      eig.eigenvectors();               // vec
      // For self-adjoint matrices use SelfAdjointEigenSolver<>
# 2. Pangolin 3D视觉和3D导航可视化

## 库介绍
      * Pangolin是一个用于OpenGL显示/交互以及视频输入的一个轻量级、快速开发库
      * Pangolin是对OpenGL进行封装的轻量级的OpenGL输入/输出和视频显示的库。
      * 可以用于3D视觉和3D导航的视觉图，可以输入各种类型的视频、并且可以保留视频和输入数据用于debug。
      * 下载工具Pangolin    github: https://github.com/stevenlovegrove/Pangolin

## 安装 pangolin 需要的依赖库 
      OpenGL (Desktop / ES / ES2)依赖

      Glew 依赖
      sudo apt-get install libglew-dev     
      CMake 依赖   编译需要
      sudo apt-get install cmake
      Boost 依赖    多线程  文件系统  是C++标准化进程的开发引擎之一
      sudo apt-get install libboost-dev libboost-thread-dev libboost-filesystem-dev
      python 依赖
      sudo apt-get install libpython2.7-dev



##  编译 安装l pangolin
      cd [path-to-pangolin]
      mkdir build
      cd build
      cmake ..
      make 
      sudo make install 

## 编译此项目
      * compile this program:
      mkdir build
      cd build
      cmake ..
      make 

      * run the build/visualizeGeometry

      2. How to use this program:

      The UI in the left panel displays different representations of T_w_c ( camera to world ). 
      显示 旋转矩阵 平移向量 欧拉角 四元素 
      Drag your left mouse button to move the camera, 左键 移动相机
      right button to rotate it around the box,                  右键 以箱子为中心旋转相机
      center button to rotate the camera itself,                 中键 旋转相机本身
      and press both left and right button to roll the view. 
      Note that in this program the original X axis is right (red line), Y is up (green line) and Z in back axis (blue line). You (camera) are looking at (0,0,0) standing on (3,3,3) at first. 

      3. Problems may happen:
      * I found that in virtual machines there may be an error in pangolin, which was solved in its issue: https://github.com/stevenlovegrove/Pangolin/issues/74 . You need to comment the two lines mentioned by paulinus, and the recompile and reinstall Pangolin, if you happen to find this problem. 

      If you still have problems using this program, please contact: gaoxiang12@mails.tsinghua.edu.cn
