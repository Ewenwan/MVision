# Monocular slam 的理论基础
      单目SLAM一般处理流程包括track和map两部分。
      所谓的track是用来估计相机的位姿。
      而map部分就是计算pixel的深度，
      如果相机的位姿有了，就可以通过三角法(triangulation)确定pixel的深度，
      把这些计算好深度的pixel放到map里就重建出了三维环境。 
