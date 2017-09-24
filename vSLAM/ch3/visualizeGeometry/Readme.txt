1. How to compile this program:

* use pangolin: slambook/3rdpart/Pangolin or download it from github: https://github.com/stevenlovegrove/Pangolin

* install dependency for pangolin (mainly the OpenGL): 
sudo apt-get install libglew-dev

* compile and install pangolin
cd [path-to-pangolin]
mkdir build
cd build
cmake ..
make 
sudo make install 

* compile this program:
mkdir build
cd build
cmake ..
make 

* run the build/visualizeGeometry

2. How to use this program:

The UI in the left panel displays different representations of T_w_c ( camera to world ). It shows the rotation matrix, tranlsation vector, euler angles (in roll-pitch-yaw order) and the quaternion.
Drag your left mouse button to move the camera, right button to rotate it around the box, center button to rotate the camera itself, and press both left and right button to roll the view. 
Note that in this program the original X axis is right (red line), Y is up (green line) and Z in back axis (blue line). You (camera) are looking at (0,0,0) standing on (3,3,3) at first. 

3. Problems may happen:
* I found that in virtual machines there may be an error in pangolin, which was solved in its issue: https://github.com/stevenlovegrove/Pangolin/issues/74 . You need to comment the two lines mentioned by paulinus, and the recompile and reinstall Pangolin, if you happen to find this problem. 

If you still have problems using this program, please contact: gaoxiang12@mails.tsinghua.edu.cn
