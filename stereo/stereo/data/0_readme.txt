
棋盘格 大小2cm 20mm  200  0.1mm
8 格点  *  10格点

基线大致距离 16.7cm 实测

 ./Stereo_Calibr -w=8 -h=10 -s=200 stereo_calib.xml


./Stereo_Match -l=left01.jpg -r=right01.jpg --algorithm=bm -i=intrinsics.yml -e=extrinsics.yml

./Stereo_Match

