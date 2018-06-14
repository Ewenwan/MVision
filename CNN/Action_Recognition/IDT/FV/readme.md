# 特征编码
[源码](https://github.com/Ewenwan/dtfv)

# 编译
## 下载vl_feat并编译  训练高斯混合模型
[下载](http://www.vlfeat.org/download.html)

[github源码](github.com/vlfeat/vlfeat)

    linux下直接 make编译
    编译好vl_feat工具箱后，将vlfeats/bin/glnx64中的libvl.so文件
    拷贝到dtfv/src/vl文件夹中，
    如果拷贝glnx86中的libvl.so后编译时会报错
    
## 编译 dtfv
    cd defv/src
    make 
    
## 运行dtfv
    1. 修改
       dtfv/script/extract_fv.py
       
```python       
#-*- coding:utf-8 -*-
# 提取idt特征
# 进行fv编码
# 之后考虑svm分类

#视频数据集列表文件　输出编码后的文件地址　每隔10个视频提取特征编码
# python2 extract_fv.py test_list_10.txt out_code_feature 10

# extract DT features on the fly and select feature points randomly
# need select_pts and dt binaries

# 错误解决办法
# error while loading shared libraries: libvl.so: 
# cannot open shared object file: No such file or directory 

# 链接库 环境变量
# LD_LIBRARY_PATH=$LD_LIBRARY_PATH:dtfv-master/src/vl 
# LD_LIBRARY_PATH=$LD_LIBRARY_PATH: dtfv-master/src/alglib 

# 可能还是找不到 直接把 libvl.so && libalg.so 拷贝到 /usr/local/lib  && /usr/lib

import subprocess, os, ffmpeg
import sys

# Dense Trajectories binary idt特征可执行文件
dtBin = '../../improved_trajectory_release/release/DenseTrackStab'
# Compiled fisher vector binary 计算fv编码的 可执行文件
fvBin = '../src/compute_fv'
# 零时文件夹 存放 分割的视频
tmpDir = '../tmp'
# Process ID for running in parallel
pID = 0

# 以下文件需要自己使用训练集训练
# ucf / med 数据库 
# PCA list 压缩　矩阵　coefficients系数  列表文件
# 
#pcaList = '../data/med.pca.lst'
pcaList = '../data/ucf.pca.lst'

'''
../data/ucf.traj.mat
../data/ucf.hog.mat
../data/ucf.hof.mat
../data/ucf.mbhx.mat
../data/ucf.mbhy.mat
'''

# GMM list 高斯混合模型 列表
#codeBookList = '../data/med.codebook.lst'
codeBookList = '../data/ucf.codebook.lst'

'''
../data/ucf.traj.gmm
../data/ucf.hog.gmm
../data/ucf.hof.gmm
../data/ucf.mbhx.gmm
../data/ucf.mbhy.gmm
'''

def extract(videoName, outputBase):
    videoName = '../../date/UCF101/UCF-101/'+ videoName
    if not os.path.exists(videoName):
        print '%s does not exist!' % videoName
        return False
    if check_dup(outputBase):
        print '%s processed' % videoName
        return True
    #resizedName = os.path.join(tmpDir, os.path.basename(videoName))
    # 变形 320*240
    #if not ffmpeg.resize(videoName, resizedName):
    resizedName = videoName     # resize failed, just use the input video
    #调用子程序计算特征并编码 
    subprocess.call('%s %s | %s %s %s %s' % (dtBin, resizedName, fvBin, pcaList, codeBookList, outputBase), shell=True)
    return True

def check_dup(outputBase):
    """
    Check if fv of all modalities have been extracted
    """
    featTypes = ['traj', 'hog', 'hof', 'mbhx', 'mbhy']
    featDims = [20, 48, 54, 48, 48]
    # 在DT/iDT算法中，选取L=15。轨迹特征为15*2=30维向量
    # HOG特征:96（2*2*3*8）
    # HOF特征:108（2*2*3*9）
    # MBHx/MBHy:96（2*2*3*8） 
    for i in range(len(featTypes)):
        featName = '%s.%s.fv.txt' % (outputBase, featTypes[i])
        if not os.path.isfile(featName) or not os.path.getsize(featName) > 0:
            return False
        # check if the length of feature can be fully divided by featDims
        f = open(featName)
        featLen = len(f.readline().rstrip().split())
        f.close()
        if featLen % (featDims[i] * 512) > 0:
            return False
    return True

if __name__ == '__main__':
    videoList = sys.argv[1]#视频文件
    outputBase = sys.argv[2]#输出
    totalTasks = int(sys.argv[3])#任务 int 间隔处理视频
    try:
        f = open(videoList, 'r')
        videos = f.readlines()#每一行是一个视频
        f.close()#
        videos = [video.rstrip() for video in videos]
        for i in range(0, len(videos)):
            if i % totalTasks == int(pID):
                print pID, videos[i]
                outputName = os.path.join(outputBase, os.path.basename(videos[i]))
                extract(videos[i], outputName)
    except IOError:
        sys.exit(0)
```
