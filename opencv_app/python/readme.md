# python opencv  接口

[参考](https://github.com/PacktPublishing/Mastering-OpenCV-4-with-Python)

## 1. 简单图形载入 灰度化 显示
```python
# 导入包
import cv2
# 默认方式读取图片
image = cv2.imread("logo.png")
# gray_img = cv2.imread('logo.png', cv2.IMREAD_GRAYSCALE)

dimensions = img.shape
# 矩阵维度   1/3
print(dimensions)
if len(dimensions) < 3:
    print("grayscale image!")
if len(dimensions) == 3:
    print("color image!")


(h, w, c) = img.shape
print("Dimensions of the image - Height: {}, Width: {}, Channels: {}".format(h, w, c))

# 总像素数量
total_number_of_pixels = img.size
print("Total number of elements: {}".format(total_number_of_pixels))
print("Total number of elements: {}".format(h * w * c))

# 图像数据类型
image_dtype = img.dtype
print("Image datatype: {}".format(image_dtype))
# This should print 'Image datatype: uint8'
# (uint8) = unsigned char

# 获取指定像素位置值
# Get the value of the pixel (x=40, y=6):
(b, g, r) = img[6, 40]
print("Pixel at (6,40) - Red: {}, Green: {}, Blue: {}".format(r, g, b))

b = img[6, 40, 0]
g = img[6, 40, 1]
r = img[6, 40, 2]

# 修改像素值
img[6, 40] = (0, 0, 255)

# 获取指定ROI区域图像
top_left_corner = img[0:50, 0:50]
cv2.imshow("top left corner original", top_left_corner)
cv2.waitKey(0)

# 修改roi图像
img[20:70, 20:70] = top_left_corner
img[0:50, 0:50] = (255, 0, 0) #blue bgr

# 彩色图灰度化 (BGR to GRAY):
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 单一通道
gray_height, gray_width = gray_image.shape
# 像素数量
total_number_of_pixels = gray_img.size
# 修改/获取 指定 位置/roi 像素值

# 自适应窗口显示图像
cv2.imshow("OpenCV logo", image)
b, g, r = cv2.split(image)            # bgr
img_matplotlib = cv2.merge([r, g, b]) # rgb

# 矩阵连接 图像连接  image在左边 img_matplotlib 在右边
img_concats = np.concatenate((image, img_matplotlib), axis=1) # 按列 闫拓

# 功能同 cv2.split(image)
B = image[:, :, 0]
G = image[:, :, 1]
R = image[:, :, 2]

# 第三通道 逆序 bgr ----> rgb  Numpy 
img_RGB = img_OpenCV[:, :, ::-1]

# 显示灰度图
cv2.imshow("OpenCV logo gray format", gray_image)

# 保存文件
cv2.imwrite("./gray", gray_image)


# difference = cv2.subtract(bgr_image, temp)# 矩阵相减 cv2.add(image, scalar) 相加
# b, g, r = cv2.split(difference)           # 矩阵通道分割  merge() 合并
# img_matplotlib = cv2.merge([r, g, b]) # brg 图像 变成 RGB 图像 !!!!!!!!
# assert cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0


# 等待键盘按键
cv2.waitKey(0)

# 销毁显示窗口
cv2.destroyAllWindows()
```


## 2. 命令行参数解析 摄像头读取 录制视频
```python
import argparse
import cv2

# 时间 ====
import datetime
import time

# 解析命令行参数
parser = argparse.ArgumentParser()

# 添加key
parser.add_argument("path_image", help="path to input image to be displayed")# 添加 key
parser.add_argument("path_image_output", help="path of the processed image to be saved")
parser.add_argument("index_camera", help="index of the camera to read from", type=int)# 摄像头id
parser.add_argument("ip_url", help="IP URL to connect") # 网络摄像头
parser.add_argument("video_path", help="path to the video file") # 视频文件
# https://github.com/PacktPublishing/Mastering-OpenCV-4-with-Python/blob/master/Chapter03/01-chapter-content/read_video_file_all_properties.py


args = parser.parse_args()# 解析
image_input = cv2.imread(args.path_image)
# 解析参数成字典 类型
arg_dict = vars(parser.parse_args())
image2 = cv2.imread(arg_dict["path_image"])

gray_image = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
# 保存
cv2.imwrite(arg_dict["path_image_output"], gray_image)


# 打开摄像头
capture = cv2.VideoCapture(arg_dict["index_camera"])
# capture = cv2.VideoCapture(arg_dict["ip_url"]) # 打开网络摄像头
# https://github.com/PacktPublishing/Mastering-OpenCV-4-with-Python/blob/master/Chapter03/01-chapter-content/write_video_file.py # 写视频文件 录制视频

# capture = cv2.VideoCapture(arg_dict["video_path"]) # 打开 视频文件
# https://github.com/PacktPublishing/Mastering-OpenCV-4-with-Python/blob/master/Chapter03/01-chapter-content/read_video_file_backwards.py # 反向播放视频


# 获取参数
frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = capture.get(cv2.CAP_PROP_FPS) # 帧率

# 检测摄像头是否打开
if capture.isOpened()is False:
    print("Error opening the camera")


# 读取摄像头
while capture.isOpened():
    # Capture frame-by-frame from the camera
    ret, frame = capture.read()

    if ret is True:
        # 开始时间
        processing_start = time.time()
        # 显示原图
        cv2.imshow('Input frame from the camera', frame)

        # 转换成灰度图
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 显示 灰度图
        cv2.imshow('Grayscale input camera', gray_frame)
        
        # c 键 保存图像
        if cv2.waitKey(20) & 0xFF == ord('c'):
            frame_name = "camera_frame_{}.png".format(frame_index)
            gray_frame_name = "grayscale_camera_frame_{}.png".format(frame_index)
            cv2.imwrite(frame_name, frame)
            cv2.imwrite(gray_frame_name, gray_frame)
            frame_index += 1
            
        # 按键检测 关闭 q键
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
        
        # 结束时间
        processing_end = time.time()
        processing_time_frame = processing_end - processing_start  #时间差
        print("fps: {}".format(1.0 / processing_time_frame))# 帧率
        
    # Break the loop
    else:
        break
 
# Release everything:
capture.release()

```


## 3. 绘制图形 GUI 鼠标响应
https://github.com/PacktPublishing/Mastering-OpenCV-4-with-Python/tree/master/Chapter04/01-chapter-content

```python
# 显示数字时钟
"""
Example to show how to draw an analog clock OpenCV
"""

# Import required packages:
import cv2
import numpy as np
import datetime
import math


def array_to_tuple(arr):
    return tuple(arr.reshape(1, -1)[0])


# Dictionary containing some colors
colors = {'blue': (255, 0, 0), 'green': (0, 255, 0), 'red': (0, 0, 255), 'yellow': (0, 255, 255),
          'magenta': (255, 0, 255), 'cyan': (255, 255, 0), 'white': (255, 255, 255), 'black': (0, 0, 0),
          'gray': (125, 125, 125), 'rand': np.random.randint(0, high=256, size=(3,)).tolist(),
          'dark_gray': (50, 50, 50), 'light_gray': (220, 220, 220)}

# We create the canvas to draw: 640 x 640 pixels, 3 channels, uint8 (8-bit unsigned integers)
# We set background to black using np.zeros()
image = np.zeros((640, 640, 3), dtype="uint8")

# If you want another background color you can do the following:
image[:] = colors['light_gray']

# Coordinates to define the origin for the hour markings:
hours_orig = np.array(
    [(620, 320), (580, 470), (470, 580), (320, 620), (170, 580), (60, 470), (20, 320), (60, 170), (169, 61), (319, 20),
     (469, 60), (579, 169)])

# Coordinates to define the destiny for the hour markings:
hours_dest = np.array(
    [(600, 320), (563, 460), (460, 562), (320, 600), (180, 563), (78, 460), (40, 320), (77, 180), (179, 78), (319, 40),
     (459, 77), (562, 179)])

# We draw the hour markings:
for i in range(0, 12):
    cv2.line(image, array_to_tuple(hours_orig[i]), array_to_tuple(hours_dest[i]), colors['black'], 3)

# We draw a big circle, corresponding to the shape of the analog clock
cv2.circle(image, (320, 320), 310, colors['dark_gray'], 8)

# We draw the rectangle containig the text and the text "Mastering OpenCV 4 with Python":
cv2.rectangle(image, (150, 175), (490, 270), colors['dark_gray'], -1)
cv2.putText(image, "Mastering OpenCV 4", (150, 200), 1, 2, colors['light_gray'], 1, cv2.LINE_AA)
cv2.putText(image, "with Python", (210, 250), 1, 2, colors['light_gray'], 1, cv2.LINE_AA)

# We make a copy of the image with the "static" information
image_original = image.copy()

# Now, we draw the "dynamic" information:
while True:
    # Get current date:
    date_time_now = datetime.datetime.now()
    # Get current time from the date:
    time_now = date_time_now.time()
    # Get current hour-minute-second from the time:
    hour = math.fmod(time_now.hour, 12)
    minute = time_now.minute
    second = time_now.second

    print("hour:'{}' minute:'{}' second: '{}'".format(hour, minute, second))

    # Get the hour, minute and second angles:
    second_angle = math.fmod(second * 6 + 270, 360)
    minute_angle = math.fmod(minute * 6 + 270, 360)
    hour_angle = math.fmod((hour * 30) + (minute / 2) + 270, 360)

    print("hour_angle:'{}' minute_angle:'{}' second_angle: '{}'".format(hour_angle, minute_angle, second_angle))

    # Draw the lines corresponding to the hour, minute and second needles
    second_x = round(320 + 310 * math.cos(second_angle * 3.14 / 180))
    second_y = round(320 + 310 * math.sin(second_angle * 3.14 / 180))
    cv2.line(image, (320, 320), (second_x, second_y), colors['blue'], 2)

    minute_x = round(320 + 260 * math.cos(minute_angle * 3.14 / 180))
    minute_y = round(320 + 260 * math.sin(minute_angle * 3.14 / 180))
    cv2.line(image, (320, 320), (minute_x, minute_y), colors['blue'], 8)

    hour_x = round(320 + 220 * math.cos(hour_angle * 3.14 / 180))
    hour_y = round(320 + 220 * math.sin(hour_angle * 3.14 / 180))
    cv2.line(image, (320, 320), (hour_x, hour_y), colors['blue'], 10)

    # Finally, a small circle, corresponding to the point where the three needles joint, is drawn:
    cv2.circle(image, (320, 320), 10, colors['dark_gray'], -1)

    # Show image:
    cv2.imshow("clock", image)

    # We get the image with the static information:
    image = image_original.copy()

    # A wait of 500 milliseconds is performed (to see the displayed image):
    cv2.waitKey(500)

```



## 4. 图像处理 滤波 平滑 位操作

```python
image = cv2.imread('lenna.png')
# 高斯核滤波
image_filtered = cv2.GaussianBlur(image, (3, 3), 0)
# 灰度图
gray_image = cv2.cvtColor(image_filtered, cv2.COLOR_BGR2GRAY)
# 水平梯度 输出精度 16位 signed integers  
gradient_x = cv2.Sobel(gray_image, cv2.CV_16S, 1, 0, 3)
# 垂直梯度
gradient_y = cv2.Sobel(gray_image, cv2.CV_16S, 0, 1, 3)

# 转换到 无符号，取绝对值
abs_gradient_x = cv2.convertScaleAbs(gradient_x)
abs_gradient_y = cv2.convertScaleAbs(gradient_y)
# 叠加
sobel_image = cv2.addWeighted(abs_gradient_x, 0.5, abs_gradient_y, 0.5, 0)


# 中值滤波=====
img_gray = cv2.medianBlur(img_gray, 5)
# 拉普拉斯变换 检测边缘====
edges = cv2.Laplacian(img_gray, cv2.CV_8U, ksize=5)
# 阈值二值化====
ret, thresholded = cv2.threshold(edges, 70, 255, cv2.THRESH_BINARY_INV)

# 双边滤波
filtered = cv2.bilateralFilter(img, 10, 250, 250)
# https://github.com/PacktPublishing/Mastering-OpenCV-4-with-Python/blob/master/Chapter05/01-chapter-content/cartoonizing.py # 图片卡通 化
# 各种颜色空间
# https://github.com/PacktPublishing/Mastering-OpenCV-4-with-Python/blob/master/Chapter05/01-chapter-content/color_map_all.py

# 位操作======================掩码=========
# Create the first image:
img_1 = np.zeros((300, 300), dtype="uint8")
cv2.rectangle(img_1, (10, 10), (110, 110), (255, 255, 255), -1)
cv2.circle(img_1, (200, 200), 50, (255, 255, 255), -1)

# Create the second image:
img_2 = np.zeros((300, 300), dtype="uint8")
cv2.rectangle(img_2, (50, 50), (150, 150), (255, 255, 255), -1)
cv2.circle(img_2, (225, 200), 50, (255, 255, 255), -1)

# Bitwise OR  位或
bitwise_or = cv2.bitwise_or(img_1, img_2)

# Bitwise AND 位与
bitwise_and = cv2.bitwise_and(img_1, img_2)

# Bitwise XOR 位异或
bitwise_xor = cv2.bitwise_xor(img_1, img_2)

# Bitwise NOT 位 非操作
bitwise_not_1 = cv2.bitwise_not(img_1)

# Bitwise NOT
bitwise_not_2 = cv2.bitwise_not(img_2)

# 腐蚀膨胀 开闭 区域操作
https://github.com/PacktPublishing/Mastering-OpenCV-4-with-Python/blob/master/Chapter05/01-chapter-content/morphological_operations.py

```



## 5.  颜色直方图

```python
histr = []
histr.append(cv2.calcHist([img], [0], None, [256], [0, 256]))
histr.append(cv2.calcHist([img], [1], None, [256], [0, 256]))
histr.append(cv2.calcHist([img], [2], None, [256], [0, 256]))
# https://github.com/PacktPublishing/Mastering-OpenCV-4-with-Python/tree/master/Chapter06/01-chapter-content


# 直方图模板匹配 分类器
https://github.com/PacktPublishing/OpenCV-4-for-Secret-Agents-Second-Edition/blob/master/Chapter002/HistogramClassifier.py

```



## 6.阈值处理

```python
# 自适应阈值
thresh1 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
thresh2 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 3)
thresh3 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
thresh4 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 3)
# https://github.com/PacktPublishing/Mastering-OpenCV-4-with-Python/tree/master/Chapter07/01-chapter-content
```



## 7.边缘轮廓区域

```python
#  https://github.com/PacktPublishing/Mastering-OpenCV-4-with-Python/tree/master/Chapter08/01-chapter-content

```



## 8. 相机 aruco 校正
```python
# 相机 aruco 码 校正
https://github.com/PacktPublishing/Mastering-OpenCV-4-with-Python/blob/master/Chapter09/01-chapter-content/aruco_camera_calibration.py

# aruco 码识别 增强现实
# https://github.com/PacktPublishing/Mastering-OpenCV-4-with-Python/blob/master/Chapter09/01-chapter-content/aruco_detect_markers_augmented_reality.py
# https://blog.csdn.net/ZJU_fish1996/article/details/72312514?fps=1&locationNum=7
```


## 9. 特征检测

```python
# orb 特征点 =====================
# Load test image:
image = cv2.imread('opencv_logo_with_text.png')

# Initiate ORB detector:
orb = cv2.ORB_create()

# Detect the keypoints using ORB:
keypoints = orb.detect(image, None)

# Compute the descriptors of the detected keypoints:
keypoints, descriptors = orb.compute(image, keypoints)

# Print one ORB descriptor:
print("First extracted descriptor: {}".format(descriptors[0]))

# Draw detected keypoints:
image_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(255, 0, 255), flags=0)

# 检测 + 描述子
# https://github.com/PacktPublishing/Mastering-OpenCV-4-with-Python/blob/master/Chapter09/01-chapter-content/feature_matching.py

# Detect the keypoints and compute the descriptors with ORB:
keypoints_1, descriptors_1 = orb.detectAndCompute(image_query, None)
keypoints_2, descriptors_2 = orb.detectAndCompute(image_scene, None)
# 特征匹配
bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
bf_matches = bf_matcher.match(descriptors_1, descriptors_2)
# Sort the matches in the order of their distance:
bf_matches = sorted(bf_matches, key=lambda x: x.distance)

# 特征检测模板匹配物体识别
# https://github.com/PacktPublishing/Mastering-OpenCV-4-with-Python/blob/master/Chapter09/01-chapter-content/feature_matching_object_recognition.py

# 二维码扫描
https://github.com/PacktPublishing/Mastering-OpenCV-4-with-Python/blob/master/Chapter09/01-chapter-content/qr_code_scanner.py


# 级联回归 人脸 人眼检测 + 卡通眼睛
# https://github.com/PacktPublishing/Mastering-OpenCV-4-with-Python/blob/master/Chapter09/01-chapter-content/snapchat_augmeted_reality_glasses.py

```





```python


```





```python


```





```python


```





```python


```





```python


```





```python


```




```python


```



```python


```




```python


```





```python


```





```python


```





```python


```





```python


```





```python


```




```python


```




```python


```





```python


```





```python


```





```python


```





```python


```





```python


```





```python


```




```python


```
