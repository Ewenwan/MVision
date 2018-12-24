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


# difference = cv2.subtract(bgr_image, temp)# 矩阵相减
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
