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




```python


```
