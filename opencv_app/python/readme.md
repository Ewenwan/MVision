# python opencv  接口

[参考](https://github.com/PacktPublishing/Mastering-OpenCV-4-with-Python)

## 1. 简单图形载入 灰度化 显示
```python
# 导入包
import cv2
# 默认方式读取图片
image = cv2.imread("logo.png")

# 彩色图灰度化 (BGR to GRAY):
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_height, gray_width = gray_image.shape

# 自适应窗口显示图像
cv2.imshow("OpenCV logo", image)

# 显示灰度图
cv2.imshow("OpenCV logo gray format", gray_image)

# 保存文件
cv2.imwrite("./gray", gray_image)


# difference = cv2.subtract(bgr_image, temp)# 矩阵相减
# b, g, r = cv2.split(difference)           # 矩阵通道分割  merge() 合并
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
