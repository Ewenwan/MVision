# coding=utf-8
# k-means ++ for YOLOv2 anchors
# 通过k-means ++ 算法获取YOLOv2需要的anchors的尺寸
import numpy as np

# 定义Box类，描述bounding box的坐标
class Box():
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h


# 计算两个box在某个轴上的重叠部分
# x1是box1的中心在该轴上的坐标
# len1是box1在该轴上的长度
# x2是box2的中心在该轴上的坐标
# len2是box2在该轴上的长度
# 返回值是该轴上重叠的长度
def overlap(x1, len1, x2, len2):
    len1_half = len1 / 2
    len2_half = len2 / 2

    left = max(x1 - len1_half, x2 - len2_half)#左边交点
    right = min(x1 + len1_half, x2 + len2_half)#右边交点

    return right - left#重叠长度


# 计算box a 和box b 的交集面积
# a和b都是Box类型实例
# 返回值area是box a 和box b 的交集面积
def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w)
    h = overlap(a.y, a.h, b.y, b.h)
    if w < 0 or h < 0:
        return 0

    area = w * h
    return area


# 计算 box a 和 box b 的并集面积
# a和b都是Box类型实例
# 返回值u是box a 和box b 的并集面积
def box_union(a, b):
    i = box_intersection(a, b)
    u = a.w * a.h + b.w * b.h - i
    return u


# 计算 box a 和 box b 的 iou
# a和b都是Box类型实例
# 返回值是box a 和box b 的iou
def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b)


# 使用k-means ++ 初始化 centroids，减少随机初始化的centroids对最终结果的影响
# boxes是所有bounding boxes的Box对象列表
# n_anchors是k-means的k值
# 返回值centroids 是初始化的n_anchors个centroid
def init_centroids(boxes,n_anchors):
    centroids = []
    boxes_num = len(boxes)

    centroid_index = np.random.choice(boxes_num, 1)#随机选择一个初始中心
    centroids.append(boxes[centroid_index])

    print(centroids[0].w,centroids[0].h)

    for centroid_index in range(0,n_anchors-1):#再选出 n_anchors-1个框

        sum_distance = 0
        distance_thresh = 0
        distance_list = []
        cur_sum = 0

        for box in boxes:#筛选每一个 box框
            min_distance = 1
            for centroid_i, centroid in enumerate(centroids):# index 实例 每一个中心框
                distance = (1 - box_iou(box, centroid))# 当前框和 中心框的交叠程度逆  dis越大 交叠的越少
                if distance < min_distance:
                    min_distance = distance#最小的距离  最相似 本box框离哪个中心最近
            sum_distance += min_distance
            distance_list.append(min_distance)

        distance_thresh = sum_distance*np.random.random()

        for i in range(0,boxes_num):#对于所以的框
            cur_sum += distance_list[i]#当前距离和
            if cur_sum > distance_thresh:#距离超过阈值
                centroids.append(boxes[i])# 新添加一个聚类中心框
                print(boxes[i].w, boxes[i].h)
                break#结束本次循环

    return centroids


# 进行 k-means 计算新的centroids
# boxes是所有bounding boxes的Box对象列表
# n_anchors是k-means的k值
# centroids是所有簇的中心
# 返回值new_centroids 是计算出的新簇中心
# 返回值groups是n_anchors个簇包含的boxes的列表
# 返回值loss是所有box距离所属的最近的centroid的距离的和
def do_kmeans(n_anchors, boxes, centroids):
    loss = 0
    groups = []
    new_centroids = []
    for i in range(n_anchors):#初始化每一个中心
        groups.append([])
        new_centroids.append(Box(0, 0, 0, 0))

    for box in boxes:#对于每一个数据 框
        min_distance = 1
        group_index = 0
        for centroid_index, centroid in enumerate(centroids):#遍历已经存在 中心框
            distance = (1 - box_iou(box, centroid))#距离
            if distance < min_distance:
                min_distance = distance#离那个中心最近
                group_index = centroid_index#记录所属的中心 索引
        groups[group_index].append(box)#添加到 对应的中心簇
        loss += min_distance#离中心的距离
        new_centroids[group_index].w += box.w#该中心所以边框大小
        new_centroids[group_index].h += box.h

    for i in range(n_anchors):
        new_centroids[i].w /= len(groups[i])#边框大小均值
        new_centroids[i].h /= len(groups[i])

    return new_centroids, groups, loss


# 计算给定bounding boxes的n_anchors数量的centroids
# label_path是训练集列表文件地址
# n_anchors 是anchors的数量
# loss_convergence是允许的loss的最小变化值
# grid_size * grid_size 是栅格数量
# iterations_num是最大迭代次数
# plus = 1时启用k means ++ 初始化centroids
def compute_centroids(label_path,n_anchors,loss_convergence,grid_size,iterations_num,plus):

    boxes = []
    label_files = []
    f = open(label_path)
    for line in f:
        label_path = line.rstrip().replace('images', 'labels')
        label_path = label_path.replace('JPEGImages', 'labels')
        label_path = label_path.replace('.jpg', '.txt')
        label_path = label_path.replace('.JPEG', '.txt')
        label_files.append(label_path)#标记框
    f.close()

    for label_file in label_files:
        f = open(label_file)
        for line in f:#每一个文件
            temp = line.strip().split(" ")# 按 " " 分割读取
            if len(temp) > 1:
                boxes.append(Box(0, 0, float(temp[3]), float(temp[4])))#提取 宽度和高度

    if plus:
        centroids = init_centroids(boxes, n_anchors)# 按距离 初始化 聚类中心
    else:
        centroid_indices = np.random.choice(len(boxes), n_anchors)#随机选择 初始化
        centroids = []
        for centroid_index in centroid_indices:
            centroids.append(boxes[centroid_index])

    # 迭代 聚类 iterate k-means
    centroids, groups, old_loss = do_kmeans(n_anchors, boxes, centroids)
    iterations = 1
    while (True):
        centroids, groups, loss = do_kmeans(n_anchors, boxes, centroids)
        iterations = iterations + 1
        print("loss = %f" % loss)# 返回值loss是所有box距离所属的最近的centroid的距离的和
        if abs(old_loss - loss) < loss_convergence or iterations > iterations_num:
            break# 两次loss 之差过小 / 迭代次数超过限制
        old_loss = loss

        for centroid in centroids:
            print(centroid.w * grid_size, centroid.h * grid_size)

    # print result
    for centroid in centroids:
        print("k-means result：\n")
        print(centroid.w * grid_size, centroid.h * grid_size)


label_path = "/raid/pengchong_data/Data/Lists/paul_train.txt"
n_anchors = 5# 聚类中心数量
loss_convergence = 1e-6# 两次loss 之差 阈值
grid_size = 13# 格子数量 13*13
iterations_num = 100# 最大迭代100次
plus = 0# 随机初始化聚类中心
compute_centroids(label_path,n_anchors,loss_convergence,grid_size,iterations_num,plus)
