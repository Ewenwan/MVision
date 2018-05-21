import Image
import ImageDraw
import ImageFont
import math

import matplotlib.image as mpimg


class Point(object):
    def __init__(self, x=0., y=0.):
        self.x = x
        self.y = y

    def __str__(self):
        return "({},{})".format(self.x, self.y)


class Rectangle(object):
    def __init__(self, posn, w, h):
        self.corner = posn
        self.width = w
        self.height = h

    def __str__(self):
        return "({0},{1},{2})".format(self.corner, self.width, self.height)

    def iou(self, rect):
        return self.intersection(rect) / self.union(rect)

    def intersection(self, rect):
        w = overlap(self.corner.x, self.width, rect.corner.x, rect.width)
        h = overlap(self.corner.y, self.height, rect.corner.y, rect.height)
        if w < 0 or h < 0:
            return 0
        area = w * h
        return area

    def union(self, rect):
        i = self.intersection(rect)
        u = self.width * self.height + rect.width * rect.height - i
        return u


class Box(object):
    def __init__(self, rect, prob=0.0, category=-1):
        self.rect = rect
        self.prob = prob
        self.category = category

    def __str__(self):
        return "({0},{1},{2})".format(self.rect, self.prob, self.category)

    def iou(self, box2):
        return self.rect.iou(box2.rect)


def overlap(x1, w1, x2, w2):
    l1 = x1 - w1 / 2
    l2 = x2 - w2 / 2
    left = l1 if l1 > l2 else l2
    r1 = x1 + w1 / 2
    r2 = x2 + w2 / 2
    right = r1 if r1 < r2 else r2
    return right - left


def logistic_activate(x):
    return 1. / (1. + math.exp(-x))


def get_region_box(x, biases, n, index, i, j, w, h):
    rect = Rectangle(Point((i + logistic_activate(x[index + 0])) / w,
                           (j + logistic_activate(x[index + 1])) / h),
                     math.exp(x[index + 2]) * biases[2 * n] / w,
                     math.exp(x[index + 3]) * biases[2 * n + 1] / h)
    box = Box(rect)
    return box


def get_region_boxes(feat, boxes_of_each_grid, classes, thresh, biases, nms=0.4):
    boxes = []
    channel, height, width = feat.shape
    predictions = feat.reshape(-1)
    for i in xrange(height * width):
        row = i / width
        col = i % width
        for n in xrange(boxes_of_each_grid):
            index = i * boxes_of_each_grid + n
            p_index = index * (classes + 5) + 4
            scale = predictions[p_index]
            box_index = index * (classes + 5)
            box_tmp = get_region_box(predictions, biases, n, box_index, col, row, width, height)
            class_index = index * (classes + 5) + 5
            for j in xrange(classes):
                prob = scale * predictions[class_index + j]
                if prob > thresh:
                    box_tmp.category = j
                    box_tmp.prob = prob
                    boxes.append(box_tmp)
    result = []
    for i in xrange(boxes.__len__()):
        for j in xrange(i + 1, boxes.__len__()):
            if boxes[i].iou(boxes[j]) > nms:
                if boxes[i].prob > boxes[j].prob:
                    boxes[j].prob = 0
                else:
                    boxes[i].prob = 0
    for box in boxes:
        if box.prob > 0:
            result.append(box)
    del boxes
    return result


def get_names_from_file(filename):
    result = []
    fd = file(filename, 'r')
    for line in fd.readlines():
        result.append(line.replace('\n', ''))
    return result


def get_color_from_file(filename):
    colors = []
    fd = file(filename, 'r')
    for line in fd.readlines():
        words = line.split(r',')
        color = (int(words[0]), int(words[1]), int(words[2]))
        colors.append(color)
    return colors


def draw_image(pic_name, boxes, namelist_file):
    name_list = get_names_from_file(namelist_file)
    color_list = get_color_from_file('ink.color')
    im = Image.open(pic_name)
    draw = ImageDraw.Draw(im)
    lena = mpimg.imread(pic_name)
    height, width = lena.shape[:2]
    for box in boxes:
        x = box.rect.corner.x
        y = box.rect.corner.y
        w = box.rect.width
        h = box.rect.height
        left = (x - w / 2) * width
        right = (x + w / 2) * width
        top = (y - h / 2) * height
        bot = (y + h / 2) * height
        if left < 0:
            left = 0
        if right > width - 1:
            right = width - 1
        if top < 0:
            top = 0
        if bot > height - 1:
            bot = height - 1
        category = name_list[box.category]
        color = color_list[box.category % color_list.__len__()]
        draw.line((left, top, right, top), fill=color, width=5)
        draw.line((right, top, right, bot), fill=color, width=5)
        draw.line((left, top, left, bot), fill=color, width=5)
        draw.line((left, bot, right, bot), fill=color, width=5)
        font_size = 20
        my_font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu-font-family/Ubuntu-M.ttf", size=font_size)
        draw.text([left + 5, top], category, font=my_font, fill=color)
    im.show()


def show_image(pic_name):
    im = Image.open(pic_name)
    # rect = [(0, 0), (300, 300)]
    # draw.rectangle([(0, 0), (200, 200)], outline='yellow', width=20)
    # draw.line((100, 2, 200, 500), fill='yellow', width=5)
    # im.save(pic_name, "PNG")
    im.show()
    # lena = mpimg.imread(pic_name)
    # print lena.shape
    # plt.imshow(lena)
    # plt.axis('off')
    # plt.show()
