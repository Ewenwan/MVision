import caffe
import numpy as np

import detect_tool as tool

pic_name = '/home/zxd/projects/darknet-master/data/person.jpg'
# caffe.set_device(0)
# caffe.set_mode_gpu()
caffe.set_mode_cpu()
image = caffe.io.load_image(pic_name)
transformer = caffe.io.Transformer({'data': (1, 3, 416, 416)})
transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
transformed_image = transformer.preprocess('data', image)
print transformed_image.shape

model_def = '/home/zxd/projects/test/prototxt_test/yolo.prototxt'
model_weights = '/home/zxd/projects/PycharmProjects/test_caffe/yolo.caffemodel'

net = caffe.Net(model_def, model_weights, caffe.TEST)
net.blobs['data'].reshape(1, 3, 416, 416)
net.blobs['data'].data[...] = transformed_image
output = net.forward()
feat = net.blobs['region1'].data[0]
print feat.shape

boxes_of_each_grid = 5
classes = 80
thread = 0.45
biases = np.array([0.738768, 0.874946, 2.42204, 2.65704, 4.30971, 7.04493, 10.246, 4.59428, 12.6868, 11.8741])
boxes = tool.get_region_boxes(feat, boxes_of_each_grid, classes, thread, biases)

for box in boxes:
    print box

tool.draw_image(pic_name, boxes=boxes, namelist_file='coco.names')
