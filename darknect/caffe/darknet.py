import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from cfg import *

class FCView(nn.Module):
    def __init__(self):
        super(FCView, self).__init__()

    def forward(self, x):
        nB = x.data.size(0)
        x = x.view(nB,-1)
        return x
    def __repr__(self):
        return 'view(nB, -1)'

class MaxPoolStride1(nn.Module):
    def __init__(self):
        super(MaxPoolStride1, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, (0,1,0,1), mode='replicate'), 2, stride=1)
        return x

class Reorg(nn.Module):
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride
    def forward(self, x):
        stride = self.stride
        assert(x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert(H % stride == 0)
        assert(W % stride == 0)
        ws = stride
        hs = stride
        x = x.view(B, C, H/hs, hs, W/ws, ws).transpose(3,4).contiguous()
        x = x.view(B, C, H/hs*W/ws, hs*ws).transpose(2,3).contiguous()
        x = x.view(B, C, hs*ws, H/hs, W/ws).transpose(1,2).contiguous()
        x = x.view(B, hs*ws*C, H/hs, W/ws)
        return x

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        N = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = F.avg_pool2d(x, (H, W))
        x = x.view(N, C)
        return x

# for route and shortcut
class EmptyModule(nn.Module):
    def __init__(self):
        super(EmptyModule, self).__init__()

    def forward(self, x):
        return x

# support route shortcut and reorg
class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.models = self.create_network(self.blocks) # merge conv, bn,leaky
        self.loss = self.models[len(self.models)-1]

        self.width = int(self.blocks[0]['width'])
        self.height = int(self.blocks[0]['height'])

        if self.blocks[(len(self.blocks)-1)]['type'] == 'region':
            region_block = self.blocks[(len(self.blocks)-1)]
            anchors = region_block['anchors'].split(',')
            self.anchors = [float(i) for i in anchors]
            self.num_anchors = int(region_block['num'])
            self.anchor_step = len(self.anchors)/self.num_anchors
            self.num_classes = int(region_block['classes'])

        self.header = torch.IntTensor([0,0,0,0])
        self.seen = 0
        self.has_mean = False

    def forward(self, x):
        if self.has_mean:
            batch_size = x.data.size(0)
            x = x - torch.autograd.Variable(self.mean_img.repeat(batch_size, 1, 1, 1))

        ind = -2
        self.loss = None
        outputs = dict()
        for block in self.blocks:
            ind = ind + 1
            #if ind > 14:
            #    return x

            if block['type'] == 'net':
                continue
            elif block['type'] == 'convolutional' or block['type'] == 'maxpool' or block['type'] == 'reorg' or block['type'] == 'avgpool' or block['type'] == 'softmax' or block['type'] == 'connected' or block['type'] == 'dropout':
                x = self.models[ind](x)
                outputs[ind] = x
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
                if len(layers) == 1:
                    x = outputs[layers[0]]
                    outputs[ind] = x
                elif len(layers) == 2:
                    x1 = outputs[layers[0]]
                    x2 = outputs[layers[1]]
                    x = torch.cat((x1,x2),1)
                    outputs[ind] = x
            elif block['type'] == 'shortcut':
                from_layer = int(block['from'])
                activation = block['activation']
                from_layer = from_layer if from_layer > 0 else from_layer + ind
                x1 = outputs[from_layer]
                x2 = outputs[ind-1]
                x  = x1 + x2
                if activation == 'leaky':
                    x = F.leaky_relu(x, 0.1, inplace=True)
                elif activation == 'relu':
                    x = F.relu(x, inplace=True)
                outputs[ind] = x
            elif block['type'] == 'cost':
                continue
            elif block['type'] == 'region':
                continue
            else:
                print('unknown type %s' % (block['type']))
        return x

    def print_network(self):
        print_cfg(self.blocks)

    def create_network(self, blocks):
        models = nn.ModuleList()
    
        prev_filters = 3
        out_filters =[]
        out_width =[]
        out_height =[]
        conv_id = 0
        prev_width = 0
        prev_height = 0
        for block in blocks:
            if block['type'] == 'net':
                prev_filters = int(block['channels'])
                prev_width = int(block['width'])
                prev_height = int(block['height'])
                continue
            elif block['type'] == 'convolutional':
                conv_id = conv_id + 1
                batch_normalize = int(block['batch_normalize'])
                filters = int(block['filters'])
                kernel_size = int(block['size'])
                stride = int(block['stride'])
                is_pad = int(block['pad'])
                pad = (kernel_size-1)/2 if is_pad else 0
                activation = block['activation']
                model = nn.Sequential()
                if batch_normalize:
                    model.add_module('conv{0}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=False))
                    model.add_module('bn{0}'.format(conv_id), nn.BatchNorm2d(filters))
                else:
                    model.add_module('conv{0}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad))
                if activation == 'leaky':
                    model.add_module('leaky{0}'.format(conv_id), nn.LeakyReLU(0.1, inplace=True))
                elif activation == 'relu':
                    model.add_module('relu{0}'.format(conv_id), nn.ReLU(inplace=True))
                prev_filters = filters
                prev_width = (prev_width + 2*pad - kernel_size)/stride + 1
                prev_height = (prev_height + 2*pad - kernel_size)/stride + 1
                out_filters.append(prev_filters)
                out_width.append(prev_width)
                out_height.append(prev_height)
                models.append(model)
            elif block['type'] == 'maxpool':
                pool_size = int(block['size'])
                stride = int(block['stride'])
                padding = 0
                if block.has_key('pad') and int(block['pad']) == 1:
                    padding = int((pool_size-1)/2)
                if stride > 1:
                    model = nn.MaxPool2d(pool_size, stride, padding=padding)
                else:
                    model = MaxPoolStride1()
                out_filters.append(prev_filters)
                if stride > 1:
                    prev_width = (prev_width - kernel_size + 1)/stride + 1
                    prev_height = (prev_height - kernel_size + 1)/stride + 1
                else:
                    prev_width = prev_width - kernel_size + 1
                    prev_height = prev_height - kernel_size + 1
                out_width.append(prev_width)
                out_height.append(prev_height)
                models.append(model)
            elif block['type'] == 'avgpool':
                model = GlobalAvgPool2d()
                out_filters.append(prev_filters)
                prev_width = 1
                prev_height = 1
                out_width.append(prev_width)
                out_height.append(prev_height)
                models.append(model)
            elif block['type'] == 'softmax':
                model = nn.Softmax()
                prev_width = 1
                prev_height = 1
                out_filters.append(prev_filters)
                out_width.append(prev_width)
                out_height.append(prev_height)
                models.append(model)
            elif block['type'] == 'cost':
                if block['_type'] == 'sse':
                    model = nn.MSELoss(size_average=True)
                elif block['_type'] == 'L1':
                    model = nn.L1Loss(size_average=True)
                elif block['_type'] == 'smooth':
                    model = nn.SmoothL1Loss(size_average=True)
                prev_width = 1
                prev_height = 1
                out_filters.append(1)
                out_width.append(prev_width)
                out_height.append(prev_height)
                models.append(model)
            elif block['type'] == 'reorg':
                stride = int(block['stride'])
                prev_filters = stride * stride * prev_filters
                prev_width = prev_width / stride
                prev_height = prev_height / stride
                out_filters.append(prev_filters)
                out_width.append(prev_width)
                out_height.append(prev_height)
                models.append(Reorg(stride))
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                ind = len(models)
                layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
                if len(layers) == 1:
                    prev_filters = out_filters[layers[0]]
                    prev_width = out_width[layers[0]]
                    prev_height = out_height[layers[0]]
                elif len(layers) == 2:
                    assert(layers[0] == ind - 1)
                    prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
                    prev_width = out_width[layers[0]]
                    prev_height = out_height[layers[0]]
                out_filters.append(prev_filters)
                out_width.append(prev_width)
                out_height.append(prev_height)
                models.append(EmptyModule())
            elif block['type'] == 'shortcut':
                ind = len(models)
                prev_filters = out_filters[ind-1]
                prev_width = out_width[ind-1]
                prev_height = out_height[ind-1]
                out_filters.append(prev_filters)
                out_width.append(prev_width)
                out_height.append(prev_height)
                models.append(EmptyModule())
            elif block['type'] == 'dropout':
                ind = len(models)
                ratio = float(block['probability'])
                prev_filters = out_filters[ind-1]
                prev_width = out_width[ind-1]
                prev_height = out_height[ind-1]
                out_filters.append(prev_filters)
                out_width.append(prev_width)
                out_height.append(prev_height)
                models.append(nn.Dropout2d(ratio))
            elif block['type'] == 'connected':
                prev_filters = prev_filters * prev_width * prev_height
                filters = int(block['output'])
                is_first = (prev_width * prev_height != 1)
                if block['activation'] == 'linear':
                    if is_first:
                        model = nn.Sequential(FCView(), nn.Linear(prev_filters, filters))
                    else:
                        model = nn.Linear(prev_filters, filters)
                elif block['activation'] == 'leaky':
                    if is_first:
                        model = nn.Sequential(
                                   FCView(),
                                   nn.Linear(prev_filters, filters),
                                   nn.LeakyReLU(0.1, inplace=True))
                    else:
                        model = nn.Sequential(
                                   nn.Linear(prev_filters, filters),
                                   nn.LeakyReLU(0.1, inplace=True))
                elif block['activation'] == 'relu':
                    if is_first:
                        model = nn.Sequential(
                                   FCView(),
                                   nn.Linear(prev_filters, filters),
                                   nn.ReLU(inplace=True))
                    else:
                        model = nn.Sequential(
                                   nn.Linear(prev_filters, filters),
                                   nn.ReLU(inplace=True))
                prev_filters = filters
                prev_width = 1
                prev_height = 1
                out_filters.append(prev_filters)
                out_width.append(prev_width)
                out_height.append(prev_height)
                models.append(model)
            elif block['type'] == 'region':
                continue
            else:
                print('unknown type %s' % (block['type']))
    
        return models

    def load_weights(self, weightfile):
        if self.blocks[0].has_key('mean_file'):
            import caffe_pb2
            mean_file = self.blocks[0]['mean_file']
            mean_file = mean_file.strip('"')
            self.has_mean = True
            blob = caffe_pb2.BlobProto()
            blob.ParseFromString(open(mean_file, 'rb').read())
            mean_img = torch.from_numpy(np.array(blob.data)).float()
            channels = int(self.blocks[0]['channels'])
            height = int(self.blocks[0]['height'])
            width = int(self.blocks[0]['width'])
            mean_img = mean_img.view(channels, height, width)

            self.register_buffer('mean_img', torch.zeros(channels, height, width))
            self.mean_img.copy_(mean_img)

        fp = open(weightfile, 'rb')
        header = np.fromfile(fp, count=4, dtype=np.int32)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        buf = np.fromfile(fp, dtype = np.float32)
        fp.close()

        start = 0
        ind = -2
        is_first = True;
        for block in self.blocks:
            if start >= buf.size:
                break
            ind = ind + 1
            if block['type'] == 'net':
                continue
            elif block['type'] == 'convolutional':
                model = self.models[ind]
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    start = load_conv_bn(buf, start, model[0], model[1])
                else:
                    start = load_conv(buf, start, model[0])
            elif block['type'] == 'connected':
                model = self.models[ind]
                if block['activation'] != 'linear':
                    if is_first:
                        start = load_fc(buf, start, model[1])
                        is_first = False;
                    else:
                        start = load_fc(buf, start, model[0])
                else:
                    if is_first:
                        start = load_fc(buf, start, model[1])
                        is_first = False;
                    else:
                        start = load_fc(buf, start, model)
            elif block['type'] == 'maxpool':
                pass
            elif block['type'] == 'reorg':
                pass
            elif block['type'] == 'route':
                pass
            elif block['type'] == 'shortcut':
                pass
            elif block['type'] == 'avgpool':
                pass
            elif block['type'] == 'softmax':
                pass
            elif block['type'] == 'cost':
                pass
            elif block['type'] == 'region':
                pass
            elif block['type'] == 'dropout':
                pass
            else:
                print('unknown type %s' % (block['type']))

    def save_weights(self, outfile, cutoff=0):
        if cutoff <= 0:
            cutoff = len(self.blocks)-1

        fp = open(outfile, 'wb')
        self.header[3] = self.seen
        header = self.header
        header.numpy().tofile(fp)

        ind = -1
        is_first = True;
        for blockId in range(1, cutoff+1):
            ind = ind + 1
            block = self.blocks[blockId]
            if block['type'] == 'convolutional':
                model = self.models[ind]
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    save_conv_bn(fp, model[0], model[1])
                else:
                    save_conv(fp, model[0])
            elif block['type'] == 'connected':
                model = self.models[ind]
                if block['activation'] == 'linear':
                    if is_first:
                        save_fc(fp, model[1])
                        is_first = False
                    else:
                        save_fc(fp, model)
                else:
                    if is_first:
                        save_fc(fp, model[1])
                        is_first = False;
                    else:
                        save_fc(fp, model[0])
            elif block['type'] == 'maxpool':
                pass
            elif block['type'] == 'reorg':
                pass
            elif block['type'] == 'route':
                pass
            elif block['type'] == 'shortcut':
                pass
            elif block['type'] == 'avgpool':
                pass
            elif block['type'] == 'softmax':
                pass
            elif block['type'] == 'cost':
                pass
            elif block['type'] == 'region':
                pass
            elif block['type'] == 'dropout':
                pass
            else:
                print('unknown type %s' % (block['type']))
        fp.close()

    def save_shrink_model(self, out_cfgfile, out_weightfile, cutoff=0):
        if cutoff <= 0:
            cutoff = len(self.blocks)-1

        # shrink weights
        fp = open(out_weightfile, 'wb')
        self.header[3] = self.seen
        header = self.header
        header.numpy().tofile(fp)

        ind = -1
        is_first = True
        for blockId in range(1, cutoff+1):
            ind = ind + 1
            block = self.blocks[blockId]
            if block['type'] == 'convolutional':
                model = self.models[ind]
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    save_conv_shrink_bn(fp, model[0], model[1])
                else:
                    save_conv(fp, model[0])
            elif block['type'] == 'connected':
                model = self.models[ind]
                if block['activation'] == 'linear':
                    if is_first:
                        save_fc(fp, model[1])
                        is_first = False
                    else:
                        save_fc(fp, model)
                else:
                    if is_first:
                        save_fc(fp, model[1])
                        is_first = False
                    else:
                        save_fc(fp, model[0])
            elif block['type'] == 'maxpool':
                pass
            elif block['type'] == 'reorg':
                pass
            elif block['type'] == 'route':
                pass
            elif block['type'] == 'shortcut':
                pass
            elif block['type'] == 'avgpool':
                pass
            elif block['type'] == 'softmax':
                pass
            elif block['type'] == 'cost':
                pass
            elif block['type'] == 'region':
                pass
            elif block['type'] == 'dropout':
                pass
            else:
                print('unknown type %s' % (block['type']))
        fp.close()

        # shrink cfg file
        blocks = self.blocks
        for block in blocks:
            if block['type'] == 'convolutional':
               block['batch_normalize'] = '0' 
        save_cfg(blocks, out_cfgfile)

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 5:
        print('try:')
        print('python darknet.py tiny-yolo-voc.cfg tiny-yolo-voc.weights tiny-yolo-voc-nobn.cfg tiny-yolo-voc-nobn.weights')
        print('')
        exit()

    in_cfgfile = sys.argv[1]
    in_weightfile = sys.argv[2]
    out_cfgfile = sys.argv[3]
    out_weightfile = sys.argv[4]
    model = Darknet(in_cfgfile)
    model.load_weights(in_weightfile)
    print('save %s' % out_cfgfile)
    print('save %s' % out_weightfile)
    model.save_shrink_model(out_cfgfile, out_weightfile)
