import sys
# add your caffe/python path
sys.path.insert(0, "caffe/python")
import caffe
import sys
caffe.set_mode_cpu()
import numpy as np
from numpy import prod, sum
from pprint import pprint

def print_net_parameters_flops (deploy_file):
    print ("Net: " + deploy_file)
    net = caffe.Net(deploy_file, caffe.TEST)
    flops = 0
    typenames = ['Convolution', 'DepthwiseConvolution', 'InnerProduct']

    print ("Layer-wise parameters: ")
    print ('layer name'.ljust(20), 'Filter Shape'.ljust(20), \
            'Output Size'.ljust(20), 'Layer Type'.ljust(20), 'Flops'.ljust(20))

    for layer_name, blob in net.blobs.items():
        if layer_name not in net.layer_dict:
            continue
        if net.layer_dict[layer_name].type in typenames:
            cur_flops = 0.0
            if net.layer_dict[layer_name].type in typenames[:2]:
                cur_flops = (np.product(net.params[layer_name][0].data.shape) * \
                        blob.data.shape[-1] * blob.data.shape[-2])
            else:
                cur_flops = np.product(net.params[layer_name][0].data.shape)
            print(layer_name.ljust(20),
                    str(net.params[layer_name][0].data.shape).ljust(20),
                    str(blob.data.shape).ljust(20),
                    net.layer_dict[layer_name].type.ljust(20), str(cur_flops).ljust(20))
            # InnerProduct
            if len(blob.data.shape) == 2:
                flops += prod(net.params[layer_name][0].data.shape)
            else:
                flops += prod(net.params[layer_name][0].data.shape) * blob.data.shape[2] * blob.data.shape[3]

    print ('layers num: ' + str(len(net.params.items())))
    print ("Total number of parameters: " + str(sum([prod(v[0].data.shape) for k, v in net.params.items()])))
    print ("Total number of flops: " + str(flops))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print ('Usage:')
        print ('python calc_params.py  deploy.prototxt')
        exit()
    deploy_file = sys.argv[1]
    print_net_parameters_flops(deploy_file)
