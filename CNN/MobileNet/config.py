#-*- coding: utf-8 -*-
# 参数配置文件
import argparse

parser = argparse.ArgumentParser(description="MobileNetV2")
parser.add_mutually_exclusive_group(required=False)

parser.add_argument('--model_name', type=str, default='mobilenetv2')
parser.add_argument('--dataset_dir', type=str, default='./tfrecords', help='tfrecord file dir')
parser.add_argument('--num_samples', type=int, help='the number of train samples')
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_classes', type=int)
parser.add_argument('--height', type=int, default=224)
parser.add_argument('--width', type=int, default=224)
parser.add_argument('--is_train', dest='is_train', action='store_true')
parser.add_argument('--is_test', dest='is_train', action='store_false')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--lr_decay', type=float, default=0.98)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
parser.add_argument('--logs_dir', type=str, default='logs')
parser.add_argument('--rand_crop', dest='rand_crop', action='store_true')
parser.add_argument('--no_rand_crop', dest='rand_crop', action='store_false')
parser.add_argument('--cpu', dest='cpu', action='store_true')
parser.add_argument('--gpu', dest='cpu', action='store_false')
parser.add_argument('--renew', dest='renew', action='store_true')
# set default
parser.set_defaults(is_train=True)
parser.set_defaults(cpu=False)
parser.set_defaults(rand_crop=False)
parser.set_defaults(renew=False)

args = parser.parse_args()
