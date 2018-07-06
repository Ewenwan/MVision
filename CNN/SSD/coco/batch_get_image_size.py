import os
import subprocess
import sys

HOMEDIR = os.path.expanduser("~")
CURDIR = os.path.dirname(os.path.realpath(__file__))

### Modify the address and parameters accordingly ###
# If true, redo the whole thing.
redo = True
# The caffe root.
# 数据集路径
#CAFFE_ROOT = "{}/projects/caffe".format(HOMEDIR)
# The root directory which stores the coco images, annotations, etc.
coco_data_dir = "{}/coco".format(HOMEDIR)
# The sets that we want to get the size info.
anno_sets = ["instances_train2017", "instances_val2017"]
# The directory which contains the full annotation files for each set.
anno_dir = "{}/annotations".format(coco_data_dir)
# The directory which stores the imageset information for each set.
imgset_dir = "{}/ImageSets".format(coco_data_dir)
# The directory which stores the image id and size info.
out_dir = "{}/lmdb".format(coco_data_dir)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

### Get image size info ###
for i in xrange(0, len(anno_sets)):
    anno_set = anno_sets[i]
    anno_file = "{}/{}.json".format(anno_dir, anno_set)
    if not os.path.exists(anno_file):
        continue
    anno_name = anno_set.split("_")[-1]
    imgset_file = "{}/{}.txt".format(imgset_dir, anno_name)
    if not os.path.exists(imgset_file):
        print "{} does not exist".format(imgset_file)
        sys.exit()
    name_size_file = "{}/{}_name_size.txt".format(out_dir, anno_name)
    if redo or not os.path.exists(out_dir):
        cmd = "python {}/get_image_size.py {} {} {}" \
                .format(CURDIR, anno_file, imgset_file, name_size_file)
        print cmd
        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        output = process.communicate()[0]
        print output
