#cur_dir=$(cd $( dirname ${BASH_SOURCE[0]} ) && pwd )
#echo $cur_dir
pwd
#root_dir= "../../../caffe-ssd"

#echo $root_dir

#cd $root_dir

redo=false
# 原始 coco 图片数据库
data_root_dir="./coco/"
dataset_name="coco"

# 标签id prototxt 文件
mapfile="labelmap_coco.prototxt"

# 标签 为检测格式的标签 边框和label
anno_type="detection"
# 标签格式
label_type="json"
# 生成的数据库格式
db="lmdb"
min_dim=0
max_dim=0
width=0
height=0

shuffle=True
encode_type='jpg'
encoded=True
gray=False
check_size=False
check_label=False

caffe_root="/home/wanyouwen/ewenwan/software/caffe-ssd"

# 子集合需要使用 python data/coco/create_list.py
for subset in train2017_img_label val2017_img_label
do
# python2 create_annoset.py --anno-type=$anno_type --label-type=$label_type --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd $data_root_dir $subset.txt $db/$dataset_name"_"$subset"_"$db examples/$dataset_name
$caffe_root/build/tools/convert_annoset \
         --anno_type=$anno_type \
		 --label_type=$label_type \
		 --label_map_file=$mapfile \
		 --check_label=$check_label\
         --min_dim=$min_dim \
         --max_dim=$max_dim \
         --resize_height=$height \
         --resize_width=$width \
         --backend=$db \
         --shuffle=$shuffle \
         --check_size=$check_size \
         --encode_type=$encode_type \
         --encoded=$encoded \
         --gray=$gray \
         $data_root_dir $subset.txt  $data_root_dir/$db/$dataset_name"_"$subset"_"$db
done
