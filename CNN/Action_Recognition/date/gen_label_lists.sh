#!/bin/sh
# Command "realpath" is used in this tool. Please install it
# first if you don't have it installed.

indir=	# input dir 文件夹路径
outfile=	# output file 输出文件夹
classind= # class index   类别:id map
# 处理命令行输入
while getopts i:o:c: c
do
	case ${c} in 
	i) indir=${OPTARG};;
	o) outfile=${OPTARG};;
	c) classind=${OPTARG};;
	?) # Unknown option
		echo "gen_label_lists -i <in_dir> -o <out_file> -c <class_index>"
		exit;;
	esac
done
#打印输出文件名称
> $outfile
# 获取完整地址
outfile=$(realpath $outfile)
classind=$(realpath $classind)

# 到文件夹下获取指定文件
cd $indir
for dir in $(ls -d */)#每一个文件的路径
do
	dir=${dir%%/}
	echo "Processing directory $dir ..."
    # 获取文件标签
	label=$(grep -w "$dir" $classind | cut -d" " -f1)
    # 获取文件名称
	for v in $(ls $dir/*.avi)
	# 写入新的文件 文件名+标签
	do
		echo "$v $label" >> $outfile
	done
done

echo "Done!"
