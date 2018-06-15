#!/bin/sh
# 由fvb+idt生成的 编码后的特征文件 生成 多类别标签数据集
# 格式：
# 1 特征
# 1 特征
# 2 特征
# ...
# 10 特征

# Command "realpath" is used in this tool. Please install it
# first if you don't have it installed.

indir=	  # input dir 文件夹路径
outfile=  # output file 输出文件夹
classind= # class index   类别:id map
#key='gfzy'      # hog gof mbhx mbhy gfzy查找的关键字
# 处理命令行输入
while getopts i:o:c:k c
do
	case ${c} in 
	i) indir=${OPTARG};;
	o) outfile=${OPTARG};;
	c) classind=${OPTARG};;
# k) key=${OPTARG};;
	?) # Unknown option
		echo "./gen_label_lists.sh -i <in_dir> -o <out_file> -c <class_index> -k <searck_key>"
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
#for file in $(find *$(key)*) # 每一个文件的路径
for file in $(find *gfzy*) # 每一个文件的路径
do
	#dir=${dir%%/}
	echo "Processing file $file ..."
	file_name=`basename $file .avi.gfzy.fv.txt`
	#echo $file_name | cut -b 3-
	class_name=$(echo $file_name | cut -b 3-| cut -d _ -f 1)
    # 获取标签id
	label=$(grep -w "$class_name" $classind | cut -d" " -f1)
	file_con=$(cat $file)
	#do
	echo "$label $file_con" >> $outfile
	#
done
pwd
echo "Done!"
# 生成libsvm的估计数据格式
#printf 默认不带换行符 print默认带换行符
#cat $outfile | awk '{for(i=1;i<=NF;i++){if(i==1) printf $i " "; else printf i-1 ":" $i " ";} print '\n'}' | tee $outfile
