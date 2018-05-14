// 目标识别 层
// 网络输出 根据 loss损失函数定义得到 识别结果
#include "detection_layer.h"
#include "activations.h"
#include "softmax_layer.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

// 创建 检测识别层
detection_layer make_detection_layer(int batch, int inputs, int n, int side, int classes, int coords, int rescore)
{
    detection_layer l = {0};
    l.type = DETECTION;// 层定义 

    l.n = n;//对应论文里面的B,表示每个cell生成的bounding box数。 yolo3  每个尺度 3个
    l.batch = batch;// 一次输入的图片数量 批次大小
    l.inputs = inputs;// 输入detection的维度，计算方式根据论文是S*S*B*(5 + C)，论文里面对应11*11*3（ 5 + 20）
    l.classes = classes;// 类别总数 论文里面对应是20 voc  coco 为 80
    l.coords = coords;//坐标，（x,y,w,h）, (x,y)是相对于网格的数值[0,1],(w,h)，是相对于 预设格子 指数映射
	                  // bw = pw * exp(w)
					  // bh = ph * exp(h)
					  // 4个参数
    l.rescore = rescore;// 得分
    l.side = side;// 最后特征特 尺寸  网格数量
    l.w = side;
    l.h = side;
    assert(side*side*((1 + l.coords)*l.n + l.classes) == inputs);
    l.cost = calloc(1, sizeof(float));//代价
    l.outputs = l.inputs;// 输出
    l.truths = l.side*l.side*(1 + l.coords + l.classes);// 最后输出的维度总大小
    l.output = calloc(batch*l.outputs, sizeof(float));// 批次
    l.delta = calloc(batch*l.outputs, sizeof(float));

    l.forward = forward_detection_layer;// 前向
    l.backward = backward_detection_layer;// 反向
#ifdef GPU
    l.forward_gpu = forward_detection_layer_gpu;
    l.backward_gpu = backward_detection_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

    fprintf(stderr, "Detection Layer\n");
    srand(0);

    return l;
}

// 前向 传播
void forward_detection_layer(const detection_layer l, network net)
{
    int locations = l.side*l.side;// 格子数量
    int i,j;
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));
    //if(l.reorg) reorg(l.output, l.w*l.h, size*l.n, l.batch, 1);
    int b;
    if (l.softmax){
        for(b = 0; b < l.batch; ++b){
            int index = b*l.inputs;
            for (i = 0; i < locations; ++i) {
                int offset = i*l.classes;
                softmax(l.output + index + offset, l.classes, 1, 1,
                        l.output + index + offset);
            }
        }
    }
	// 训练
    if(net.train){
        float avg_iou = 0;// 平均交并比
        float avg_cat = 0;
        float avg_allcat = 0;
        float avg_obj = 0;// 有目标
        float avg_anyobj = 0;
        int count = 0;
        *(l.cost) = 0;
        int size = l.inputs * l.batch;//总大小
        memset(l.delta, 0, size * sizeof(float));
        for (b = 0; b < l.batch; ++b){//每一个批次
            int index = b*l.inputs;//的每一次网络输出的每一个量 
            for (i = 0; i < locations; ++i) {//每个格子
                int truth_index = (b*locations + i) * (1+l.coords+l.classes);
				   //这个地方表示的是输入图片的gound truth的index
                int is_obj = net.truth[truth_index];// 前景 背景 //表示这个cell是否是object，数据存放在state里面
				
		//计算noobj这一项的损失
                for (j = 0; j < l.n; ++j) {
                    int p_index = index + locations*l.classes + i*l.n + j;
					//对应论文损失的第四项。这个计算很是奇怪，论文里面还是要判断一下是否是noobj？？？？他怎么没有判断就直接计算了
                    l.delta[p_index] = l.noobject_scale*(0 - l.output[p_index]);
                    *(l.cost) += l.noobject_scale*pow(l.output[p_index], 2);//同上
                    avg_anyobj += l.output[p_index];
                }

                int best_index = -1;
                float best_iou = 0;
                float best_rmse = 20;

                if (!is_obj){
                    continue;
                }
        //后面都是对应于object的情况。
                int class_index = index + i*l.classes;
                for(j = 0; j < l.classes; ++j) {
                    l.delta[class_index+j] = l.class_scale * (net.truth[truth_index+1+j] - l.output[class_index+j]);//对应论文五项损失的第五项
                    *(l.cost) += l.class_scale * pow(net.truth[truth_index+1+j] - l.output[class_index+j], 2);
                    if(net.truth[truth_index + 1 + j]) avg_cat += l.output[class_index+j];//float类型做判断这样写不太好。
                    avg_allcat += l.output[class_index+j];
                }//l.classes

                box truth = float_to_box(net.truth + truth_index + 1 + l.classes, 1);
                truth.x /= l.side;//这个地方的计算不清楚。
                truth.y /= l.side;

                for(j = 0; j < l.n; ++j){
                    int box_index = index + locations*(l.classes + l.n) + (i*l.n + j) * l.coords;
                    box out = float_to_box(l.output + box_index, 1);//cell里面预测的boundingbox
                    out.x /= l.side;//同上面的不清楚
                    out.y /= l.side;

                    if (l.sqrt){
                        out.w = out.w*out.w;
                        out.h = out.h*out.h;
                    }

                    float iou  = box_iou(out, truth);//计算iou值，即out与truth的交除以他们的并。
                    //iou = 0;
                    float rmse = box_rmse(out, truth);//这个是计算boundingbox各项之间的差的平方和的开根号。
                    if(best_iou > 0 || iou > 0){
                        if(iou > best_iou){
                            best_iou = iou;
                            best_index = j;
                        }
                    }else{
                        if(rmse < best_rmse){
                            best_rmse = rmse;
                            best_index = j;//存放cell里面最好的预测的boundingbox的idx
                        }
                    }
                }

                if(l.forced){
                    if(truth.w*truth.h < .1){
                        best_index = 1;
                    }else{
                        best_index = 0;
                    }
                }
                if(l.random && *(net.seen) < 64000){
                    best_index = rand()%l.n;
                }

                int box_index = index + locations*(l.classes + l.n) + (i*l.n + best_index) * l.coords;
                int tbox_index = truth_index + 1 + l.classes;

                box out = float_to_box(l.output + box_index, 1);
                out.x /= l.side;
                out.y /= l.side;
                if (l.sqrt) {
                    out.w = out.w*out.w;
                    out.h = out.h*out.h;
                }
                float iou  = box_iou(out, truth);

                   //printf("%d,", best_index);
                int p_index = index + locations*l.classes + i*l.n + best_index;
				   //对应于论文里面的第三项。
                *(l.cost) -= l.noobject_scale * pow(l.output[p_index], 2);
				   //这里揭开前面的问题，这里相当于已经判定是objcet，就把之前算在里面noobjcet的减掉。
                *(l.cost) += l.object_scale * pow(1-l.output[p_index], 2);
                avg_obj += l.output[p_index];
				   //这里揭开前面的问题，这里相当于已经判定是objcet，就把之前算在里面noobjcet的减掉。
                l.delta[p_index] = l.object_scale * (1.-l.output[p_index]);

                if(l.rescore){
                    l.delta[p_index] = l.object_scale * (iou - l.output[p_index]);
                }

                l.delta[box_index+0] = l.coord_scale*(net.truth[tbox_index + 0] - l.output[box_index + 0]);
                l.delta[box_index+1] = l.coord_scale*(net.truth[tbox_index + 1] - l.output[box_index + 1]);
                l.delta[box_index+2] = l.coord_scale*(net.truth[tbox_index + 2] - l.output[box_index + 2]);
                l.delta[box_index+3] = l.coord_scale*(net.truth[tbox_index + 3] - l.output[box_index + 3]);
                if(l.sqrt){
                    l.delta[box_index+2] = l.coord_scale*(sqrt(net.truth[tbox_index + 2]) - l.output[box_index + 2]);
                    l.delta[box_index+3] = l.coord_scale*(sqrt(net.truth[tbox_index + 3]) - l.output[box_index + 3]);
                }

                *(l.cost) += pow(1-iou, 2);
                avg_iou += iou;
                ++count;
            }
        }

        if(0){
            float *costs = calloc(l.batch*locations*l.n, sizeof(float));
            for (b = 0; b < l.batch; ++b) {
                int index = b*l.inputs;
                for (i = 0; i < locations; ++i) {
                    for (j = 0; j < l.n; ++j) {
                        int p_index = index + locations*l.classes + i*l.n + j;
                        costs[b*locations*l.n + i*l.n + j] = l.delta[p_index]*l.delta[p_index];
                    }
                }
            }
            int indexes[100];
            top_k(costs, l.batch*locations*l.n, 100, indexes);
            float cutoff = costs[indexes[99]];
            for (b = 0; b < l.batch; ++b) {
                int index = b*l.inputs;
                for (i = 0; i < locations; ++i) {
                    for (j = 0; j < l.n; ++j) {
                        int p_index = index + locations*l.classes + i*l.n + j;
                        if (l.delta[p_index]*l.delta[p_index] < cutoff) l.delta[p_index] = 0;
                    }
                }
            }
            free(costs);
        }


        *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);


        printf("Detection Avg IOU: %f, Pos Cat: %f, All Cat: %f, Pos Obj: %f, Any Obj: %f, count: %d\n", avg_iou/count, avg_cat/count, avg_allcat/(count*l.classes), avg_obj/count, avg_anyobj/(l.batch*locations*l.n), count);
        //if(l.reorg) reorg(l.delta, l.w*l.h, size*l.n, l.batch, 0);
    }
}

void backward_detection_layer(const detection_layer l, network net)
{
    axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
}

void get_detection_detections(layer l, int w, int h, float thresh, detection *dets)
{
    int i,j,n;
    float *predictions = l.output;
    //int per_cell = 5*num+classes;
    for (i = 0; i < l.side*l.side; ++i){
        int row = i / l.side;
        int col = i % l.side;
        for(n = 0; n < l.n; ++n){
            int index = i*l.n + n;
            int p_index = l.side*l.side*l.classes + i*l.n + n;
            float scale = predictions[p_index];
            int box_index = l.side*l.side*(l.classes + l.n) + (i*l.n + n)*4;
            box b;
            b.x = (predictions[box_index + 0] + col) / l.side * w;
            b.y = (predictions[box_index + 1] + row) / l.side * h;
            b.w = pow(predictions[box_index + 2], (l.sqrt?2:1)) * w;
            b.h = pow(predictions[box_index + 3], (l.sqrt?2:1)) * h;
            dets[index].bbox = b;
            dets[index].objectness = scale;
            for(j = 0; j < l.classes; ++j){
                int class_index = i*l.classes;
                float prob = scale*predictions[class_index+j];
                dets[index].prob[j] = (prob > thresh) ? prob : 0;
            }
        }
    }
}

#ifdef GPU

void forward_detection_layer_gpu(const detection_layer l, network net)
{
    if(!net.train){
        copy_gpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
        return;
    }

    cuda_pull_array(net.input_gpu, net.input, l.batch*l.inputs);
    forward_detection_layer(l, net);
    cuda_push_array(l.output_gpu, l.output, l.batch*l.outputs);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.inputs);
}

void backward_detection_layer_gpu(detection_layer l, network net)
{
    axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
    //copy_gpu(l.batch*l.inputs, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif

