# neon_内置函数总结 

## 初始化寄存器
```c
vcreate_type: 将一个64bit的数据装入vector中，并返回元素类型为type的vector。r=a
vdup_n_type/vmov_n_type: 用类型为type的数值，初始化一个元素类型为type的新vector的所有元素。ri=a
vdupq_n_type/vmovq_n_type: 128位寄存器
vdup_lane_type: 用元素类型为type的vector的某个元素，初始化一个元素类型为type的新vector的所有元素。ri=a[b]
vdupq_lane_type:
vmovl_type: 将vector的元素bit位扩大到原来的两倍，元素值不变。
vmovn_type: 用旧vector创建一个新vector，新vector的元素bit位是旧vector的一半。新vector元素只保留旧vector元素的低半部分。
vqmovn_type: 用旧vector创建一个新vector，新vector的元素bit位是旧vector的一半。如果旧vector元素的值超过新vector元素的最大值，则新vector元素就取最大值。否则新vector元素就等于旧vector元素的值。
vqmovun_type: 作用与vqmovn_type类似，但它输入的是有符号vector，输出的是无符号vector。

```


## 从内存加载数据进neon寄存器
```c
vld1_type: 按顺序将内存的数据装入neon寄存器，并返回元素类型为type格式的vector
vld1q_type: 128位
vld1_lane_type：用旧vector创建一个同类型的新vector，同时将新vector中指定元素的值改为内存中的值。
vld1q_lane_type:
vld1_dup_type：用type类型的内存中第一个值，初始化一个元素类型为type的新vector的所有元素。
vld1q_dup_type:
vld2_type: 按交叉顺序将内存的数据装入2个neon寄存器（内存第1个数据放入第1个neon寄存器的第1个通道，内存第2个数据放入第2个neon寄存器的第1个通道，内存第3个数据放入第1个neon寄存器的第2个通道，内存第4个数据放入第2个neon寄存器的第2个通道。。。）。并返回有两个vector的结构体
vld2q_type:
vld2_lane_type:
vld2q_lane_type:
vld2_dup_type: 用type类型的内存中第一个值，初始化第一个新vector的所有元素，用内存中第二个值，初始化第二个新vector的所有元素。
vld3_type: 交叉存放，本质上与vld2_type类似，只是这里装载3个neon寄存器
vld3q_type:
vld3_lane_type:
vld3q_lane_type:
vld3_dup_type: 本质上与vld2_dup_type类似
vld4_type: 交叉存放，本质上与vld2_type类似，只是这里装载4个neon寄存器
vld4q_type:
vld4_lane_type:
vld4q_lane_type:
vld4q_dup_type: 本质上与vld2_dup_type类似
```


## 从neon寄存器加载数据进内存
```c
vst1_type: 将元素类型为type格式的vector的所有元素装入内存
vst1q_type:
vst1_lane_type: 将元素类型为type格式的vector中指定的某个元素装入内存
vst1q_lane_type:
vst2_type: 交叉存放，vld2_type的逆过程
vst2q_type:
vst2_lane_type:
vst2q_lane_type:
vst3_type: 交叉存放，vld3_type的逆过程
vst3q_type:
vst3_lane_type:
vst3q_lane_type:
vst4_type: 交叉存放，vld4_type的逆过程
vst4q_type:
vst4_lane_type:
vst4q_lane_type:
```
## 直接获取neon寄存器某个通道的值
```c
vget_low_type: 获取128bit vector的低半部分元素，输出的是元素类型相同的64bit vector。
vget_high_type: 获取128bit vector的高半部分元素，输出的是元素类型相同的64bit vector。
vget_lane_type: 获取元素类型为type的vector中指定的某个元素值。
vgetq_lane_type:
```
## 直接设置neon寄存器某个通道的值
```c
vset_lane_type: 设置元素类型为type的vector中指定的某个元素的值，并返回新vector。
vsetq_lane_type:
```
## 寄存器数据重排
```c
vext_type: 取第2个输入vector的低n个元素放入新vector的高位，新vector剩下的元素取自第1个输入vector最高的几个元素(可实现vector内元素位置的移动)
vextq_type:
如：src1 = {1,2,3,4,5,6,7,8}
       src2 = {9,10,11,12,13,14,15,16}
       dst = vext_type(src1,src2,3)时，则dst = {4,5,6,7,8, 9,10,11}

vtbl1_type: 第二个vector是索引，根据索引去第一个vector（相当于数组）中搜索相应的元素，并输出新的vector，超过范围的索引返回的是0.
如：src1 = {1,2,3,4,5,6,7,8}
       src2 = {0,0,1,1,2,2,7,8}
       dst = vtbl1_u8(src1,src2)时，则dst = {1,1,2,2,3,3,8,0}

vtbl2_type: 数组长度扩大到2个vector
如：src.val[0] = {1,2,3,4,5,6,7,8}
       src.val[1] = {9,10,11,12,13,14,15,16}
       src2 = {0,0,1,1,2,2,8,10}
       dst = vtbl2_u8(src,src2)时，则dst = {1,1,2,2,3,3,9,11}

vtbl3_type:
vtbl4_type:
vtbx1_type: 根vtbl1_type功能一样，不过搜索到的元素是用来替换第一个vector中的元素，并输出替换后的新vector，当索引超出范围时，则不替换第一个vector中相应的元素。
vtbx2_type:
vtbx3_type:
vtbx4_type:
vrev16_type: 将vector中的元素位置反转

vrev16q_type:
如：src1 = {1,2,3,4,5,6,7,8}
       dst = vrev16_u8(src1)时，则dst = {2,1,4,3,6,5,8,7}

vrev32_type:

vrev32q_type:
如：src1 = {1,2,3,4,5,6,7,8}
       dst = vrev32_u8(src1)时，则dst = {4,3,2,1,8,7,6,5}
vrev64_type:

vrev64q_type:
如：src1 = {1,2,3,4,5,6,7,8}
       dst = vrev32_u8(src1)时，则dst = {8,7,6,5,4,3,2,1}

vtrn_type: 将两个输入vector的元素通过转置生成一个有两个vector的矩阵
vtrnq_type:
如：src.val[0] = {1,2,3,4,5,6,7,8}
       src.val[1] = {9,10,11,12,13,14,15,16}
       dst = vtrn_u8(src.val[0], src.val[1])时，
       则 dst.val[0] = {1,9, 3,11,5,13,7,15}
           dst.val[1] = {2,10,4,12,6,14,8,16}

vzip_type: 将两个输入vector的元素通过交叉生成一个有两个vector的矩阵
vzipq_type:
如：src.val[0] = {1,2,3,4,5,6,7,8}
       src.val[1] = {9,10,11,12,13,14,15,16}
       dst = vzip_u8(src.val[0], src.val[1])时，
       则dst.val[0] = {1,9, 2,10,3,11,4,12}
           dst.val[1] = {5,13,6,14,7,15,8,16}

vuzp_type: 将两个输入vector的元素通过反交叉生成一个有两个vector的矩阵（通过这个可实现n-way 交织）
vuzpq_type:
如：src.val[0] = {1,2,3,4,5,6,7,8}
       src.val[1] = {9,10,11,12,13,14,15,16}
       dst = vuzp_u8(src.val[0], src.val[1])时，
       则dst.val[0] = {1,3,5,7,9, 11,13,15}
           dst.val[1] = {2,4,6,8,10,12,14,16}

vcombine_type: 将两个元素类型相同的输入vector拼接成一个同类型但大小是输入vector两倍的新vector。新vector中低部分元素存放的是第一个输入vector元素。
vbsl_type:按位选择，参数为(mask, src1, src2)。mask的某个bit为1，则选择src1中对应的bit，为0，则选择src2中对应的bit。
vbslq_type:
```

## 加法
```c
vadd_type: ri = ai + bi
vaddq_type:
vaddl_type: 变长加法运算，为了防止溢出
vaddw_type: 第一个vector元素宽度大于第二个vector元素
vaddhn_type: 结果vector元素的类型大小是输入vector元素的一半
vqadd_type: ri = sat(ai + bi) 饱和指令，相加结果超出元素的最大值时，元素就取最大值。
vqaddq_type:
vhadd_type: 相加结果再除2。ri = (ai + bi) >> 1;
vhaddq_type:
vrhadd_type: 相加结果再除2(四舍五入)。ri = (ai + bi + 1) >> 1
vrhaddq_type:
vpadd_type: r0 = a0 + a1, ..., r3 = a6 + a7, r4 = b0 + b1, ..., r7 = b6 + b7
vpaddl_type: r0 = a0 + a1, ..., r3 = a6 + a7;
vpaddlq_type:
vpadal_type: r0 = a0 + (b0 + b1), ..., r3 = a3 + (b6 + b7);
```
## 减法
```c
vsub_type: ri = ai - bi
vsubq_type:
vsubl_type:
vsubw_type:
vsubhn_type:
vqsub_type: 饱和指令 ri = sat(ai - bi)
vqsubq_type:
vhsub_type: 相减结果再除2。ri = (ai - bi) >> 1
vhsubq_type:
vrsubhn_type: 相减结果再除2(四舍五入)。ri = (ai - bi + 1) >> 1
```
## 乘法
```c
vmul_type: ri = ai * bi
vmulq_type:
vmul_n_type: ri = ai * b
vmulq_n_type:
vmul_lane_type: ri = ai * b[c]
vmulq_lane_type:
vmull_type: 变长乘法运算，为了防止溢出
vmull_n_type:
vmull_lane_type:
vqdmull_type: 变长乘法运算，参与运算的值是有符号数（所以可能溢出）,当结果溢出时，取饱和值
vqdmull_n_type:
vqdmull_lane_type:
vqdmulh_type:
vqdmulhq_type:
vqdmulh_n_type:
vqdmulhq_n_type:
vqdmulh_lane_type:
vqdmulhq_lane_type:
vqrdmulh_type:
vqrdmulhq_type:
vqrdmulh_n_type:
vqrdmulhq_n_type:
vqrdmulh_lane_type:
vqrdmulhq_lane_type:
```
## 乘加组合运算
```c
vmla_type: ri = ai + bi * ci
vmlaq_type:
vmla_n_type: ri = ai + bi * c
vmlaq_n_type:
vmla_lane_type: ri = ai + bi * c[d]
vmlaq_lane_type:
vmlal_type: 长指令 ri = ai + bi * ci
vmlal_n_type:
vmlal_lane_type:
vfma_f32：ri = ai + bi * ci 在加法之前，bi、ci相乘的结果不会被四舍五入
vqdmlal_type: ri = sat(ai + bi * ci)  bi/ci的元素大小是ai的一半
vqdmlal_n_type: ri = sat(ai + bi * c)
vqdmlal_lane_type: ri = sat(ai + bi * c[d])
```
## 乘减组合运算
```c
vmls_type: ri = ai - bi * ci
vmlsq_type:
vmls_n_type: ri = ai - bi * c
vmlsq_n_type:
vmls_lane_type: ri = ai - bi * c[d]
vmlsq_lane_type:
vmlsl_type: 长指令 ri = ai - bi * ci
vmlsl_n_type:
vmlsl_lane_type:
vfms_f32：ri = ai - bi * ci 在减法之前，bi、ci相乘的结果不会被四舍五入
vqdmlsl_type: ri = sat(ai - bi * ci） bi/ci的元素大小是ai的一半
vqdmlsl_n_type: ri = sat(ai - bi * c）
vqdmlsl_lane_type: ri = sat(ai - bi * c[d]）
```
## 取整
```c
vrndn_f32: to nearest, ties to even
vrndqn_f32:
vrnda_f32: to nearest, ties away from zero
vrndqa_f32:
vrndp_f32: towards +Inf
vrndqp_f32:
vrndm_f32: towards -Inf
vrndqm_f32:
vrnd_f32: towards 0
vrnqd_f32:
```
## 比较运算
```c
（结果为true，则所有的bit位被设置为1）

vceq_type: ri = ai == bi ? 1...1 : 0...0
vceqq_type:
vcge_type: ri = ai >= bi ? 1...1:0...0
vcgeq_type:
vcle_type: ri = ai <= bi ? 1...1:0...0
vcleq_type:
vcgt_type: ri = ai > bi ? 1...1:0...0
vcgtq_type:
vclt_type: ri = ai < bi ? 1...1:0...0
vcltq_type:
vcage_f32: ri = |ai| >= |bi| ? 1...1:0...0
vcageq_f32:
vcale_f32: ri = |ai| <= |bi| ? 1...1:0...0
vcaleq_f32:
vcagt_f32: ri = |ai| > |bi| ? 1...1:0...0
vcagtq_f32:
vcalt_f32: ri = |ai| < |bi| ? 1...1:0...0
vcaltq_f32:
vtst_type: ri = (ai & bi != 0) ? 1...1:0...0 
vtstq_type:
```
## 绝对值
```c
vabs_type: ri = |ai|
vabsq_type:
vqabs_type: ri = sat(|ai|)
vqabsq_type:
vabd_type: ri = |ai - bi|
vabdq_type:
vabdl_type: 长指令
vaba_type: ri = ai + |bi - ci|
vabaq_type:
vabal_type: 长指令
```
## 取最大最小值
```c
vmax_type: ri = ai >= bi ? ai : bi
vmaxq_type:
vpmax_type: r0 = a0 >= a1 ? a0 : a1, ..., r4 = b0 >= b1 ? b0 : b1, ...
vmin_type: ri = ai <= bi ? ai : bi
vminq_type:
vpmin_type: r0 = a0 <= a1 ? a0 : a1, ..., r4 = b0 <= b1 ? b0 : b1, ...
```c
## 倒数
```c
vrecpe_type: 求近似倒数，type是f32或者u32
vrecpeq_type:
vrecps_f32：(牛顿 - 拉夫逊迭代)
vrecpsq_f32
注：vrecpe_type计算倒数能保证千分之一左右的精度，如1.0的倒数为0.998047。执行完如下语句后能提高百万分之一精度
float32x4_t recip = vrecpeq_f32(src);此时能达到千分之一左右的精度，如1.0的倒数为0.998047
recip = vmulq_f32 (vrecpsq_f32 (src, rec), rec);执行后能达到百万分之一左右的精度，如1.0的倒数为0.999996
recip = vmulq_f32 (vrecpsq_f32 (src, rec), rec);再次执行后能基本能达到完全精度，如1.0的倒数为1.000000
```
## 平方根倒数
```c
vrsqrte_type: 计算输入值的平方根的倒数，type是f32或者u32。输入值不能是负数，否则计算出来的值是nan。
vrsqrteq_type:
vrsqrts_f32
vrsqrtsq_f32
```
## 移位运算
```c
vshl_type: ri = ai << bi 如果bi是负数，则变成右移
vshlq_type:
vshl_n_type: ri = ai << b 这里b是常数，如果传入的不是常数（即在编译的时候就要知道b的值），编译时会报错
vshlq_n_type:
vqshl_type: ri = sat(ai << bi)
vqshlq_type:
vrshl_type: ri = round(ai << bi)
vrshlq_type:
vqrshl_type: ri = sat&round(ai << bi)
vqrshlq_type:
vqshl_n_type: ri = sat(ai << b)
vqshlq_n_type:
vqshlu_n_type: ri = ai << b 输入vector是有符号，输出vector是无符号
vqshluq_n_type:
vshll_n_type:

vshr_n_type: ri = ai >> b
vshrq_n_type:
vrshr_n_type: ri = round(ai >> b)
vrshrq_n_type:
vsra_n_type: ri = (ai >> c) + (bi >> c)
vsraq_n_type:
vrsra_n_type: ri = round((ai >> c) + (bi >> c))
vrsraq_n_type:
vshrn_n_type:  窄指令ri = ai >> b
vqshrun_n_type:
vqrshrun_n_type:
vqshrn_n_type:
vrshrn_n_type:
vqrshrn_n_type:

vsri_n_type:
vsriq_n_type:
vsli_n_type:
vsliq_n_type:
```
## 取负
```c
vneg_type: ri = -ai
vnegq_type:
vqneg_type: ri = sat(-ai)
vqnegq_type:
```
## 按位运算
```c
vmvn_type: ri = ~ai
vmvnq_type:
vand_type: ri = ai & bi
vandq_type:
vorr_type: ri = ai | bi
vorrq_type:
veor_type: ri = ai ^ bi
veorq_type:
vbic_type: ri = ~ai & bi
vbicq_type:
vorn_type: ri = ai | (~bi)
vornq_type:
```
## 统计
```c
vcls_type:
vclz_type:
vcnt_type: 统计向量每个元素有多少bit位是1
vcntq_type:
```
## 数据类型转换
```c
vcvt_type1_type2: f32、u32、s32之间的转换。在f32转到u32时，是向下取整，且如果是负数，则转换后为0
vcvtq_type1_type2:
vcvt_n_type1_type2:
vcvtq_n_type1_type2:
vreinterpret_type1_type2: 将元素类型为type2的vector转换为元素类型为type1的vector。数据重新解析
vreinterpretq_type1_type2:
```
