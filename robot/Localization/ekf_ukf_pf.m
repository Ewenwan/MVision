%EKF UKF PF 的三个算法

clear;

%tic;

x =0.1; %初始状态

x_estimate=1;%状态的估计

e_x_estimate=x_estimate;%EKF的初始估计

u_x_estimate=x_estimate;%UKF的初始估计

p_x_estimate=x_estimate;%PF的初始估计

Q =10;%input('请输入过程噪声方差 Q 的值 :'); %过程状态协方差

R =1;%input('请输入测量噪声方差 R 的值 :'); %测量噪声协方差

P =5;%初始估计方差

e_P=P; %UKF方差

u_P=P;%UKF方差

pf_P=P;%PF方差

tf =50; %模拟长度

x_array=[x];%真实值数组

e_x_estimate_array=[e_x_estimate];%EKF最优估计值数组

u_x_estimate_array=[u_x_estimate];%UKF最优估计值数组

p_x_estimate_array=[p_x_estimate];%PF最优估计值数组

u_k=1; %微调参数

u_symmetry_number=4; %对称的点的个数

u_total_number=2*u_symmetry_number+1; %总的采样点的个数

linear =0.5;

N =500; %粒子滤波的粒子数

close all;

%粒子滤波初始 N 个粒子

for i =1:N

p_xpart(i)=p_x_estimate+sqrt(pf_P)*randn;

end

for k =1:tf

%模拟系统

x =linear *x +(25*x /(1+x^2))+8*cos(1.2*(k-1))+sqrt(Q)*randn; %状态值

y =(x^2/20) +sqrt(R)*randn; %观测值

%扩展卡尔曼滤波器

%进行估计 第一阶段的估计

e_x_estimate_1=linear *e_x_estimate+25*e_x_estimate/(1+e_x_estimate^2)+8*cos(1.2*(k-1));

e_y_estimate=(e_x_estimate_1)^2/20;%这是根据 k=1时估计值为 1得到的观测值;只是这个由 我估计得到的 第 24行的 y 也是观测值 不过是由加了噪声的真实值得到的

%相关矩阵

e_A=linear +25*(1-e_x_estimate^2)/((1+e_x_estimate^2)^2);%传递矩阵

e_H=e_x_estimate_1/10;%观测矩阵

%估计的误差

e_p_estimate=e_A*e_P*e_A'+Q;

%扩展卡尔曼增益

e_K=e_p_estimate*e_H'/(e_H*e_p_estimate*e_H'+R);

%进行估计值的更新 第二阶段

e_x_estimate_2=e_x_estimate_1+e_K*(y-e_y_estimate);

%更新后的估计值的误差

e_p_estimate_update=e_p_estimate-e_K*e_H*e_p_estimate;

%进入下一次迭代的参数变化

e_P=e_p_estimate_update;

e_x_estimate=e_x_estimate_2;

%粒子滤波器

%粒子滤波器

for i =1:N

p_xpartminus(i)=0.5*p_xpart(i)+25*p_xpart(i)/(1+p_xpart(i)^2)+8*cos(1.2*(k-1))+ sqrt(Q)*randn; %这个式子比下面一行的效果好

%xpartminus(i)=0.5*xpart(i)+25*xpart(i)/(1+xpart(i)^2)+8*cos(1.2*(k-1));

p_ypart=p_xpartminus(i)^2/20; %预测值

p_vhat=y -p_ypart;%观测和预测的差

p_q(i)=(1/sqrt(R)/sqrt(2*pi))*exp(-p_vhat^2/2/R); %各个粒子的权值

end

%平均每一个估计的可能性

p_qsum=sum(p_q);

for i =1:N

p_q(i)=p_q(i)/p_qsum;%各个粒子进行权值归一化

end

%重采样 权重大的粒子多采点,权重小的粒子少采点 , 相当于每一次都进行重采样;

for i =1:N

p_u=rand;

p_qtempsum=0;

for j =1:N

p_qtempsum=p_qtempsum+p_q(j);

if p_qtempsum>=p_u

p_xpart(i)=p_xpartminus(j);%在这里 xpart(i)实现循环赋值;终于找到了这里! ! ! break;

end

end

end

p_x_estimate=mean(p_xpart);

%p_x_estimate=0;

%for i =1:N

%p_x_estimate=p_x_estimate+p_q(i)*p_xpart(i);

%end

%不敏卡尔曼滤波器

%采样点的选取 存在 x(i)

u_x_par=u_x_estimate;

for i =2:(u_symmetry_number+1)

u_x_par(i,:)=u_x_estimate+sqrt((u_symmetry_number+u_k)*u_P);

end

for i =(u_symmetry_number+2):u_total_number

u_x_par(i,:)=u_x_estimate-sqrt((u_symmetry_number+u_k)*u_P);

end

%计算权值

u_w_1=u_k/(u_symmetry_number+u_k);

u_w_N1=1/(2*(u_symmetry_number+u_k));

%把这些粒子通过传递方程 得到下一个状态

for i =1:u_total_number
u_x_par(i)=0.5*u_x_par(i)+25*u_x_par(i)/(1+u_x_par(i)^2)+8*cos(1.2*(k-1));

end

%传递后的均值和方差

u_x_next=u_w_1*u_x_par(1);

for i =2:u_total_number

u_x_next=u_x_next+u_w_N1*u_x_par(i);

end

u_p_next=Q +u_w_1*(u_x_par(1)-u_x_next)*(u_x_par(1)-u_x_next)';

for i =2:u_total_number

u_p_next=u_p_next+u_w_N1*(u_x_par(i)-u_x_next)*(u_x_par(i)-u_x_next)';

end

%%对传递后的均值和方差进行采样 产生粒子 存在 y(i)

%u_y_2obser(1)=u_x_next;

%for i =2:(u_symmetry_number+1)

%u_y_2obser(i,:)=u_x_next+sqrt((u_symmetry_number+k)*u_p_next); %end

%for i =(u_symmetry_number+2) :u_total_number

%u_y_2obser(i,:)=u_x_next-sqrt((u_symmetry_number+u_k)*u_p_next); %end

%另外存在 y_2obser(i)中;

for i =1:u_total_number

u_y_2obser(i,:)=u_x_par(i);

end

%通过观测方程 得到一系列的粒子

for i =1:u_total_number

u_y_2obser(i)=u_y_2obser(i)^2/20;

end

%通过观测方程后的均值 y_obse

u_y_obse=u_w_1*u_y_2obser(1);

for i =2:u_total_number

u_y_obse=u_y_obse+u_w_N1*u_y_2obser(i);

end

%Pzz测量方差矩阵

u_pzz=R +u_w_1*(u_y_2obser(1)-u_y_obse)*(u_y_2obser(1)-u_y_obse)';

for i =2:u_total_number

u_pzz=u_pzz+u_w_N1*(u_y_2obser(i)-u_y_obse)*(u_y_2obser(i)-u_y_obse)';

end

%Pxz状态向量与测量值的协方差矩阵

u_pxz=u_w_1*(u_x_par(1)-u_x_next)*(u_y_2obser(1)-u_y_obse)';

for i =2:u_total_number

u_pxz=u_pxz+u_w_N1*(u_x_par(i)-u_x_next)*(u_y_2obser(i)-u_y_obse)';

end

%卡尔曼增益

u_K=u_pxz/u_pzz;

%估计量的更新

u_x_next_optimal=u_x_next+u_K*(y-u_y_obse);%第一步的估计值 +修正值; u_x_estimate=u_x_next_optimal;

%方差的更新

u_p_next_update=u_p_next-u_K*u_pzz*u_K';

u_P=u_p_next_update;

%进行画图程序

x_array=[x_array,x];

e_x_estimate_array=[e_x_estimate_array,e_x_estimate];

p_x_estimate_array=[p_x_estimate_array,p_x_estimate];

u_x_estimate_array=[u_x_estimate_array,u_x_estimate];

e_error(k,:)=abs(x_array(k)-e_x_estimate_array(k));

p_error(k,:)=abs(x_array(k)-p_x_estimate_array(k));

u_error(k,:)=abs(x_array(k)-u_x_estimate_array(k));

end

t =0:tf;

figure;

plot(t,x_array,'k.',t,e_x_estimate_array,'r-',t,p_x_estimate_array,'g--',t,u_x_estimate_array,'b:');

set(gca,'FontSize',10);

set(gcf,'color','White');

xlabel('时间步长 ');%lable --->label我的神

ylabel('状态 ');

legend('真实值 ','EKF 估计值 ','PF 估计值 ','UKF 估计值 ');

figure;

plot(t,x_array,'k.',t,p_x_estimate_array,'g--',t, p_x_estimate_array-1.96*sqrt(P),'r:',t, p_x_estimate_array+1.96*sqrt(P),'r:');

set(gca,'FontSize',10);

set(gcf,'color','White');

xlabel('时间步长 ');%lable --->label我的神

ylabel('状态 ');

legend('真实值 ','PF 估计值 ', '95%置信区间 ');

%rootmean square 平均值的平方根

e_xrms=sqrt((norm(x_array-e_x_estimate_array)^2)/tf);

disp(['EKF估计误差均方值 =',num2str(e_xrms)]);

p_xrms=sqrt((norm(x_array-p_x_estimate_array)^2)/tf);

disp(['PF估计误差均方值 =',num2str(p_xrms)]);

u_xrms=sqrt((norm(x_array-u_x_estimate_array)^2)/tf);

disp(['UKF估计误差均方值 =',num2str(u_xrms)]);

%plot(t,e_error,'r-',t,p_error,'g--',t,u_error,'b:');

%legend('EKF估计值误差 ','PF 估计值误差 ','UKF 估计值误差 ');

t =1:tf;

figure;

plot(t,e_error,'r-',t,p_error,'g--',t,u_error,'b:');

set(gca,'FontSize',10);

set(gcf,'color','White');

xlabel('时间步长 ');%lable --->label我的神

ylabel('状态 ');

legend('EKF估计值误差 ','PF 估计值误差 ','UKF 估计值误差 ');

%toc;
