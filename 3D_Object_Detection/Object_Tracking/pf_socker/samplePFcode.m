%Sample particle filter code to get you started
%This is a very simple, no-frills implementation.
%Bob Collins,

% 粒子滤波跟踪

%As observations, we will use the ground truth bounding 
%box information provided with the VS-PETS soccer dataset
%to simulate a (very accurate) person detector.
%observations

% 目标边框数据 glound trouth
% www.cse.psu.edu/~rcollins/CSE598B/Datasets/soccerboxes.mat
% 视频数据
% www.cse.psu.edu/~rcollins/CSE598B/Datasets/Soccer.zip
% 背景
% www.cse.psu.edu/~rcollins/CSE598B/Datasets/soccerbgimage.mat
% 介绍
% http://www.cse.psu.edu/~rtc12/CSE598C/Chap2ParticlesDAF.pdf
load soccerboxes.mat

%each box is stored as one row of allboxes
%there are 6 columns
% col1 : frame number this box comes from
% col2 : object identifier this box comes from
% col3 : x coord (matlab col) of box center
% col4 : y coord (matlab row) of box center
% col5 : width of box
% col6 : height of box

%prepare sequence structure for genfilename.m
startframe = min(allboxes(:,1));
endframe = max(allboxes(:,1));
prefix = 'Soccer/Frame';
postfix = '.jpg';
sequence = struct('prefix',prefix,'postfix',postfix,'digits',4,'startframe',startframe,'endframe',endframe)

%initialize by choosing a subsequence and one person to track
fstart = startframe; 
fend = endframe;
fend = fstart+100;  %I just want to run for a 100 frames for now, to demonstrate
fnum = fstart;

%get image frame and draw it
fname = genfilename(sequence,fnum)
imrgb = imread(fname);
figure(1); imagesc(imrgb);

%find all boxes in frame number fnum and draw each one on image
inds = find(allboxes(:,1)==fnum);
hold on
for iii=1:length(inds)
   box = allboxes(inds(iii),:);
   objnum = box(2);
   col0 = box(3);
   row0 = box(4);
   dcol = box(5)/2.0;
   drow = box(6)/2.0;
   h = plot(col0+[-dcol dcol dcol -dcol -dcol],row0+[-drow -drow drow drow -drow],'y-');
   set(h,'LineWidth',2);
end
hold off
drawnow

%intialize prior by clicking mouse near center of person 
%you want to track
[x0,y0] = ginput(1);

%number of particles for particle filtering
nsamples = 100;
%prior distribution will be gaussian
priorsigmax = 10;
priorsigmay = 10;
%generate particles from prior distribution
sampx = x0 + priorsigmax*randn(1,nsamples);
sampy = y0 + priorsigmay*randn(1,nsamples);
weights = ones(1,nsamples)/nsamples;
%plot particles
figure(1); imagesc(imrgb); hold on
plot(sampx,sampy,'b.');
hold off; drawnow;

%now start tracking
deltaframe = 2;  %set to 1 for every frame
for fnum = (fstart+deltaframe): deltaframe : fend
    %get image frame and draw it
    fname = genfilename(sequence,fnum)
    imrgb = imread(fname);
    figure(1); imagesc(imrgb);
    %find all boxes in frame number fnum
    inds = find(allboxes(:,1)==fnum);
    
    %do motion prediction step of Bayes filtering 
    %we will use a deterministic motion model plus
    %additive gaussian noise.
    %we are using simple constant position model 
    %as a simple demonstration; it would be better
    %to use constant velocity.
    motpredsigmax = 10;
    motpredsigmay = 10;
    predx = sampx + motpredsigmax*randn(1,nsamples);
    predy = sampy + motpredsigmay*randn(1,nsamples);
    
    %compute weights based on likelihood
    %recall weights should be oldweight * likelihood
    %but all old weights are equal, so new weight will
    %just be the likelihood.
    %For measuring likelihood, we are using a mixture
    %model (parzen estimate) based on the locations of
    %the ground truth bounding boxes  Note that this is
    %a semiparametric, multimodal distribtion
    obssigmax = 5;
    obssigmay = 5;

    %there surely must be a more efficient way to do the
    %following as a vectorized computation rather than
    %a loop, but I want to just get it right the first time
    weights = ones(1,nsamples);
    for i=1:nsamples
        prob = 0;
        x = predx(i); y=predy(i);
        for iii=1:length(inds)
            box = allboxes(inds(iii),:);
            midx = box(3);  %centroid of box
            midy = box(4);
            dx = midx-x; dy = midy-y;
            p = exp(- 0.5 *(dx^2 / obssigmax^2 + dy^2 / obssigmay^2));
            prob = prob + p;      
        end   
        weights(i) = prob
    end
    
    %resample particles according to likelihood weights
    %the resulting samples will then have equal weight
    
    % 重采样==================================
    indices = resampindex(weights);
    sampx = predx(indices);
    sampy = predy(indices);
    %plot resampled particles
    %jitter with a little noise so multiple copies can be seen
    figure(1); imagesc(imrgb); hold on
    plot(sampx+1*randn(1,nsamples),sampy+1*randn(1,nsamples),'b.');
    drawnow
    
end






