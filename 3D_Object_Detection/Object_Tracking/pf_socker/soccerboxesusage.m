%sample code to show how to read and interpret the ground truth boxes
%from the soccer sequence (VS-PETS2003).

%load in all boxes, it will be in variable allboxes
load soccerboxes.mat


%each box is stored as one row of allboxes
%there are 6 columns
% col1 : frame number this box comes from
% col2 : object identifier this box comes from
% col3 : x coord (matlab col) of box center
% col4 : y coord (matlab row) of box center
% col5 : width of box
% col6 : height of box


%===========================================================
%example usage: draw all boxes overlaid on frame number 10
startframe = min(allboxes(:,1));
endframe = max(allboxes(:,1));
prefix = 'Soccer/Frame';
postfix = '.jpg';
sequence = struct('prefix',prefix,'postfix',postfix,'digits',4,'startframe',startframe,'endframe',endframe)

%frame number
fnum = 10;

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


%===========================================================
%example usage: doing background subtraction on frame number 10

%load precomputed background image into bgimage
load soccerbgimage.mat
bgimage = double(bgimage);

prefix = 'Soccer/Frame';
postfix = '.jpg';
sequence = struct('prefix',prefix,'postfix',postfix,'digits',4,'startframe',startframe,'endframe',endframe)

%frame number
fnum = 10;

%get image frame and draw it
fname = genfilename(sequence,fnum)
imrgb = imread(fname);

%do background subtraction and thresholding
bgthresh = 30;
rgbabsdiff = abs(double(imrgb)-bgimage);
maxdiff = max(rgbabsdiff,[],3);  %max diff in red green or blue
bgmask = roicolor(maxdiff,bgthresh,Inf);

%display
figure(2); colormap(gray);
imagesc(bgmask);
drawnow

