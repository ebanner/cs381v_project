% Load images
img1 = imread('demo1.jpg');
img2 = imread('demo2.jpg');

% GIST Parameters:
clear param
param.imageSize = [256 256]; % it works also with non-square images (use the most common aspect ratio in your set)
param.orientationsPerScale = [8 8 8 8]; % number of orientations per scale
param.numberBlocks = 4;
param.fc_prefilt = 4;

% Computing gist:
gist1 = LMgist(img1, '', param);
gist2 = LMgist(img2, '', param);

% Distance between the two images:
D = sum((gist1-gist2).^2)