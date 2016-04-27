function [ affinity_matrix ] = computeSimilarities(images, classes)
% Compute similarities
%
% Args:
%   img_matrix

  % Create LMgist function parameters object.
  param.imageSize = [224 224];
  param.orientationsPerScale = [8 8 8 8];
  param.numberBlocks = 4;
  param.fc_prefilt = 4;

  for i = 1 : num_images
    img = imread(images{i});
    gist = LMgist(img, '', param);
    % TODO: add to GIST sum for the associated class
  end

  % Take average GIST descriptor for each class.

  % Compute similarity matrix.
  affinity_matrix = 0;

end
