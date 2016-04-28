function [ average_gist, all_gists ] = averageGist(image_paths)
%Computes the average GIST descriptor for all of the given images.
% Args:
%   image_paths: a 1 x N cell array of image file paths to be loaded.
%
% Returns:
%   average_gist: the average GIST discriptor for this class.
%   all_gists: an N x 512 matrix, where each row is a GIST descriptor for that
%       image.

    % Create LMgist function parameters object.
    param.imageSize = [224 224];
    param.orientationsPerScale = [8 8 8 8];
    param.numberBlocks = 4;
    param.fc_prefilt = 4;
  
    num_images = size(image_paths, 2);
    all_gists = zeros(num_images, 512);

    for i = 1 : num_images
        img = imread(image_paths{i});
        gist = LMgist(img, '', param);
        all_gists(i, :) = gist;
    end

    average_gist = mean(all_gists);

end

