function [ average_gist ] = averageGist(image_paths)
%Computes the average GIST descriptor for all of the given images.

    % Create LMgist function parameters object.
    param.imageSize = [224 224];
    param.orientationsPerScale = [8 8 8 8];
    param.numberBlocks = 4;
    param.fc_prefilt = 4;
  
    num_images = size(image_paths, 2);
    average_gist = zeros(1, 512);

    for i = 1 : num_images
        img = imread(image_paths{i});
        gist = LMgist(img, '', param);
        average_gist = average_gist + gist;
    end

    average_gist = average_gist / num_images;

end

