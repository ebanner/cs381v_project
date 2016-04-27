function [ affinity_matrix ] = computeSimilarities(image_paths)
% Compute GIST visual similarities.
%
% Args:
%   image_paths: an M by N cell of image paths, where there are M classes
%       and for each class there are N image paths.
%
% Returns:
%   affinity_matrix: the M by M affinity similarity matrix.

    % Compute the average GIST descriptor for each class.
    num_classes = size(image_paths, 1);
    average_gists = cell(1, num_classes);
    for i = 1 : num_classes
        average_gists{i} = averageGist(image_paths(i, :));
    end

    % Compute similarity affinity matrix.
    affinity_matrix = zeros(num_classes, num_classes);
    for row = 1 : num_classes
        g1 = average_gists{row};
        for col = 1 : num_classes
            g2 = average_gists{col};
            affinity_matrix(row, col) = 1 - sum((g1-g2).^2);
        end
    end
    
end
