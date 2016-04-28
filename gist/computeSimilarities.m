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
    all_gists = cell(1, num_classes);
    for i = 1 : num_classes
        [average_gist, all_gists_i] = averageGist(image_paths(i, :));
        average_gists{i} = average_gist;
        all_gists{i} = all_gists_i;
    end

    % Compute similarity affinity matrix.
    affinity_matrix = eye(num_classes, num_classes);
    for row = 1 : num_classes
        %avg_gist = average_gists{row};
        %avg_gist = avg_gist / norm(avg_gist);
        %for col = 1 : num_classes
        %    if row == col
        %        continue;
        %    end
        %    all_gists_col = all_gists{col};
        %    num_images = size(all_gists_col, 1);
        %    sum_dists = 0;
        %    for i = 1 : num_images
        %        gist_i = all_gists_col(i, :);
        %        gist_i = gist_i / norm(gist_i);
        %        dist = 1 - sqrt(sum((avg_gist - gist_i).^2));
        %        sum_dists = sum_dists + dist;
        %    end
        %    affinity_matrix(row, col) = sum_dists / num_images;
        %end
        g1 = average_gists{row};
        for col = 1 : num_classes
            %affinity_matrix(row, col) = 1 - sum((g1-g2).^2);
            g2 = average_gists{col};
            %dist =  1 - pdist2(g1, g2, 'cosine');
            %dist = (g1.^g2) / (sum(g1)*sum(g2));
            g1 = g1 / norm(g1);
            g2 = g2 / norm(g2);
            dist = dot(g1, g2);
            affinity_matrix(row, col) = dist;
        end
    end
    
end
