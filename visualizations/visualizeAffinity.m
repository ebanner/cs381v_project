function [] = visualizeAffinity(affinity_matrix, classnames)
%Visualizes the given affinity matrix.
%
% Args:
%   affinity_matrix: the affinity matrix that will be visualized.
%   classnames: the name of the classes that will be displayed on the
%       matrix image.

    num_classes = size(classnames, 2);
    
    % Sanity check the matrix dimensions.
    [w, h] = size(affinity_matrix);
    if w ~= num_classes || h ~= num_classes
        fprintf('Fail: mismatched matrix dimensions!\n');
        return;
    end
    
    imagesc(affinity_matrix);
    axis square;
    
    % Set the Y labels.
    max_lim = num_classes + 0.5;
    set(gca, 'FontSize', 24);
    set(gca, 'YLim', [0.5 max_lim], 'YTick', 1:num_classes, ...
        'YTickLabel', classnames);
    % Set the X labels.
    set(gca, 'XLim', [0.5 max_lim], 'XTick', 1:num_classes, ...
        'XTickLabel', classnames, 'XTickLabelRotation', 45);
    
end

