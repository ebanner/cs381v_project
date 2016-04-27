function [] = saveAffinityMatrix(...
    train_file_path, affinity_mat_path, num_classes, num_images_per_class)
% Collects the affinity matrices.

    % Read the image paths from the file.
    image_paths = cell(num_classes, num_images_per_class);
    i = 1;
    j = 1;
    fid = fopen(train_file_path);
    fline = fgets(fid);
    while ischar(fline)
        fline = fline(1:end-1);
        if size(fline, 2) > 0 && ~strcmp(fline(1), '#')
            parts = strsplit(fline);
            image_paths{i,j} = parts{1};
            %disp(['fline = ' parts{1}]);
            %disp(['      = ' parts{2}]);
            j = j + 1;
            if j > num_images_per_class
                i = i + 1;
                j = 1;
            end
        end
        fline = fgets(fid);
    end
    fclose(fid);

    % Get the affinity matrix.
    affinity_matrix = computeSimilarities(image_paths);

    % Save the affinity matrix.
    dlmwrite(affinity_mat_path, affinity_matrix, ' ');
    
end