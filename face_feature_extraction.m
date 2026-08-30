%% ============================================================
% FACE FEATURE EXTRACTION - MobileNetV2
% ============================================================

clc;

faceFeaturesTrain = [];
faceFeaturesTest = [];

fprintf('========== FACE FEATURE EXTRACTION ==========\n');

%% Training images

for i = 1:length(faceTrainFiles)

    img = imread(faceTrainFiles{i});

    % Convert grayscale to RGB
    if size(img,3) == 1
        img = cat(3,img,img,img);
    end

    % Resize to MobileNetV2 input size
    img = imresize(img,[224 224]);

    % Extract feature
    feature = activations( ...
        net, ...
        img, ...
        'global_average_pooling2d_1', ...
        'OutputAs','rows');

    faceFeaturesTrain(i,:) = feature;

    if mod(i,10) == 0
        fprintf('Face training: %d / %d\n', ...
            i,length(faceTrainFiles));
    end
end


%% Testing images

for i = 1:length(faceTestFiles)

    img = imread(faceTestFiles{i});

    % Convert grayscale to RGB
    if size(img,3) == 1
        img = cat(3,img,img,img);
    end

    % Resize
    img = imresize(img,[224 224]);

    % Extract feature
    feature = activations( ...
        net, ...
        img, ...
        'global_average_pooling2d_1', ...
        'OutputAs','rows');

    faceFeaturesTest(i,:) = feature;

    if mod(i,10) == 0
        fprintf('Face testing: %d / %d\n', ...
            i,length(faceTestFiles));
    end
end


%% Display feature size

fprintf('\n========================================\n');
fprintf('Face Train Features: %d x %d\n', ...
    size(faceFeaturesTrain,1), ...
    size(faceFeaturesTrain,2));

fprintf('Face Test Features : %d x %d\n', ...
    size(faceFeaturesTest,1), ...
    size(faceFeaturesTest,2));

fprintf('========================================\n');
