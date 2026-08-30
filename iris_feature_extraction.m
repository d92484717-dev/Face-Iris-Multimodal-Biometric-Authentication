%% ============================================================
% IRIS FEATURE EXTRACTION - MobileNetV2
% ============================================================

irisFeaturesTrain = [];
irisFeaturesTest = [];

fprintf('\n========== IRIS FEATURE EXTRACTION ==========\n');

%% Training images

for i = 1:length(irisTrainFiles)

    img = imread(irisTrainFiles{i});

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

    irisFeaturesTrain(i,:) = feature;

    if mod(i,10) == 0
        fprintf('Iris training: %d / %d\n', ...
            i,length(irisTrainFiles));
    end
end


%% Testing images

for i = 1:length(irisTestFiles)

    img = imread(irisTestFiles{i});

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

    irisFeaturesTest(i,:) = feature;

    if mod(i,10) == 0
        fprintf('Iris testing: %d / %d\n', ...
            i,length(irisTestFiles));
    end
end


%% Display feature size

fprintf('\n========================================\n');
fprintf('Iris Train Features: %d x %d\n', ...
    size(irisFeaturesTrain,1), ...
    size(irisFeaturesTrain,2));

fprintf('Iris Test Features : %d x %d\n', ...
    size(irisFeaturesTest,1), ...
    size(irisFeaturesTest,2));

fprintf('========================================\n');
