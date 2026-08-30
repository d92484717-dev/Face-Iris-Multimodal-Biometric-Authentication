%% ============================================================
% FACE + IRIS MULTIMODAL TEMPLATE CREATION
% ============================================================

fprintf('\n========== CREATING MULTIMODAL TEMPLATES ==========\n');

numPersons = 20;

% Feature dimensions
faceDim = size(faceFeaturesTrain,2);
irisDim = size(irisFeaturesTrain,2);

% Template storage
faceTemplates = zeros(numPersons, faceDim);
irisTemplates = zeros(numPersons, irisDim);

%% Create one Face template per person

for p = 1:numPersons

    idx = (faceTrainLabels == p);

    faceTemplates(p,:) = mean( ...
        faceFeaturesTrain(idx,:), 1);

end


%% Create one Iris template per person

for p = 1:numPersons

    idx = (irisTrainLabels == p);

    irisTemplates(p,:) = mean( ...
        irisFeaturesTrain(idx,:), 1);

end


fprintf('Face templates: %d x %d\n', ...
    size(faceTemplates,1), ...
    size(faceTemplates,2));

fprintf('Iris templates: %d x %d\n', ...
    size(irisTemplates,1), ...
    size(irisTemplates,2));

fprintf('===============================================\n');
