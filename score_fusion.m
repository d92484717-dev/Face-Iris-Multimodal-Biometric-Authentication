%% ============================================================
% FACE + IRIS SCORE-LEVEL FUSION - CORRECTED
% ============================================================

fprintf('\n========== SCORE-LEVEL FUSION ==========\n');

numPersons = 20;

% Make sure all feature matrices are numeric
faceFeaturesTest  = double(faceFeaturesTest);
irisFeaturesTest  = double(irisFeaturesTest);
faceTemplates     = double(faceTemplates);
irisTemplates     = double(irisTemplates);

% Force every feature vector to be a ROW vector
faceTemplates = reshape(faceTemplates, numPersons, []);
irisTemplates = reshape(irisTemplates, numPersons, []);

facePersonScore = zeros(numPersons,1);
irisPersonScore = zeros(numPersons,1);

%% ---------- FACE SCORES ----------

for p = 1:numPersons

    idx = find(faceTestLabels == p);
    scores = zeros(length(idx),1);

    template = faceTemplates(p,:);
    template = template(:)';       % force row

    template = template ./ (norm(template) + eps);

    for j = 1:length(idx)

        query = faceFeaturesTest(idx(j),:);
        query = query(:)';         % force row

        query = query ./ (norm(query) + eps);

        % Cosine similarity
        scores(j) = sum(query .* template);

    end

    facePersonScore(p) = mean(scores);

end


%% ---------- IRIS SCORES ----------

for p = 1:numPersons

    idx = find(irisTestLabels == p);
    scores = zeros(length(idx),1);

    template = irisTemplates(p,:);
    template = template(:)';

    template = template ./ (norm(template) + eps);

    for j = 1:length(idx)

        query = irisFeaturesTest(idx(j),:);
        query = query(:)';

        query = query ./ (norm(query) + eps);

        % Cosine similarity
        scores(j) = sum(query .* template);

    end

    irisPersonScore(p) = mean(scores);

end


%% ---------- SCORE NORMALIZATION ----------

faceNorm = ...
    (facePersonScore - min(facePersonScore)) ./ ...
    (max(facePersonScore) - min(facePersonScore) + eps);

irisNorm = ...
    (irisPersonScore - min(irisPersonScore)) ./ ...
    (max(irisPersonScore) - min(irisPersonScore) + eps);


%% ---------- MULTIMODAL FUSION ----------

faceWeight = 0.5;
irisWeight = 0.5;

fusionScore = ...
    faceWeight .* faceNorm + ...
    irisWeight .* irisNorm;


%% ---------- DISPLAY RESULTS ----------

fprintf('\nPerson   FaceScore   IrisScore   FusionScore\n');
fprintf('--------------------------------------------\n');

for p = 1:numPersons

    fprintf('%-7d  %.4f      %.4f      %.4f\n', ...
        p, ...
        faceNorm(p), ...
        irisNorm(p), ...
        fusionScore(p));

end

fprintf('============================================\n');

fprintf('\nMean Face Score   : %.4f\n', mean(faceNorm));
fprintf('Mean Iris Score   : %.4f\n', mean(irisNorm));
fprintf('Mean Fusion Score : %.4f\n', mean(fusionScore));
