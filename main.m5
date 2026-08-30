%% ============================================================
% GENUINE / IMPOSTOR SCORE GENERATION
% ============================================================

fprintf('\n========== GENUINE / IMPOSTOR EVALUATION ==========\n');

numPersons = 20;

genuineScores = [];
impostorScores = [];

% Normalize templates
faceTemplatesNorm = zeros(size(faceTemplates));
irisTemplatesNorm = zeros(size(irisTemplates));

for p = 1:numPersons
    faceTemplatesNorm(p,:) = ...
        faceTemplates(p,:) ./ ...
        (norm(faceTemplates(p,:)) + eps);

    irisTemplatesNorm(p,:) = ...
        irisTemplates(p,:) ./ ...
        (norm(irisTemplates(p,:)) + eps);
end


%% Compare every test sample against every identity

for i = 1:length(faceTestLabels)

    person = faceTestLabels(i);

    faceQuery = faceFeaturesTest(i,:);
    faceQuery = faceQuery ./ (norm(faceQuery) + eps);

    % Corresponding iris samples:
    % use iris samples belonging to same person
    irisIdx = find(irisTestLabels == person);

    for j = 1:length(irisIdx)

        irisQuery = irisFeaturesTest(irisIdx(j),:);
        irisQuery = irisQuery ./ (norm(irisQuery) + eps);

        for candidate = 1:numPersons

            faceScore = sum( ...
                faceQuery .* faceTemplatesNorm(candidate,:));

            irisScore = sum( ...
                irisQuery .* irisTemplatesNorm(candidate,:));

            % Score-level fusion
            fusedScore = ...
                0.5 * faceScore + ...
                0.5 * irisScore;

            % Genuine or impostor
            if candidate == person
                genuineScores(end+1) = fusedScore;
            else
                impostorScores(end+1) = fusedScore;
            end

        end

    end

end


%% Display score statistics

fprintf('\nGenuine comparisons : %d\n', ...
    length(genuineScores));

fprintf('Impostor comparisons: %d\n', ...
    length(impostorScores));

fprintf('\nMean Genuine Score  : %.4f\n', ...
    mean(genuineScores));

fprintf('Mean Impostor Score : %.4f\n', ...
    mean(impostorScores));

fprintf('===============================================\n');
