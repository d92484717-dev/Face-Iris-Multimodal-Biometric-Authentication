%% ============================================================
% ACCURACY, FAR, FRR, ROC AND EER
% ============================================================

fprintf('\n========== FINAL PERFORMANCE EVALUATION ==========\n');

% All scores
allScores = [genuineScores(:); impostorScores(:)];

% Genuine = 1, Impostor = 0
trueLabels = [
    ones(length(genuineScores),1);
    zeros(length(impostorScores),1)
];

%% ROC Curve

[FAR_ROC, TAR_ROC, thresholds, AUC] = perfcurve( ...
    trueLabels, allScores, 1);

FRR_ROC = 1 - TAR_ROC;


%% Find EER

[~, eerIndex] = min(abs(FAR_ROC - FRR_ROC));

EER = mean([
    FAR_ROC(eerIndex), ...
    FRR_ROC(eerIndex)
]);

EER_threshold = thresholds(eerIndex);


%% Find best threshold using accuracy

candidateThresholds = linspace( ...
    min(allScores), ...
    max(allScores), ...
    1000);

bestAccuracy = 0;
bestThreshold = 0;
bestFAR = 0;
bestFRR = 0;

for t = candidateThresholds

    genuineAccepted = sum(genuineScores >= t);
    genuineRejected = sum(genuineScores < t);

    impostorAccepted = sum(impostorScores >= t);
    impostorRejected = sum(impostorScores < t);

    TP = genuineAccepted;
    TN = impostorRejected;

    total = ...
        length(genuineScores) + ...
        length(impostorScores);

    accuracy = (TP + TN) / total;

    FAR = impostorAccepted / ...
        length(impostorScores);

    FRR = genuineRejected / ...
        length(genuineScores);

    if accuracy > bestAccuracy

        bestAccuracy = accuracy;
        bestThreshold = t;
        bestFAR = FAR;
        bestFRR = FRR;

    end

end


%% Display final results

fprintf('\n--------------------------------------------\n');
fprintf('BEST THRESHOLD : %.4f\n', bestThreshold);
fprintf('ACCURACY       : %.2f %%\n', bestAccuracy*100);
fprintf('FAR            : %.2f %%\n', bestFAR*100);
fprintf('FRR            : %.2f %%\n', bestFRR*100);
fprintf('EER            : %.2f %%\n', EER*100);
fprintf('EER THRESHOLD  : %.4f\n', EER_threshold);
fprintf('AUC            : %.4f\n', AUC);
fprintf('--------------------------------------------\n');


%% ROC Plot

figure;

plot(FAR_ROC, TAR_ROC, 'LineWidth', 2);

xlabel('False Acceptance Rate (FAR)');
ylabel('True Acceptance Rate (TAR)');

title('Face + Iris Multimodal Authentication ROC');

grid on;


%% FAR-FRR Plot

figure;

plot(thresholds, FAR_ROC, 'LineWidth', 2);
hold on;

plot(thresholds, FRR_ROC, 'LineWidth', 2);

xlabel('Threshold');
ylabel('Rate');

title('FAR and FRR vs Threshold');

legend('FAR','FRR');

grid on;
