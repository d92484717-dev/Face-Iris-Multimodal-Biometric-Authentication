
clc;
clear;
close all;

%% Dataset paths

facePath = '/MATLAB Drive/face&irs/face dataset';
irisPath = '/MATLAB Drive/face&irs/iris dataset';

numPersons = 20;

% Storage
faceTrainFiles = {};
faceTestFiles  = {};

irisTrainFiles = {};
irisTestFiles  = {};

faceTrainLabels = [];
faceTestLabels  = [];

irisTrainLabels = [];
irisTestLabels  = [];

%% Create train/test split

for p = 1:numPersons

    person = sprintf('person%d', p);

    % ---------- FACE ----------
    faceFolder = fullfile(facePath, person);

    faceFiles = [
        dir(fullfile(faceFolder, '*.jpg'));
        dir(fullfile(faceFolder, '*.jpeg'));
        dir(fullfile(faceFolder, '*.png'))
    ];

    % Sort files
    [~,idx] = sort({faceFiles.name});
    faceFiles = faceFiles(idx);

    % First 2 = training
    for k = 1:2
        faceTrainFiles{end+1} = ...
            fullfile(faceFolder, faceFiles(k).name);
        faceTrainLabels(end+1) = p;
    end

    % Remaining 2 = testing
    for k = 3:4
        faceTestFiles{end+1} = ...
            fullfile(faceFolder, faceFiles(k).name);
        faceTestLabels(end+1) = p;
    end


    % ---------- IRIS ----------
    irisFolder = fullfile(irisPath, person);

    irisFiles = [
        dir(fullfile(irisFolder, '*.jpg'));
        dir(fullfile(irisFolder, '*.jpeg'));
        dir(fullfile(irisFolder, '*.png'))
    ];

    [~,idx] = sort({irisFiles.name});
    irisFiles = irisFiles(idx);

    % First 4 = training
    for k = 1:4
        irisTrainFiles{end+1} = ...
            fullfile(irisFolder, irisFiles(k).name);
        irisTrainLabels(end+1) = p;
    end

    % Remaining 4 = testing
    for k = 5:8
        irisTestFiles{end+1} = ...
            fullfile(irisFolder, irisFiles(k).name);
        irisTestLabels(end+1) = p;
    end

end

%% Display results

fprintf('\n========================================\n');
fprintf('       TRAIN / TEST SPLIT\n');
fprintf('========================================\n');

fprintf('Persons              : %d\n', numPersons);

fprintf('Face training images : %d\n', length(faceTrainFiles));
fprintf('Face testing images  : %d\n', length(faceTestFiles));

fprintf('Iris training images : %d\n', length(irisTrainFiles));
fprintf('Iris testing images  : %d\n', length(irisTestFiles));

fprintf('========================================\n');
