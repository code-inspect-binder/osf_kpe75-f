%exportToExcel

clear
clc
close all

% Get all file names
files = dir('processed2/*.mat');
num_files = length(files);

% Loop over all datasets
for study_num=1:length(files)
    
    % Load the data. Data comes in a nx4 matrix with the 4 columns
    % corresponding to:
    % 1: subject number (1:n)
    % 2: mean accuracy for each subject
    % 3: mean confidence for each subject
    % 4: number of trials for each subject
    load(fullfile(files(study_num).folder, files(study_num).name));
    dataset_name{study_num} = files(study_num).name(1:end-4);
    
    % Compute basic quantities
    corr_coef(study_num) = corr(data(:,2), data(:,3));
    num_sub(study_num) = size(data,1);
    mean_trial_num(study_num) = mean(data(:,4));   
end

% Export data to Excel
data_to_export = [corr_coef; num_sub; mean_trial_num]';
T_names = cell2table(dataset_name', 'VariableNames', {'Name'});
T_data = array2table(data_to_export, 'VariableNames',{'R_value','Num_subjects','Mean_trials_per_subject'}); %,'SD_accuracy'});
T = [T_names, T_data];
%writetable(T, 'data216_matlabPortion.xlsx');