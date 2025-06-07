%preprocess2

clear
clc

% Get all file names
files = dir('processed1/*.mat');
num_files = length(files);

% Loop over all datasets
for file=1:length(files)
    
    % Load the data. Data comes in a nx4 matrix with the 4 columns
    % corresponding to:
    % 1: subject number (1:n)
    % 2: mean accuracy for each subject
    % 3: mean confidence for each subject
    % 4: number of trials for each subject
    load(fullfile(files(file).folder, files(file).name));
    
    % Reverse accuracy for cases where response was on a continuous scale
    % and higher values actually mean lower accuracy (note: Kreis_2019 was
    % also a continuous scale but low conf values indicated high certainty)
    % 6:9: Akdogan_2017_Exp1-4
    % 25:29: Duyan_2018_Exp1-2, Duyan_2019, & Duyan_unpub_Exp1-2
    % 133: Rausch_2014
    % 150:151: Samaha_2017_exp1-2
    if any(file==[6:9, 25:29, 133, 150:151])            
        data(:,2) = -data(:,2);
    end
    
    % Save data with an improved name
    if strcmp(files(file).name(end-6:end), 'csv.mat')
        file_name = files(file).name(6:end-8);
    elseif strcmp(files(file).name, 'data_Wierzchon_2012.csv_C10.mat')
        file_name = 'Wierzchon_2012_C10';
    else
        file_name = [files(file).name(6:end-11), files(file).name(end-6:end-4)];
    end
    %save(fullfile(pwd, 'processed2', file_name), 'data');
end