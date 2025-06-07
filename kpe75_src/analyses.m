%analyses

clear
clc
close all

% Add helper functions
addpath(genpath(fullfile(pwd, 'helperFunctions')));

% Read table with all relevant data
T = readtable('data.xlsx');
head(T)


%% Display relevant results
% Summary info
total_subjects = sum(T.Num_subjects)
total_trials = sum(T.Num_subjects .* T.Mean_trials_per_subject)
average_number_trials = mean(T.Mean_trials_per_subject)

% Average correlation coefficient (results from meta-analyses in R)
mean_meta_95CI_all = z2r([.2226, .1801, .2651]) %mean + 95CI; input taken from R
mean_meta_95CI_N20 = z2r([.2062, .1553, .2571]) %mean + 95CI; input taken from R

% One-sample t-test (no meta-analysis)
mean_nonMeta = mean(z2r(T.Z_value))
perform_ttest(T.Z_value, 'One-sample test, all studies', 1);


%% Plot figures
% FIGURE 1
plot_fig1(T)

% FIGURE 2
figure('Color','w', 'DefaultAxesFontSize',16);
colormap jet
scatter(T.R_value, T.Num_subjects, [], T.Mean_trials_per_subject, 'filled');
xlabel('R-value', 'FontSize',24);
ylabel('Sample size', 'FontSize',24);
h = colorbar;
set(get(h,'label'),'string','Mean trials per participant');
box off

% FIGURE 3
figure('Color','w', 'DefaultAxesFontSize',20);
rng(sum(clock),'twister'); %reset random function for consistent plotting

% Confidence scale
plot_correlation(T.Confidence_scale, T.R_value)

% Domain
domain{1}=T.R_value(T.Perception==1);
domain{2}=T.R_value(T.Memory==1); 
domain{3}=T.R_value(T.Other==1); 
plot_consecutive_bars(domain, {'Perception','Memory','Other'}, {'p = 0.004','p = 0.32','p = 0.07'}, 3, 'Domain');

% Timing of confidence response
conf_timing{1}=T.R_value(T.Conf_simultaneous_with_decision==1);
conf_timing{2}=T.R_value(T.Conf_simultaneous_with_decision==0); 
plot_consecutive_bars(conf_timing, {'With decision','After decision'}, 'p = 0.37', 4, 'Timing of confidence judgment');

% Feedback
feedback{1}=T.R_value(T.Feedback_binary==1);
feedback{2}=T.R_value(T.Feedback_binary==0); 
plot_consecutive_bars(feedback, {'Yes','No'}, 'p = 0.23', 5, 'Trial-by-trial feedback');

% Timing of confidence response
discrim_estim{1}=T.R_value(T.Estimation_cetagorization==0); %Discrimination
discrim_estim{2}=T.R_value(T.Estimation_cetagorization==1); %Estimation
plot_consecutive_bars(discrim_estim, {'Discrimination','Estimation'}, 'p = 0.22', 6, 'Type of task');