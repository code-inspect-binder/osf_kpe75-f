function plot_fig1(T)

n_datasets = length(T.Name);
data{1} = T.Domain;
Num_subjects = T.Num_subjects;
Mean_trials_per_subject = T.Mean_trials_per_subject;
Conf_simultaneous_with_decision = T.Conf_simultaneous_with_decision;
Feedback = T.Feedback_binary;
Estimation_cetagorization = T.Estimation_cetagorization;
Confidence_scale = T.Confidence_scale;
titles = {'Domain', 'Number of participants', 'Number of trials per participant',...
    'Confidence in respect to decision', 'Trial-by-trial feedback', 'Task type', 'Confidence scale'};

% Transform values for plotting
for dataset_num=1:length(data{1})
    if ~strcmp(data{1}(dataset_num),'Perception') && ~strcmp(data{1}(dataset_num),'Memory')
        data{1}{dataset_num} = 'Other';
    end
    
    if Num_subjects(dataset_num) <= 20
        data{2}{dataset_num} = '4-20';
    elseif Num_subjects(dataset_num) <= 50
        data{2}{dataset_num} = '21-50';
    elseif Num_subjects(dataset_num) <= 100
        data{2}{dataset_num} = '51-100';
    else
        data{2}{dataset_num} = '101-589';
    end
    
    if Mean_trials_per_subject(dataset_num) <= 100
        data{3}{dataset_num} = '20-100';
    elseif Mean_trials_per_subject(dataset_num) <= 400
        data{3}{dataset_num} = '101-400';
    elseif Mean_trials_per_subject(dataset_num) <= 1000
        data{3}{dataset_num} = '401-800';
    else
        data{3}{dataset_num} = '801-5307';
    end
    
    if Conf_simultaneous_with_decision(dataset_num) == 1
        data{4}{dataset_num} = 'Simultaneous';
    else
        data{4}{dataset_num} = 'After';
    end
    
    if Feedback(dataset_num) == 1
        data{5}{dataset_num} = 'Yes';
    else
        data{5}{dataset_num} = 'No';
    end
    
    if Estimation_cetagorization(dataset_num) == 0
        data{6}{dataset_num} = 'Discrimination';
    else
        data{6}{dataset_num} = 'Estimation';
    end
    
    if Confidence_scale(dataset_num) == 100
        data{7}{dataset_num} = 'Continuous';
    elseif Confidence_scale(dataset_num) >= 5
        data{7}{dataset_num} = '5- to 11-point';
    else
        data{7}{dataset_num} = [num2str(Confidence_scale(dataset_num)) '-point'];
    end
end

% Pie chart for type of study
figure('Color','w', 'DefaultAxesFontSize',16);

% Loop over the different characteristics
for subpl=1:length(data)
    % Create a new subplot
    ax = subplot(2,4,subpl);
    
    % Get number of categories and generate good colors
    categories = unique(data{subpl});
    
    % Generate labels for each category
    clear labels
    for cat=1:length(categories)
        n_thisCat = sum(strcmp(data{subpl}, categories{cat}));
        pct_thisCat = round(100*(n_thisCat/n_datasets));
        labels{cat} = [categories{cat} ', n = ' num2str(n_thisCat)]; % ', ' num2str(pct_thisCat) '%'];
    end
    
    % Generate pie chart
    p = pie(ax,categorical(data{subpl}), [], labels);
    
    % Change font size and edge colors
    for cat=1:length(categories)
        t = p(2*cat);
        t.FontSize = 16;
        
        t = p(2*cat-1);
        %t.FaceColor = all_colors{color_scheme{length(categories)}(cat)};
    end
    
    % Add title to each subplot
    title(ax,titles{subpl});
end