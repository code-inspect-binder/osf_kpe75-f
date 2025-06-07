function plot_consecutive_bars(data, xTickLabels, p_text, subplot_num, title_text)

% New subplot
ax = subplot(3,2,subplot_num);

%Plot confidence intervals
for i=1:length(data)
    bar(i, mean(data{i}));
    hold on
    plot([i,i], [mean(data{i})-std(data{i})/sqrt(length(data{i})), ...
        mean(data{i})+std(data{i})/sqrt(length(data{i}))], 'k', 'LineWidth', 3);

    % Plot individual data
    rand_offset = unifrnd(.1,.3,length(data{i}),1);
    plot(i+rand_offset, data{i}, 'k.','MarkerSize',10);
end

% Plot details
title(title_text);
ylabel('R value', 'FontSize', 24);
xlim([.5, length(data)+.5]);
ylim([-1,1.4]);
set(ax,'XTick',[1:length(data)]);
set(gca,'XTickLabel',xTickLabels);
box off

% Plot p values
if length(data)==2
    plot([1,2], [1,1], 'k-');
    text(1.5, 1.1, p_text, 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 12);
else
    plot([1,2], [1,1], 'k-');
    plot([2,3], [1.03,1.03], 'k-');
    plot([1,3], [1.2,1.2], 'k-');
    text(1.5, 1.1, p_text{1}, 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 12);
    text(2.5, 1.13, p_text{2}, 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 12);
    text(2, 1.3, p_text{3}, 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 12);
end