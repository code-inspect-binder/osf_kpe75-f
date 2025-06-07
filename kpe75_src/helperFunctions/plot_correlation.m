function plot_correlation(X, Y)

% Transform confidenceScale values from 100 to 15 for plotting purposes
X(X==100) = 15;

% Create a subplot
ax = subplot(3,2,[1,2]);

% Plot scatter
plot(X, Y, 'kd')
hold on

% Perform regression + plot line of best fit
b = regress(Y, [ones(length(X),1), X]);
xLimits = [min(X)-.5, max(X)+.5];
plot(xLimits, b(1) + xLimits.*b(2), 'k')

% Set labels and limits
xlim(xLimits);
ylim([min(Y)-.1, max(Y)+.1]);
ylabel('R value', 'FontSize', 24)
title('Confidence scale')
box off

% Set ticks
set(ax,'XTick',[2,3,4,5,6,9,11,15]);
set(gca,'XTickLabel',{'2-point','3-point','4-point','5-point','6-point','9-point','11-point','continuous'});
xtickangle(45);

% Plot the p value
%[r,p] = corr(X, Y);
[r,~,p] = spear(X, Y);
corr_text = {['r = ' num2str(round(r*1000)/1000)], ['p = ' num2str(round(p*10000)/10000)]};
text(max(X)-2.5, min(Y)+.2, corr_text, 'HorizontalAlignment', 'left', ...
    'FontWeight', 'bold', 'FontSize', 12);