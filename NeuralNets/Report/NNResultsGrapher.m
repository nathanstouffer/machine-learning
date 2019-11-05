%% NEURAL NETWORK GRAPH GENERATOR
% Author: Andy
%
%

%%

% plotRBF_LR();


%% Read in and plot learning rate tuning for RBF
% function plotRBF_LR()
    [file, path] = uigetfile('../Output/*.csv','Select learning rate data:');
    data = readmatrix(strcat(path, file), 'OutputType','string');
    
    %%
    colors = ["r", "g", "b", "c", "m", "y", "k"];
    textures = ["-", "--", ":"];
    
    %%
    % Make array types of values
    datasets = unique(data(:,1), 'stable');       % colors
%     datasets = ["abalone.csv", "car.csv", "segmentation.csv", "forestfires.csv", "machine.csv", "winequality-red.csv"];
    clusters = unique(data(:,2), 'stable');       % clustering method
    learning_rates = unique(data(:,4), 'stable'); % x-axis
%     learning_rates = str2double(learning_rates);
    
    %% PLOT CLASSIFICATION STUFF
    figure();
    for d = 1:3
        p1 = data(data(:) == datasets(d), :);
        for c = 1:length(clusters)
            p2 = p1(p1(:,2) == clusters(c), :);
            
            accuracy = p2(:,5);
            accuracy = str2double(accuracy);
            mse = p2(:,6);
            mse = str2double(mse);
            
            linetitle = strcat(datasets(d), " ", clusters(c));
            % Plot stuff
            subplot(1,2,1);
            plot(accuracy, strcat(textures(c), ".", colors(d)), ...
                'DisplayName', linetitle);
            xticklabels(learning_rates);
            xlabel('Learning Rate')
            ylabel('Accuracy')
            legend('FontSize', 7)
            hold on
            subplot(1,2,2);
            plot(mse, strcat(textures(c), ".", colors(d)), ...
                'DisplayName', linetitle);
            xticklabels(learning_rates);
            xlabel('Learning Rate')
            ylabel('MSE')
            legend('FontSize', 7)
            hold on
        end
    end
    % Plot Accuracy
    
    
    %% PLOT REGRESSION STUFF
    
    
    % Plot dataset by color
    % Plot clustering method by pattern
    % Plot learning rate on x axis
% end