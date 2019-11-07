%% NEURAL NETWORK GRAPH GENERATOR
% Author: Andy
%
%

%% Run plotting functions
plotRBF_LR();
plotMLP_LR();


%% Read in and plot learning rate tuning for MLP
function plotMLP_LR()
    [file, path] = uigetfile('../Output/*.csv','Select learning rate data:');
    data = readmatrix(strcat(path, file), 'OutputType','string');
    
    %%
    colors = ["r", "g", "b", "c", "m", "k"];
    textures = ["-", "--", ":"];
    linewidth = 2.0;
    
    %%
    % Make array types of values
    datasets = unique(data(:,1), 'stable');       % colors
    hidden_layers = unique(data(:,2), 'stable');  % texture
    hidden_nodes = unique(data(:,3), 'stable');
    learning_rates = unique(data(:,4), 'stable'); % x-axis
    
    HIDDEN_NODES_TO_PLOT = "2";
%     HIDDEN_NODES_TO_PLOT = "2.333333333";
%     HIDDEN_NODES_TO_PLOT = "1";
    
    %% PLOT CLASSIFICATION STUFF
    figure();
    for d = 1:3 % Go through datasets
        p1 = data(data(:) == datasets(d), :);
        for h = 1:length(hidden_layers)
            p2 = p1(p1(:,2) == hidden_layers(h), :); % filter by hidden layer
            p2 = p2(p2(:,3) == HIDDEN_NODES_TO_PLOT, :); % filter by number of hidden nodes
            
            accuracy = p2(:,6);
            accuracy = str2double(accuracy);
            mse = p2(:,7);
            mse = str2double(mse);
            
            linetitle = strcat(datasets(d), " ", hidden_layers(h), " Hidden Layers");
            % Plot stuff
            subplot(2,2,1);
            plot(accuracy, strcat(textures(h), "", colors(d)), ...
                'DisplayName', linetitle, ...
                'LineWidth', linewidth);
            xticks(1:length(accuracy));
            xticklabels(learning_rates);
            xlabel('Learning Rate')
            ylabel('Accuracy')
            legend('FontSize', 7)
            hold on
            subplot(2,2,2);
            plot(mse, strcat(textures(h), "", colors(d)), ...
                'DisplayName', linetitle, ...
                'LineWidth', linewidth);
            xticklabels(learning_rates);
            xticks(1:length(mse));
            xlabel('Learning Rate')
            ylabel('MSE')
            legend('FontSize', 7)
            hold on
        end
    end
     %% PLOT REGRESSION STUFF
    for d = 4:6
        p1 = data(data(:) == datasets(d), :);
        for h = 1:length(hidden_layers)
            p2 = p1(p1(:,2) == hidden_layers(h), :);
            p2 = p2(p2(:,3) == HIDDEN_NODES_TO_PLOT, :); % filter by number of hidden nodes
            p2 = p2(p2(:,4) ~= "2", :); % filter out learning rate higher than 2 b/c its trash and messes up scaling
            
            
            mse = p2(:,6);
            mse = str2double(mse);
            me = p2(:,7);
            me = str2double(me);
            
            linetitle = strcat(datasets(d), " ", hidden_layers(h), " Hidden Layers");
            % Plot stuff
            subplot(2,2,3);
            plot(mse, strcat(textures(h), "", colors(d)), ...
                'DisplayName', linetitle, ...
                'LineWidth', linewidth);
            xticks(1:length(mse));
            xticklabels(learning_rates);
            xlabel('Learning Rate')
            ylabel('MSE')
            legend('FontSize', 7)
            %BOUND Y AXIS
            ylim([0 1.4]);
            
            hold on
            subplot(2,2,4);
            plot(me, strcat(textures(h), "", colors(d)), ...
                'DisplayName', linetitle, ...
                'LineWidth', linewidth);
            xticklabels(learning_rates);
            xticks(1:length(me));
            xlabel('Learning Rate')
            ylabel('ME')
            legend('FontSize', 7)
            hold on
        end
    end
    
end

%% Read in and plot learning rate tuning for RBF
function plotRBF_LR()
    [file, path] = uigetfile('../Output/*.csv','Select learning rate data:');
    data = readmatrix(strcat(path, file), 'OutputType','string');
    
    %%
    colors = ["r", "g", "b", "c", "m", "k"];
    textures = ["-", "--", ":"];
    linewidth = 2.0;
    
    %%
    % Make array types of values
    datasets = unique(data(:,1), 'stable');       % colors
    clusters = unique(data(:,2), 'stable');       % clustering method
    learning_rates = unique(data(:,4), 'stable'); % x-axis
    
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
            subplot(2,2,1);
            plot(accuracy, strcat(textures(c), "", colors(d)), ...
                'DisplayName', linetitle, ...
                'LineWidth', linewidth);
            xticks(1:length(accuracy));
            xticklabels(learning_rates);
            xlabel('Learning Rate')
            ylabel('Accuracy')
            legend('FontSize', 7)
            hold on
            subplot(2,2,2);
            plot(mse, strcat(textures(c), "", colors(d)), ...
                'DisplayName', linetitle, ...
                'LineWidth', linewidth);
            xticklabels(learning_rates);
            xticks(1:length(mse));
            xlabel('Learning Rate')
            ylabel('MSE')
            legend('FontSize', 7)
            hold on
        end
    end
    %% PLOT REGRESSION STUFF
    for d = 4:6
        p1 = data(data(:) == datasets(d), :);
        for c = 1:length(clusters)
            p2 = p1(p1(:,2) == clusters(c), :);
            
            mse = p2(:,5);
            mse = str2double(mse);
            me = p2(:,6);
            me = str2double(me);
            
            linetitle = strcat(datasets(d), " ", clusters(c));
            % Plot stuff
            subplot(2,2,3);
            plot(mse, strcat(textures(c), "", colors(d)), ...
                'DisplayName', linetitle, ...
                'LineWidth', linewidth);
            xticks(1:length(mse));
            xticklabels(learning_rates);
            xlabel('Learning Rate')
            ylabel('MSE')
            legend('FontSize', 7)
            hold on
            subplot(2,2,4);
            plot(me, strcat(textures(c), "", colors(d)), ...
                'DisplayName', linetitle, ...
                'LineWidth', linewidth);
            xticklabels(learning_rates);
            xticks(1:length(me));
            xlabel('Learning Rate')
            ylabel('ME')
            legend('FontSize', 7)
            hold on
        end
    end
end