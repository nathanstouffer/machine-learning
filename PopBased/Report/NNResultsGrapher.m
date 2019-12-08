%% NEURAL NETWORK GRAPH GENERATOR
% Author: Andy Kirby
% Last Edit: 11/6/2019
% Generatess various graphs to visually evaluate the performance of our
% network during tuning and final verification.

%% Run plotting functions
% Comment/uncomment the graphs that you want
% plotRBF_LR();
% plotMLP_LR();
% plotMLP_M();
% plotRBF_final();
% plotMLP_final();
plotPop_final(0);
plotPop_final(1);
plotPop_final(2);

function plotPop_final(num_hl)
    [file, path] = uigetfile('*.csv','Select final data:');
    data = readmatrix(strcat(path, file), 'OutputType','string');
    
    %%
    colors = ["r", "g", "b", "c", "m", "k"];
    textures = ["-", "--", ":"];
    linewidth = 2.0;
    
    %%
    % Make array types of values
    datasets = unique(data(:,2), 'stable');       % column
    method = unique(data(:,1), 'stable');
%     hidden_layers = unique(data(:,3), 'stable');  % colors
    hidden_layers = num_hl;
    
    %% PLOT CLASSIFICATION STUFF
    figure();
    for d = 1:3 % Go through datasets
        p1 = data(data(:,2) == datasets(d), :);
%         for h = 1:length(hidden_layers)
        for m = 1:length(method)
            p2 = p1(p1(:,1) == method(m), :); % filter by method
            p2 = p2(p2(:,3) == num2str(hidden_layers), :); % filter by hidden layer
%             p2 = p2(p2(:,3) == HIDDEN_NODES_TO_PLOT, :); % filter by number of hidden nodes
            
            accuracy = p2(:,7);
            accuracy = str2double(accuracy);
            mse = p2(:,8);
            mse = str2double(mse);
            
            class_acc(d, m) = accuracy;
            class_mse(d, m) = mse;
        end
    end
     %% PLOT REGRESSION STUFF
    for d = 4:6
        p1 = data(data(:,2) == datasets(d), :);
        for m = 1:length(method)
            p2 = p1(p1(:,1) == method(m), :); % filter by method
            p2 = p2(p2(:,3) == num2str(hidden_layers), :); % filter by hidden layer
%             p2 = p1(p1(:,2) == hidden_layers(h), :);
%             p2 = p2(p2(:,3) == HIDDEN_NODES_TO_PLOT, :); % filter by number of hidden nodes
%             p2 = p2(p2(:,4) ~= "2", :); % filter out learning rate higher than 2 b/c its trash and messes up scaling
            
            mse = p2(:,7);
            mse = str2double(mse);
            me = p2(:,8);
            me = str2double(me);
            
            reg_mse(d-3, m) = mse;
            reg_me(d-3, m) = me;
        end
    end
    
%     hidden_layers(:,:) = strcat(hidden_layers(:,:), " Hidden Layers");
%     hidden_layers = strcat(num2str(hidden_layers), " Hidden Layers");
    figure();
    subplot(2,2,1);
    bar(class_acc)
    xticklabels(datasets(1:3));
    ylabel('Accuracy')
    xlabel('Dataset')
    legend(method,'FontSize', 10)
    title('Classification')
    
    subplot(2,2,3);
    bar(class_mse)
    ylim([0, 60]);
    xticklabels(datasets(1:3));
    ylabel('MSE')
    xlabel('Dataset')
    legend(method,'FontSize', 10)
    subplot(2,2,2);
    bar(reg_mse)
    xticklabels(datasets(4:6));
    ylabel('MSE')
    xlabel('Dataset')
    legend(method,'FontSize', 10)
    title('Regression')
    subplot(2,2,4);
    bar(reg_me)
    xticklabels(datasets(4:6));
    ylabel('ME')
    xlabel('Dataset')
    legend(method,'FontSize', 10)
    
    sgtitle(strcat("Final Results: ", num2str(hidden_layers) ," Hidden Layers"));

end


function plotMLP_final()
    [file, path] = uigetfile('../Output/*.csv','Select MLP final data:');
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
    
    %% PLOT CLASSIFICATION STUFF
    figure();
    for d = 1:3 % Go through datasets
        p1 = data(data(:) == datasets(d), :);
        for h = 1:length(hidden_layers)
            p2 = p1(p1(:,2) == hidden_layers(h), :); % filter by hidden layer
%             p2 = p2(p2(:,3) == HIDDEN_NODES_TO_PLOT, :); % filter by number of hidden nodes
            
            accuracy = p2(:,6);
            accuracy = str2double(accuracy);
            mse = p2(:,7);
            mse = str2double(mse);
            
            class_acc(d, h) = accuracy;
            class_mse(d, h) = mse;
        end
    end
     %% PLOT REGRESSION STUFF
    for d = 4:6
        p1 = data(data(:) == datasets(d), :);
        for h = 1:length(hidden_layers)
            p2 = p1(p1(:,2) == hidden_layers(h), :);
%             p2 = p2(p2(:,3) == HIDDEN_NODES_TO_PLOT, :); % filter by number of hidden nodes
            p2 = p2(p2(:,4) ~= "2", :); % filter out learning rate higher than 2 b/c its trash and messes up scaling
            
            
            mse = p2(:,6);
            mse = str2double(mse);
            me = p2(:,7);
            me = str2double(me);
            
            reg_mse(d-3, h) = mse;
            reg_me(d-3, h) = me;
        end
    end
    
    hidden_layers(:,:) = strcat(hidden_layers(:,:), " Hidden Layers");
    figure();
    subplot(2,2,1);
    bar(class_acc)
    xticklabels(datasets(1:3));
    ylabel('Accuracy')
    xlabel('Dataset')
    legend(hidden_layers,'FontSize', 10)
    title('Classification')
    
    subplot(2,2,3);
    bar(class_mse)
    ylim([0, 60]);
    xticklabels(datasets(1:3));
    ylabel('MSE')
    xlabel('Dataset')
    legend(hidden_layers,'FontSize', 10)
    subplot(2,2,2);
    bar(reg_mse)
    xticklabels(datasets(4:6));
    ylabel('MSE')
    xlabel('Dataset')
    legend(hidden_layers(1:3),'FontSize', 10)
    title('Regression')
    subplot(2,2,4);
    bar(reg_me)
    xticklabels(datasets(4:6));
    ylabel('ME')
    xlabel('Dataset')
    legend(hidden_layers(1:3),'FontSize', 10)
    
    sgtitle("MLP Network Final Results");
end

%% Read in and plot final results for RBF
function plotRBF_final()
    [file, path] = uigetfile('../Output/*.csv','Select RBF final data:');
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
    
    %% Get CLASSIFICATION data
    for d = 1:3
        p1 = data(data(:) == datasets(d), :);
        for c = 1:length(clusters)
            p2 = p1(p1(:,2) == clusters(c), :);            
            accuracy = p2(:,5);
            accuracy = str2double(accuracy);
            mse = p2(:,6);
            mse = str2double(mse);
            
            class_acc(d, c) = accuracy;
            class_mse(d, c) = mse;
        end
    end
    %% Get REGRESSION data
    for d = 4:6
        p1 = data(data(:) == datasets(d), :);
        for c = 2:length(clusters)
            p2 = p1(p1(:,2) == clusters(c), :);
            
            mse = p2(:,5);
            mse = str2double(mse);
            me = p2(:,6);
            me = str2double(me);
            
            reg_mse(d-3, c-1) = mse;
            reg_me(d-3, c-1) = me;
        end
    end
    
    figure();
    subplot(2,2,1);
    bar(class_acc)
    xticklabels(datasets(1:3));
    ylabel('Accuracy')
    xlabel('Dataset')
    legend(clusters,'FontSize', 10)
    title('Classification')
    subplot(2,2,3);
    bar(class_mse)
    xticklabels(datasets(1:3));
    ylabel('MSE')
    xlabel('Dataset')
    legend(clusters,'FontSize', 10)
    subplot(2,2,2);
    bar(reg_mse)
    xticklabels(datasets(4:6));
    ylabel('MSE')
    xlabel('Dataset')
    legend(clusters(2:3),'FontSize', 10)
    title('Regression')
    subplot(2,2,4);
    bar(reg_me)
    xticklabels(datasets(4:6));
    ylabel('ME')
    xlabel('Dataset')
    legend(clusters(2:3),'FontSize', 10)
    
    sgtitle("RBF Network Final Results");
end

%% Read in and plot momentum tuning for MLP
function plotMLP_M()
    [file, path] = uigetfile('../Output/*.csv','Select MLP momentum data:');
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
    learning_rates = unique(data(:,4), 'stable'); 
    momentums = unique(data(:,5), 'stable'); % x-axis
    
    %% PLOT CLASSIFICATION STUFF
    figure();
    for d = 1:3 % Go through datasets
        p1 = data(data(:) == datasets(d), :);
            
        accuracy = p1(:,6);
        accuracy = str2double(accuracy);
        mse = p1(:,7);
        mse = str2double(mse);
        
        linetitle = strcat(datasets(d));
        % Plot stuff
        subplot(2,2,1);
        plot(accuracy, strcat(textures(1), "", colors(d)), ...
            'DisplayName', linetitle, ...
            'LineWidth', linewidth);
        xticks(1:length(momentums));
        xticklabels(momentums);
        xlabel('Momentum')
        ylabel('Accuracy')
        legend('FontSize', 10)
        ylim([0 inf]);
        title('Classification')
        hold on
        subplot(2,2,3);
        plot(mse, strcat(textures(1), "", colors(d)), ...
            'DisplayName', linetitle, ...
            'LineWidth', linewidth);
        xticklabels(momentums);
        xticks(1:length(mse));
        xlabel('Momentum')
        ylabel('MSE')
        legend('FontSize', 10)
        ylim([0 inf]);
        hold on
    end
     %% PLOT REGRESSION STUFF
    for d = 4:6
        p1 = data(data(:) == datasets(d), :);
           
        mse = p1(:,6);
        mse = str2double(mse);
        me = p1(:,7);
        me = str2double(me);
        
        linetitle = strcat(datasets(d));
        % Plot stuff
        subplot(2,2,2);
        plot(mse, strcat(textures(1), "", colors(d)), ...
            'DisplayName', linetitle, ...
            'LineWidth', linewidth);
        xticks(1:length(momentums));
        xticklabels(momentums);
        xlabel('Momentum')
        ylabel('MSE')
        legend('FontSize', 10)
        %BOUND Y AXIS
        %             ylim([0 1.4]);
        ylim([0 inf]);
        title('Regression')
        
        hold on
        subplot(2,2,4);
        plot(me, strcat(textures(1), "", colors(d)), ...
            'DisplayName', linetitle, ...
            'LineWidth', linewidth);
        xticklabels(momentums);
        xticks(1:length(me));
        xlabel('Momentum')
        ylabel('ME')
        legend('FontSize', 10)
        yline(0,'HandleVisibility','off');
        hold on
    end
    sgtitle('MLP Momentum Tuning');
end

%% Read in and plot learning rate tuning for MLP
function plotMLP_LR()
    [file, path] = uigetfile('../Output/*.csv','Select MLP learning rate data:');
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
            legend('FontSize', 8)
            ylim([0 inf]);
            title('MLP Learning Rate Tuning - Classification');
            hold on
            subplot(2,2,3);
            plot(mse, strcat(textures(h), "", colors(d)), ...
                'DisplayName', linetitle, ...
                'LineWidth', linewidth);
            xticklabels(learning_rates);
            xticks(1:length(mse));
            xlabel('Learning Rate')
            ylabel('MSE')
            legend('FontSize', 8)
            ylim([0 inf]);
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
            subplot(2,2,2);
            plot(mse, strcat(textures(h), "", colors(d)), ...
                'DisplayName', linetitle, ...
                'LineWidth', linewidth);
            xticks(1:length(mse));
            xticklabels(learning_rates);
            xlabel('Learning Rate')
            ylabel('MSE')
            legend('FontSize', 8)
            %BOUND Y AXIS
%             ylim([0 1.4]);
            ylim([0 inf]);
            title('MLP Learning Rate Tuning - Regression');
            
            hold on
            subplot(2,2,4);
            plot(me, strcat(textures(h), "", colors(d)), ...
                'DisplayName', linetitle, ...
                'LineWidth', linewidth);
            xticklabels(learning_rates);
            xticks(1:length(me));
            xlabel('Learning Rate')
            ylabel('ME')
            legend('FontSize', 9)
            yline(0,'HandleVisibility','off');
            hold on
        end
    end
end

%% Read in and plot learning rate tuning for RBF
function plotRBF_LR()
    [file, path] = uigetfile('../Output/*.csv','Select RBF learning rate data:');
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
            legend('FontSize', 10)
            title('RBF Learning Rate Tuning - Classification');
            ylim([0 inf])
            hold on
            subplot(2,2,3);
            plot(mse, strcat(textures(c), "", colors(d)), ...
                'DisplayName', linetitle, ...
                'LineWidth', linewidth);
            xticklabels(learning_rates);
            xticks(1:length(mse));
            xlabel('Learning Rate')
            ylabel('MSE')
            legend('FontSize', 10)
            ylim([0 inf])
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
            subplot(2,2,2);
            plot(mse, strcat(textures(c), "", colors(d)), ...
                'DisplayName', linetitle, ...
                'LineWidth', linewidth);
            xticks(1:length(mse));
            xticklabels(learning_rates);
            xlabel('Learning Rate')
            ylabel('MSE')
            legend('FontSize', 10)
            ylim([0 inf])
            title('RBF Learning Rate Tuning - Regression');
            hold on
            subplot(2,2,4);
            plot(me, strcat(textures(c), "", colors(d)), ...
                'DisplayName', linetitle, ...
                'LineWidth', linewidth);
            xticklabels(learning_rates);
            xticks(1:length(me));
            xlabel('Learning Rate')
            ylabel('ME')
            legend('FontSize', 10)
            yline(0,'HandleVisibility','off');
            hold on
        end
    end
    
    sgtitle("RBF Learning Rate Tuning");
end