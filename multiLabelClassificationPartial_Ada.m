function [theta_nodes_ml, theta_pairwise_ml] = multiLabelClassificationPartial_Ada(feature_nodes,feature_pairwise,groundTruth, labelmasks)


warning off;
folder='mlp/';
%%Initialization **************************************************************************************
save_after=10;
n_training=size(groundTruth,2);
global theta_pairwise_ml;
global theta_nodes_ml;
global n_node_features;
global n_nodes;
global n_word2vec_features;
global n_pairs;
global weight_size_node;
global weight_size_pairwise;
global word2vecFeatures;
global mypath;
n_nodes=size(feature_pairwise,2); % number of classes
n_word2vec_features=size(feature_pairwise,3); %word2vec features
n_node_features=size(feature_nodes,1);
%n_pairs= (n_nodes*(n_nodes-1))/2;% n_nodes*n_nodes; %
%n_pairs=81*81;
weight_size_node=n_node_features;
weight_size_pairwise=n_word2vec_features;
alpha=1e-5;
maxiteration=1000;
theta_node_all=zeros(n_nodes,maxiteration);
theta_pairwise_all=zeros(n_word2vec_features,maxiteration);

avg_game_value_maximizer = zeros(maxiteration,1);  % the avg of game values over training examples
avg_objective_value_maximizer=zeros(maxiteration,1);

avg_grads_magnitude_node = zeros(maxiteration,1); %
avg_grads_magnitude_pairwise = zeros(maxiteration,1); %

sum_game_value_maximizer_batch=0;
sum_objective_value_maximizer_batch=0;

sum_objective_value_maximizer_total=0; % the sum of objective function values over training examples
sum_game_value_maximizer_total=0;

sum_grad_batch_node = zeros (n_nodes,weight_size_node);  % % the sum of gradients over training examples in each batch
sum_grad_batch_pairwise = zeros (weight_size_pairwise,1);  % % the sum of gradients over training examples in each batch

avg_grad_batch_node = zeros (n_nodes,weight_size_node);
avg_grad_batch_pairwise = zeros (weight_size_pairwise,1);

% adagrad ********************************************
%https://xcorr.net/2014/01/23/adagrad-eliminating-learning-rates-in-stochastic-gradient-descent/
autocorr = 0.95;
fudge_factor=1e-6; %for numerical stability
master_stepsize = 1e-2;
historical_grad_node= zeros (n_nodes,weight_size_node);
historical_grad_pairwise= zeros (weight_size_pairwise,1);
%********************************************

batchSize=10;
if (n_training<batchSize)
    batchSize=n_training;
end
n_batch=n_training/batchSize;
%% Training
%while 1
%nitr=nitr+1;
for nitr=1:maxiteration
    % order = randperm ( n_training);
    
    % sum_objective_value_maximizer=0;
    %  for i=1:n_training
    %           ind=order(i)
    order = randperm ( n_training);
    %     report=ones(1000,4);
    %     ctr=1;
    sum_objective_value_maximizer_total=0;
    sum_game_value_maximizer_total=0;
    
    for idx=1:batchSize:n_training
        sum_objective_value_maximizer_batch=0;
        sum_game_value_maximizer_batch=0;
        %          tic;
        
        for bindex=idx:(idx+batchSize-1)
            if bindex<=n_training
                
                %             bindex
                ind=order(bindex);
                [sample_grad_nodes,sample_grad_pairwise,sum_game_value_maximizer,sum_objective_value_maximizer]...
                    = game_step_activeML_ADA(feature_nodes(:,ind),feature_pairwise,groundTruth(:,ind),labelmasks(:,ind));
                sum_grad_batch_node=sum_grad_batch_node+sample_grad_nodes; %n_classes*n_node_features
                sum_grad_batch_pairwise=sum_grad_batch_pairwise+(reshape(sum(sum(sample_grad_pairwise)),weight_size_pairwise,1))/n_pairs;
                sum_game_value_maximizer_batch=sum_game_value_maximizer_batch+sum_game_value_maximizer;
                sum_objective_value_maximizer_batch=sum_objective_value_maximizer_batch+sum_objective_value_maximizer;
            end
        end
        
        avg_grad_batch_node=sum_grad_batch_node./batchSize;
        sum_grad_batch_node= zeros (n_nodes,weight_size_node);
        
        avg_grad_batch_pairwise=sum_grad_batch_pairwise./batchSize;
        sum_grad_batch_pairwise= zeros (weight_size_pairwise,1);
        
        
        sum_game_value_maximizer_total=sum_game_value_maximizer_total+(sum_game_value_maximizer_batch./batchSize);
        sum_objective_value_maximizer_total=sum_objective_value_maximizer_total+(sum_objective_value_maximizer_batch./batchSize);
        %% adagrad
        %node features
        %                 if historical_grad_node==0
        historical_grad_node=historical_grad_node+(avg_grad_batch_node.^2);
        %                 else
        %                     historical_grad_node=(autocorr*historical_grad_node)+((1-autocorr)*(avg_grad_batch_node.^2));
        %                 end
        %         word2vec features
        %                 if historical_grad_pairwise==0
        historical_grad_pairwise=historical_grad_pairwise+(avg_grad_batch_pairwise.^2);
        
        adjusted_grad_node=avg_grad_batch_node./(sqrt(historical_grad_node)+fudge_factor);
        adjusted_grad_pairwise=avg_grad_batch_pairwise./(sqrt(historical_grad_pairwise)+fudge_factor);
        
        %% gradient update
        theta_nodes_ml=theta_nodes_ml - master_stepsize * adjusted_grad_node;
        theta_pairwise_ml=theta_pairwise_ml - master_stepsize * adjusted_grad_pairwise;
        theta_pairwise_ml=max(theta_pairwise_ml,0);
        
    end
    
    
    %% Recording
    
    theta_pairwise_all(:,nitr)=theta_pairwise_ml;
    theta_node_all(:,nitr)=sum(theta_nodes_ml,2)/n_node_features;
    
    avg_game_value_maximizer(nitr) = sum_game_value_maximizer_total/n_batch; %n_training; % average
    sum_game_value_maximizer_total=0;
    
    avg_objective_value_maximizer(nitr)= sum_objective_value_maximizer_total/n_batch; %n_training;
    if (sum_objective_value_maximizer_total<0)
        nitr
        ind
    end
    sum_objective_value_maximizer_total=0;
    
    avg_grads_magnitude_node(nitr) = (1/(n_nodes*weight_size_node))*sum(sum(abs(adjusted_grad_node)));
    avg_grads_magnitude_pairwise(nitr) = (1/weight_size_pairwise)*sum(abs(adjusted_grad_pairwise));
    
    if (nitr == maxiteration)
        
        des1=strcat(mypath,'\','theta_nodes_ml.mat');
        save (des1,'theta_nodes_ml');
        des2=strcat(mypath,'\','theta_pairwise_ml.mat');
        save (des2,'theta_pairwise_ml');
        %        save mlp/theta_nodes_ml theta_nodes_ml;
        %        save mlp/theta_pairwise_ml theta_pairwise_ml;
        disp('exceeded maximum iteration');
        
        break_condition = 'exceeded maximum iteration';
        
    end
    
    if(avg_grads_magnitude_node(nitr) < 0.01 && avg_grads_magnitude_pairwise(nitr) < 0.01)
        
        des1=strcat(mypath,'\','theta_nodes_ml.mat');
        save (des1,'theta_nodes_ml');
        des2= strcat(mypath,'\','theta_pairwise_ml.mat');%sprintf(mypath,'\','theta_pairwise_ml.mat');
        save (des2,'theta_pairwise_ml');
        break;
    end
    
    if( mod (nitr, save_after) == 0 ) % based on data size, itr takes variable times. so save on update count instead
        
        %*****************************************************************
        
        fig=figure('Visible','off','Position', [0 0 1024 800]);
        
        plot(avg_objective_value_maximizer(1:nitr));
        
        figName=strcat(mypath,'Objectiveplot.png');
        
        saveas(fig, figName);
        close(fig)
        %*****************************************************************
        
        fig=figure('Visible','off','Position', [0 0 1024 800]);
        
        plot(avg_grads_magnitude_node(1:nitr));
        figName = strcat(mypath,'gradplot_node.png');
        saveas(fig, figName);
        close(fig)
        %
        %         %*****************************************************************
        %
        fig = figure('Visible','off','Position', [0 0 1024 800]);
        
        plot(avg_grads_magnitude_pairwise(1:nitr));
        figName = strcat(mypath,'gradplot_pairwise.png');
        saveas(fig, figName);
        close(fig)
        des1=strcat(mypath,'\','theta_nodes_ml.mat');
        save (des1,'theta_nodes_ml');
        des2=strcat(mypath,'\','theta_pairwise_ml.mat');
        save (des2,'theta_pairwise_ml');
        
    end
    
end

