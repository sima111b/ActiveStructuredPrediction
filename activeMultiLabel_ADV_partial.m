function loss= activeMultiLabel_ADV_partial(address)

warning off;
% clc;
% clear;
restoredefaultpath;
addpath(genpath(pwd));
global word2vecFeatures;
% Please set the matrices in a way that the last dimention defines the
% number of entities in the matrix
load(address,'groundTruth');
load(address, 'nodeFeatures');
load(address, 'word2vecFeatures');
C = strsplit(address,'\');
ds_name=C{end}(1:end-4);
mkdir('experiments',ds_name);
mypath=strcat('\experiments\',ds_name);
% Initialization
n_training=size(groundTruth,2);
global n_node_features;
global n_nodes;
global n_word2vec_features;
global n_pairs;
global theta_nodes_ml;
global theta_pairwise_ml;
global n_solicitation;
global n_solutions;
global mypath;
n_solutions=100;
n_nodes=size(word2vecFeatures,2); % number of classes
n_word2vec_features=size(word2vecFeatures,1); %word2vec features
n_node_features=size(nodeFeatures,1);
n_pairs=n_nodes*n_nodes; %(n_nodes*(n_nodes-1))/2;
n_initial_training= 100; %round(0.1*n_training);
ground_truth_initial = groundTruth(:, end-n_initial_training+1:end);
node_features_initial = nodeFeatures(:, end-n_initial_training+1:end );
labelmasks = ones(n_nodes, n_initial_training+1);
n_training=n_training-n_initial_training;
randomizedDataIndex=randperm(n_training); %shuffle the indicies
n_solicitation=ceil(n_nodes/2); % for partial solicitation
node_features = nodeFeatures(:, randomizedDataIndex);
node_features = zscore(node_features);
feature_pairwise=feature_pairwise_generator(ones(n_nodes,1),word2vecFeatures,1); %81*81*300
ground_truth = groundTruth(:, randomizedDataIndex);
n_randomization=30;
n_datapoints=300;
loss_list = zeros(n_randomization, n_datapoints);
weight_size_node=n_node_features;
weight_size_pairwise=n_word2vec_features;
test_percentage = 0.30;
n_test = round(size(node_features, 2) * test_percentage);
index_test = [size(node_features, 2) - n_test + 1 :size(node_features, 2) ];
theta_nodes_ml=randn(n_nodes,weight_size_node);
theta_pairwise_ml=abs(randn(n_word2vec_features,1));
total_cuts=0;

for j = 1 : n_randomization
    
    sampleIndex_start = [randi(n_training, 1)]; %choose a random index
    feature_nodes_train = [node_features_initial node_features(:, sampleIndex_start)];
    groundTruth_train = [ground_truth_initial  groundTruth(:, sampleIndex_start)];
    chosenSample= [sampleIndex_start];
    index_nextSample=chosenSample;
    visitedSamples=zeros(n_samples,n_nodes);
    
    for i = 1 : n_datapoints
        
        [theta_nodes_ml, theta_pairwise_ml] = multiLabelClassificationPartial_Ada(feature_nodes_train,feature_pairwise,groundTruth_train,labelmasks);
        
        [cuts,samples_evaluation,loss, p_maximizer, s_maximizer_binary, p_minimizer, s_minimizer_binary] = ...
            multiLabelClassificationPartial_test(node_features, feature_pairwise, ground_truth);
        %hammingLoss
        loss_list(i, j) = loss;
        total_cuts=total_cuts+cuts;
        [max_value,vector_index]= max(samples_evaluation(:,1));
        maxIndex = find(samples_evaluation == max_value);
        if size(maxIndex, 1) == 1
            index_nextSample = maxIndex;
            
        else
            
            randix = randi(size(maxIndex, 1), 1);
            index_nextSample = maxIndex(randix);
            
        end
        chosenLabels=samples_evaluation(index_nextSample,2:n_nodes+1);
        
        for ctr=1:n_nodes
            if (visitedSamples(index_nextSample,1+chosenLabels(ctr))==0)
                visitedSamples(index_nextSample,1+chosenLabels(ctr))=1;
                break;
            end
        end
        if (ismember(chosenSample,index_nextSample)==1)
            ind=find(chosenSample(1,:)==index_nextSample);
            labelmasks(n_initial_training+ind,:)=visitedSamples(index_nextSample,:);
        else
            chosenSample = [chosenSample index_nextSample];
            nodes_features_train = [nodes_features_train node_features_test(:, index_nextSample) ];
            groundTruth_train = [groundTruth_train ground_truth_test(:, index_nextSample) ];
        end
        
        if mod(i, 10) == 0
            des=strcat(mypath,'\','loss_list');
            save (des,'loss_list.mat');
            %  save mlp/loss_list.mat loss_list;
        end
    end
end
cut_avg=total_cuts/(n_datapoints*n_randomization);
des2=strcat(mypath,'\','ws');
save (des2);
