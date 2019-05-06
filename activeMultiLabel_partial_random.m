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
% address="bibtex1_train.mat";
% load("bibtex1_train.mat");
% load(address);
% groundTruth;
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
%multiLabelClassificationPartial_Ada(node_features_initial,feature_pairwise,ground_truth_initial,labelmasks);
total_cuts=0;

        src1=strcat(mypath,'\','theta_nodes_ml.mat');
        load(src1);
      src2=strcat(mypath,'\','theta_pairwise_ml.mat');
        load(src2);

for j = 1 : n_randomization

     sampleIndex_start = [randi(n_training, 1)]; %choose a random index
    feature_nodes_train = [node_features_initial node_features(:, sampleIndex_start)];
    groundTruth_train = [ground_truth_initial  groundTruth(:, sampleIndex_start)];
    chosenSample= [sampleIndex_start];
    index_nextSample=chosenSample;
 visitedSamples=zeros(n_samples,n_nodes);

     for i = 1 : n_datapoints
          
         %groundTruth_train_partial= groundTruth_train.*labelmasks;
           [theta_nodes_ml, theta_pairwise_ml] = multiLabelClassificationPartial_Ada(feature_nodes_train,feature_pairwise,groundTruth_train,labelmasks);
            
%            trained_indicies = zeros(size(chosenSample, 2), n_nodes+1);

%         trained_indicies(: , 1) = chosenSample';
%         trained_indicies(: , 2:n_nodes+1) = groundTruth_train';
%        node_features(:,index_nextSample)=[];
%        ground_truth(:,index_nextSample)=[];
%       
        [cuts,samples_evaluation,loss, p_maximizer, s_maximizer_binary, p_minimizer, s_minimizer_binary] = ...
            multiLabelClassificationPartial_test(node_features, feature_pairwise, ground_truth);
        %hammingLoss
       %hammingLoss
        loss_list_rba_random(i, j) = loss;
                    %index_next = label_sample(s_maximizer_binary, p_maximizer, chosen_index, index_test);
%        [max_value,vector_index]= max(samples_evaluation(:,1));
%         maxIndex = find(samples_evaluation == max_value);
%         if size(maxIndex, 1) == 1
        index_nextSample = randi(n_samples);
        
%         
%     else
%         
%         randix = randi(size(maxIndex, 1), 1);
%         index_nextSample = maxIndex(randix);
%         
%         end
        
                  chosenLabels=samples_evaluation(index_nextSample,2:n_nodes+1);
                    order=randperm(n_nodes);
               for ctr=1:n_nodes
                   if (visitedSamples(index_nextSample,1+chosenLabels(order(ctr)))==0)
                       visitedSamples(index_nextSample,1+chosenLabels(order(ctr)))=1;
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
            des=strcat(mypath,'\','loss_list_ADV_partial_random');
        save (des,'loss_list_ADV_partial_random.mat');
          %  save mlp/loss_list.mat loss_list;
        end
    end
end
cut_avg=total_cuts/(n_datapoints*n_randomization);
 des2=strcat(mypath,'\','ws');
        save (des2);
  %save  ws1;
