function [avg_cuts, hammingLoss_avg, p_maximizer, s_maximizer_ml, p_minimizer, s_minimizer_ml]= multiLabelClassificationPartial_test(feature_nodes, feature_pairwise, groundTruth)

%%Initialization **************************************************************************************
n_sample = size(groundTruth, 2);
global n_node_features;
global n_nodes;
global n_pairs;

global weight_size_node;
global weight_size_pairwise;
%n_nodes = size(groundTruth,2); % number of classes
%n_node_features = size(feature_nodes,1);
%n_pairs = n_nodes * (n_nodes );
weight_size_node = n_node_features;
st = 0;
global n_solicitation;
global theta_nodes_ml;
global theta_pairwise_ml;
groundTruth_labels = groundTruth(:, st+1 : st+n_sample);
predicted_labels= zeros(n_sample,n_nodes);
hammingLoss=zeros(n_sample,1);
n_cuts=zeros(n_sample,1);
samples=zeros(n_sample,1+n_solicitation);
%% Test
for i=1:n_sample
    lagrangianPotentials_node = (theta_nodes_ml * feature_nodes(:,i)); %/n_nodes;
    lagrangianPotentials_pairwise = reshape(reshape(feature_pairwise, n_nodes*n_nodes, weight_size_pairwise)...
        *theta_pairwise_ml, n_nodes, n_nodes);
    lagrangianPotentials_pairwise=lagrangianPotentials_pairwise;%/n_pairs;
    
    [p_maximizer, p_minimizer, game_value_maximizer, s_maximizer_ml, s_minimizer_ml]...
        = DOMMulti_activeML(groundTruth(:,i), feature_nodes(:,i), lagrangianPotentials_node', ...
        feature_pairwise,lagrangianPotentials_pairwise);
        n_cuts(i)= size(s_maximizer_ml,1);    
    [k, maxIndex] = max(p_maximizer);
    predicted_labels(i,:) = s_maximizer_ml(maxIndex, :);
    hammingLoss(i) = pdist2(groundTruth(:,i)', predicted_labels(i,:), 'hamming');
end
hammingLoss_avg = sum(hammingLoss)/size(hammingLoss,1);
avg_cuts=sum(n_cuts)/n_sample;
end









