function [sample_grad_node,sample_grad_pairwise,sum_game_value_maximizer,sum_objective_value_maximizer]=...
    game_step_activeML_ADA(node_features,feature_pairwise,groundTruth,labelMask)
global n_nodes;
global n_pairs;
global n_word2vec_features;
global weight_size_node;
global weight_size_pairwise;
global theta_nodes_ml;
global theta_pairwise_ml;
global word2vecFeatures;
sum_objective_value_maximizer=0;
sum_game_value_maximizer=0;
lagrangianPotentials_pairwise=zeros(n_nodes,n_nodes);
lagrangianPotentials_node=(theta_nodes_ml*node_features)';
    for slice = 1 : n_nodes
        for sl=1:n_nodes
            lagrangianPotentials_pairwise(slice,sl)=reshape(feature_pairwise(slice,sl,:),1,size(feature_pairwise,3))*theta_pairwise_ml; 

        end
    end

    lagrangianPotentials_pairwise=lagrangianPotentials_pairwise; 
   lagrangianPotentials_node_gt=(lagrangianPotentials_node*groundTruth)/n_nodes; 
   groundTruth_features_pairwise_temp=feature_pairwise_generator(groundTruth,word2vecFeatures,0);
  
    for it=1:size(labelMask)
        if labelMask(it)==0
         groundTruth_features_pairwise_temp(it,:,:)=0;
         groundTruth_features_pairwise_temp(:,it,:)=0;
        end
    end
   groundTruth_features_pairwise=groundTruth_features_pairwise_temp;
    
   templagrangianPotentials_pairwise_gt=zeros(n_nodes,n_nodes);
     for slice = 1 :n_nodes
        for sl=1:n_nodes
            templagrangianPotentials_pairwise_gt(slice,sl)=(reshape(groundTruth_features_pairwise(slice,sl,:),1,n_word2vec_features)*theta_pairwise_ml)'; 
        end
     end
    lagrangianPotentials_piarwise_gt=(sum(sum(templagrangianPotentials_pairwise_gt))); %/n_pairs;
    [p_maximizer,game_value_maximizer,s_maximizer_nodes]= DOMMulti_activeML_ADA(groundTruth,...
    node_features,lagrangianPotentials_node,feature_pairwise,lagrangianPotentials_pairwise);
%     
  sum_objective_value_maximizer=sum_objective_value_maximizer+(lagrangianPotentials_node_gt)+(lagrangianPotentials_piarwise_gt)+game_value_maximizer(1);
%     
  sum_game_value_maximizer=sum_game_value_maximizer+ game_value_maximizer(1);
    
 maximizer_size=size(s_maximizer_nodes,1);
maximizer_expectation_pairwise=zeros(n_nodes,n_nodes,300);
%     
for id=1:maximizer_size
    temp=feature_pairwise_generator(double(s_maximizer_nodes(id,:)),word2vecFeatures,0);
    for it=1:size(labelMask)
        if labelMask(it)==0
         temp(it,:)=0;
         temp(:,it)=0;
        end
    end
      maximizer_expectation_pairwise=maximizer_expectation_pairwise+(p_maximizer(id)*temp);
end
sample_grad_node=((groundTruth.*double(labelMask)+(p_maximizer'*double(s_maximizer_nodes))'.*labelMask)*node_features');%/n_pairs;

 sample_grad_pairwise=((groundTruth_features_pairwise)+maximizer_expectation_pairwise);
%     
    
end


