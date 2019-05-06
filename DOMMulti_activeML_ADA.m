function [p_maximizer,game_value_maximizer,s_maximizer_binary]= DOMMulti_activeML_ADA(gt,...
    nodeFeature,lagrangianPotentials_node,pairwiseFeature,lagrangianPotentials_pairwise)

%---------------------------------------------------------------------
global n_nodes;
global n_pairs;
global n_node_features;
n_strategies=2^n_nodes-1;
global n_word2vec_features;
global word2vecFeatures;
global theta_pairwise_ml;
global theta_nodes_ml;
weight_size_pairwise=n_word2vec_features;
maxCondition=1;
minCondition=1;
p_maximizer=1;
p_minimizer=1;
lp_pairwise=cell(1,1000);
global n_solutions;

s_maximizer_index= 1;%randi(n_strategies-1,1,1)  %maximizer strategies- an index of possible_strategies
s_minimizer_index= 1;%randi(n_strategies-1,1,1) %maximizer strategies- an index of possible_strategies
s_minimizer_binary= dec2bin(s_minimizer_index,n_nodes)-48;
s_maximizer_binary= dec2bin(s_maximizer_index,n_nodes)-48;
n_maximizer_st=1;


tempFeature=feature_pairwise_generator(double(s_maximizer_binary)',word2vecFeatures,0);
lagrangianPotentials_pairwise_maximizer=zeros(n_nodes,n_nodes);
for slice = 1 : n_nodes
    for sl=1:n_nodes
        lagrangianPotentials_pairwise_maximizer(slice,sl)=reshape(tempFeature(slice,sl,:),1,300)*theta_pairwise_ml; %feature_pairwise bala mosalasii hast va
    end
end
lp_pairwise{n_maximizer_st}=(sum(sum(lagrangianPotentials_pairwise_maximizer))); %/n_pairs; %

game_matrix_loss = pdist2(s_minimizer_binary, s_maximizer_binary, 'hamming') ;

while (maxCondition  || minCondition  )
    
    game_matrix_loss=pdist2(s_minimizer_binary,s_maximizer_binary,'hamming');
    lagrangianPotentials_maximizer_nodes=(double(s_maximizer_binary)*lagrangianPotentials_node'); %/n_nodes;
    lagrangianPotentials_maximizer_total=lagrangianPotentials_maximizer_nodes'+[lp_pairwise{1:n_maximizer_st}];
    [p_minimizer,game_value_minimizer]=findMinimizerProbabilities(lagrangianPotentials_maximizer_total,game_matrix_loss,p_maximizer);
    [p_maximizer,game_value_maximizer]=findMaximizerProbabilities(lagrangianPotentials_maximizer_total,game_matrix_loss,p_minimizer);
    [cut_ml,max_best_value]= findBestS_maximizer(s_minimizer_binary,lagrangianPotentials_node,lagrangianPotentials_pairwise,p_minimizer);
    
    if ((sum(ismember(s_maximizer_binary,cut_ml','rows') )>0 )|| (n_maximizer_st > n_solutions))
        
        maxCondition = 0;
        
    else
        
        maxCondition=1;
        s_maximizer_binary = [s_maximizer_binary; cut_ml'];
        n_maximizer_st = n_maximizer_st + 1;
        
        % new lagrangian potentials/
        
        lagrangianPotentials_maximizer_nodes = (double(s_maximizer_binary)*lagrangianPotentials_node'); %/n_nodes;
        tempFeature=feature_pairwise_generator(double(cut_ml)',word2vecFeatures,0);
        
        for slice = 1 : n_nodes
            for sl=1:n_nodes
                lagrangianPotentials_pairwise_maximizer(slice,sl)=reshape(tempFeature(slice,sl,:),1,300)*theta_pairwise_ml;
            end
        end
        lp_pairwise{n_maximizer_st}=(sum(sum(lagrangianPotentials_pairwise_maximizer))); %/n_pairs;
        
        lagrangianPotentials_maximizer_total=lagrangianPotentials_maximizer_nodes'+[lp_pairwise{1:n_maximizer_st}];
        
        
    end
    
    game_matrix_loss = pdist2(s_minimizer_binary, s_maximizer_binary, 'hamming') ;
    
    [p_maximizer,game_value_maximizer] = findMaximizerProbabilities(lagrangianPotentials_maximizer_total,game_matrix_loss,p_minimizer);
    [p_minimizer,game_value_minimizer] = findMinimizerProbabilities(lagrangianPotentials_maximizer_total,game_matrix_loss,p_maximizer);
    
    if(isnan(game_value_maximizer))
        
        break;
    end
    
    
    s_minimizer = findBestS_minimizer(s_maximizer_binary, p_maximizer);
    if (sum(ismember(s_minimizer_binary,s_minimizer','rows'))>0 || n_maximizer_st > n_solutions)
        minCondition = 0;
        
    else
        minCondition=1;
        s_minimizer_binary = [s_minimizer_binary; s_minimizer'];
        
    end
    
    game_matrix_loss = pdist2(s_minimizer_binary, s_maximizer_binary, 'hamming');
    
    [p_minimizer,game_value_minimizer] = findMinimizerProbabilities(lagrangianPotentials_maximizer_total,game_matrix_loss,p_maximizer);
    
    [p_maximizer,game_value_maximizer] = findMaximizerProbabilities(lagrangianPotentials_maximizer_total,game_matrix_loss,p_minimizer);
    
end



