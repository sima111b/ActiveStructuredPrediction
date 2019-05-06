function [sample_value,bestIndicies] = sampleEvaluation ( p_maximizer, s_maximizer)

global n_solicitation;
n_nodes = size(s_maximizer, 2);
nodes_values= zeros(n_nodes, 1);

for i = 1 : n_nodes
    p_node= calcMarginal(i, p_maximizer, s_maximizer);
    H_node = -sum(p_node.* log(p_node));
    node_IM = 0;
    for j = 1 : n_nodes
        if i ~= j
            p_neighborNode= calcMarginal(j,p_maximizer,s_maximizer);
            H_neighborNode =  -sum(p_neighborNode .* log(p_neighborNode));
            p_joint = calcJoint( i, j,p_maximizer,s_maximizer);
            H_joint = -sum(p_joint .* log(p_joint));
            IM_joint = H_node + H_neighborNode - H_joint;
            node_IM = node_IM + IM_joint;
        end
    end
    nodes_values(i, 1) = (node_IM + H_node)/n_nodes;
end

sample_value=sum(nodes_values)/n_nodes;

[sortedValues,sortedIndicies]=sort(nodes_values,'descend');
bestIndicies=sortedIndicies(1:n_solicitation);
[max_value, ~] = max(sortedValues(:, 1));
maxIndex = find(sortedValues == max_value);


if size(maxIndex, 1) >= n_solicitation
    
    m=size(maxIndex, 1);
    order = randperm ( m);
    tempIndicies = maxIndex(order);
    bestIndicies=tempIndicies(1:n_solicitation);
end

end

