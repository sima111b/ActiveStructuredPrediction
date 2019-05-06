
function p_marginal = calcMarginal(node_index, probabilities, strategies)
            p_marginal_0=sum(probabilities(strategies(:, node_index) == 0));
            p_marginal_1=sum(probabilities(strategies(:, node_index) == 1));
            p_marginal = [ p_marginal_0 p_marginal_1];
            p_marginal(p_marginal == 0) = 10e-10;
    end