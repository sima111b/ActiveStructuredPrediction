

function [cut,max_value]= findBestS_maximizer(s_minimizer_binary,lagrangianPotentials_nodes,lagrangianPotentials_pairwise,p_minimizer)
global n_solutions;
global n_nodes;
gr=mincutGraphCreator(s_minimizer_binary,p_minimizer,lagrangianPotentials_nodes,lagrangianPotentials_pairwise);

A=sparse(gr(2:n_nodes+1,2:n_nodes+1));
T=sparse(n_nodes,2);
T(:,1)=gr(1,2:n_nodes+1);
T(:,2)=gr(n_nodes+2,2:n_nodes+1);
[max_value,cut] = maxflow(A,T);
end
