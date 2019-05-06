
function graph=mincutGraphCreator(minimizer_strategies,p_minimizer,lagrangianPotentials_nodes,lagrangianPotentials_pairwise)
global n_nodes;
n_minimizer_strategies=size(minimizer_strategies,1); % number of strategies for minimizer
p_marginal_1=zeros(n_nodes,1);
p_marginal_0=zeros(n_nodes,1);

%calculation marginal distribution for every label
for j=1:n_nodes
    
    p_1= find(minimizer_strategies(:,j)==1);
    
    p_marginal_1(j)=sum(p_minimizer(p_1));
    
    p_0=find(minimizer_strategies(:,j)==0);
    
    p_marginal_0(j)=sum(p_minimizer(p_0));
    
end


% bit_1= (2^n_bits)/2; % 50% of the bits are zero and 50% are 1, so it would have thes loss 1 for 50% of the total strategies.
%
sink_edges=(-lagrangianPotentials_nodes'+(p_marginal_0)); % sink edges are connected to y_check=0
source_edges=(p_marginal_1)';   % source edges are connected to y_check=1
graph=ones(n_nodes+2,n_nodes+2);
%graph=ones(n_bits,n_bits);
graph(1,2:n_nodes+1)=source_edges(:);
graph(2:n_nodes+1,1)=source_edges(:);
graph(n_nodes+2,2:n_nodes+1)=sink_edges(:);
graph(2:n_nodes+1,n_nodes+2)=sink_edges(:);
graph=graph+abs(min(0,min(min(graph))));
% lagrangianPotentials_pairwise=triu(lagrangianPotentials_pairwise)';
graph(2:n_nodes+1,2:n_nodes+1)=lagrangianPotentials_pairwise(:,:);
%graph=graph;


