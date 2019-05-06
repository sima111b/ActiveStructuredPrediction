function minimizer = findBestS_minimizer(s_maximizer_binary,p_maximizer)
global n_nodes;
n_strategies=size(s_maximizer_binary,1); % number of strategies for minimizer
p_marginal_1=zeros(n_nodes,1);
p_marginal_0=zeros(n_nodes,1);
%calculation marginal distribution for every bit
for j=1:n_nodes
    
    p_1= find(s_maximizer_binary(:,j)==1);
    
    p_marginal_1(j)=sum(p_maximizer(p_1));
    
    p_0=find(s_maximizer_binary(:,j)==0);
    
    p_marginal_0(j)=sum(p_maximizer(p_0));
    
end
minimizer=zeros(n_nodes,1);
th=1/n_nodes;
for i=1:n_nodes
    if(p_marginal_1(i)>=0.5)
        minimizer(i)=1;
    end
end
% gameValue=sum(minimizer==1);



