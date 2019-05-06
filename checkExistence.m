function  check=checkExistence(strategies, newStrategy)
check=0; 
n_strategies=size(strategies,2);
 for i=1:n_strategies
    if isequal(strategies(:,i),newStrategy)
        check=1;
    end
 end
     