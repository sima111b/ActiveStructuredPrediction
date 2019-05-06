
function [p,game_value] = findMinimizerProbabilities(lagrangianPotentials_total,loss,p_maximizer)
% timeout=300;
n_repeat=size(loss,1);
temp=repmat(lagrangianPotentials_total,n_repeat,1);
scoreMatrix=loss-temp;
gameMatrix=double(-scoreMatrix);
neg = min(min(gameMatrix)); % this method requires all the values to be positive
if (neg <= 0)
    gameMatrix = gameMatrix + (1 - neg);
end
% x = gurobiWrapper(gameMatrix, timeout);
x = gurobiWrapper(gameMatrix);
if (x==0)
   loss
   temp
end
v = 1 / sum(x);
p = x * v;
if(neg <= 0)
    v = v - (1 - neg);
end
v=-v;
game_value=(p' * (scoreMatrix*p_maximizer));

end



%%
function x = gurobiWrapper(gameMatrix)
[m, n] = size(gameMatrix);
clear model;
% c'*x; A*x > b
model.obj = ones(1, m); % c
model.A = sparse(gameMatrix'); % A
model.rhs = ones(n, 1); % b
model.sense = '>';
% default lower bounds are 0's
% default upper bounds are infinites
% default is minimization
clear params;
% params.timeLimit = timeout;
params.outputFlag = 0;

result = gurobi(model, params);
if(~strcmp(result.status, 'OPTIMAL'))
    error(['Error! status=' result.status]);
end
x = result.x;
end


