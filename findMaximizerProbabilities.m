function [p,game_value] = findMaximizerProbabilities(lagrangianPotentials_total,loss,p_minimizer)
% timeout=300;
n_repeat=size(loss,1);
temp_lagrangian_potentials=repmat(lagrangianPotentials_total,n_repeat,1);
scoreMatrix=loss-temp_lagrangian_potentials;
gameMatrix=double(scoreMatrix');
neg = min(min(gameMatrix)); % this method requires all the values are positive
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
game_value= [p_minimizer'*scoreMatrix] * p;
% if (game_value ~= v)
%     v
% end
end



%%
function x = gurobiWrapper(scoreMatrix)
[m, n] = size(scoreMatrix);

clear model;
% c'*x; A*x > b
model.obj = ones(1, m); % c
model.A = sparse(scoreMatrix'); % A
model.rhs = ones(n, 1); % b
model.sense = '>';
% default lower bounds are 0's
% default upper bounds are infinites
% default is minimization
clear params;
% % params.timeLimit = timeout;
params.outputFlag = 0;

result = gurobi(model, params);
if(~strcmp(result.status, 'OPTIMAL'))
    error(['Error! status=' result.status]);
end
x = result.x;
end
