function [X2, out2] = subgradient(S, rho)
%% subgradient Solving Sparse Inverse Covariance Estimation using the subgradient method
%  nonfixed stepsize + continuation + stop criteria

maxITER = 1000; 
epsilon = 1e-8;

n = size(S,1);
X = eye(n);
k=1;
while k < maxITER
   X0 =X;
   g = inv(X0) - S - rho * sign(X0);
   alpha = 0.2;
   % alpha[model1:rho=10,0.1,0.001] =[0.08,0.1,0.1]
   % alpha[model2:rho=10,0.1,0.001] = [0.08,0.2,1e-6]
   L = target(X0,S,rho);
   while target(X0+alpha*g,S,rho)<L
        alpha=0.99*alpha;
   end
   X = X0 + alpha*g;

   if norm(X0-X) < epsilon
        break;
   end
   k = k + 1;
end

X2 = X;
out2 = log(det(X2)) - trace(S*X2) - rho* norm(vec(X2), 1);
end

function L=target(X,S,rho)
    L=log(det(X)) - trace(S*X) - rho* norm(vec(X), 1);
end

