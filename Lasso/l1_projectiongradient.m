function [x5, out5] = l1_projectiongradient(x0, A, b, mu, opts5)
%% Solving Lasso using the projection gradient method
z0 = [max(x0,0);max(-x0,0)];
[~,n] = size(A);
bigmu = mu * 1e5;
epison = 1e-4;
lr_l = 1e-6; % lower bound of learning rate
lr_u = 1;    % upper bound of learning rate
lr=1e-4;     % initial learning rate
maxIter = 300;
% preprocess
Ab = A' * b;
AA = A'* A;
c = mu * ones(2*n,1) - [Ab; -Ab]; % coefficients of linear part
%% iteration
while bigmu >= mu
    cur_c = bigmu * ones(2*n,1) - [Ab; -Ab]; % cache coefficients of linear part
% initialize gradient
    AAz0 = AA* (z0(1:n)-z0((n+1):2*n));
    g0 = [AAz0;-AAz0] + cur_c;
    z = z0;
    g = g0;
    k = 1;
    while norm(max(z-g , 0) - z) > epison && k < maxIter
% update z
        z0 = z;
        z = max(z - lr*g, 0);    
% gradient
        g0 = g;
        AAz = AA* (z(1:n)-z((n+1):2*n));
        g = [AAz;-AAz] + cur_c;
% BB step size
        y = g - g0;
        s = z - z0;
        BB = (s'*s) / (s'*y); 
        if s'*y <=0
            lr = lr_u;
        else
            lr = max(lr_l, min(BB, lr_u));
        end
        k = k + 1;
    end
    bigmu = bigmu / 10; 
end
x5 =  z(1:n)-z((n+1):2*n);
out5 = 0.5*sum_square(A*x5 - b) + mu * sum(abs(x5));
end
