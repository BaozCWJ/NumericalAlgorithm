% function Test_l1_minimized_problems
% min  ||Ax-b||_1 + mu*||x||_1

clear all
% generate data
rng('default')
n = 1024; m = 512;
A = randn(m,n);
u = sprandn(n,1,0.1);
b = A*u;
mu = 1e-2;
x0 = rand(n,1);
% error with cvx_mosek
errfun = @(x1, x2) norm(x1-x2)/(1+norm(x1));



% cvx calling mosek
opts1 = 'mosek'; %modify options
tic; 
[x1, out1] = l1_cvx_mosek(x0, A, b, mu, opts1);
t1 = toc;

% cvx calling gurobi
opts2 = 'gurobi'; %modify options
tic; 
[x2, out2] = l1_cvx_gurobi(x0, A, b, mu, opts2);
t2 = toc;

% Augmented Lagrangian method with proximal gradient
opts3 = []; %modify options
tic; 
[x3, out3] = l1_augmentedLagrangian(x0, A, b, mu, opts3);
t3 = toc;

% ADMM for primal problem
opts4 = []; %modify options
tic; 
[x4, out4] = l1_admm_primal(x0, A, b, mu, opts4);
t4 = toc;

% print comparison results
fprintf('cvx-call-mosek:        time: %5.2f, err-to-cvx-mosek: %3.2e, fval: %15.14f\n', t1, errfun(x1, x1),out1);
fprintf('cvx-call-gurobi:       time: %5.2f, err-to-cvx-mosek: %3.2e, fval: %15.14f\n', t2, errfun(x1, x2),out2);
fprintf('l1_augmentedLagrangian:     time: %5.2f, err-to-cvx-mosek: %3.2e, fval: %15.14f\n', t3, errfun(x1, x3),out3);
fprintf('l1_admm_primal:     time: %5.2f, err-to-cvx-mosek: %3.2e, fval: %15.14f\n', t4, errfun(x1, x4),out4);

