% function Test_l1_regularized_problems
% min 0.5 ||Ax-b||_2^2 + mu*||x||_1

clear all
% generate data
rng('default')
n = 1024; m = 512;
A = randn(m,n);
u = sprandn(n,1,0.1);
b = A*u;
mu = 1e-3;
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

% call mosek directly
opts3 = []; %modify options
tic; 
[x3, out3] = l1_mosek(x0, A, b, mu, opts3);
t3 = toc;

% call gurobi directly
opts4 = []; %modify options
tic; 
[x4, out4] = l1_gurobi(x0, A, b, mu, opts4);
t4 = toc;

% % other approaches
%projection gradient method
opts5 = []; %modify options
tic; 
[x5, out5] = l1_projectiongradient(x0, A, b, mu, opts5);
t5 = toc;

%subgradient method
opts6 = []; %modify options
tic; 
[x6, out6] = l1_subgradient(x0, A, b, mu, opts6);
t6 = toc;

%smoothed model
opts7 = []; %modify options
tic; 
[x7, out7] = l1_smoothgradient(x0, A, b, mu, opts7);
t7 = toc;

%fast gradient smoothed model
opts8 = []; %modify options
tic; 
[x8, out8] = l1_fastsmoothgradient(x0, A, b, mu, opts8);
t8 = toc;


%proximal gradient
opts9 = []; %modify options
tic; 
[x9, out9] = l1_proximalgradient(x0, A, b, mu, opts9);
t9 = toc;

%fast proximal gradient
opts10 = []; %modify options
tic; 
[x10, out10] = l1_fastproximalgradient(x0, A, b, mu, opts10);
t10 = toc;

%Augmented Lagrangian method for dual problem
opts11 = []; %modify options
tic; 
[x11, out11] = l1_augmentedLagrangian_dual(x0, A, b, mu, opts11);
t11 = toc;

%ADMM for dual problem
opts12 = []; %modify options
tic; 
[x12, out12] = l1_admm_dual(x0, A, b, mu, opts12);
t12 = toc;

%ADMM with linearization for primal problem
opts13 = []; %modify options
tic; 
[x13, out13] = l1_admm_primal(x0, A, b, mu, opts13);
t13 = toc;

%AdaGrad
opts14 = []; %modify options
tic; 
[x14, out14] = l1_AdaGrad(x0, A, b, mu, opts14);
t14 = toc;

%Adam
opts15 = []; %modify options
tic; 
[x15, out15] = l1_Adam(x0, A, b, mu, opts15);
t15 = toc;

%RMSProp
opts16 = []; %modify options
tic; 
[x16, out16] = l1_RMSProp(x0, A, b, mu, opts16);
t16 = toc;

%momentum
opts17 = []; %modify options
tic; 
[x17, out17] = l1_momentum(x0, A, b, mu, opts17);
t17 = toc;


% print comparison results
fprintf('cvx-call-mosek:        time: %5.2f, err-to-cvx-mosek: %3.2e, fval: %15.14f\n', t1, errfun(x1, x1),out1);
%fprintf('cvx-call-gurobi:       time: %5.2f, err-to-cvx-mosek: %3.2e, fval: %15.14f\n', t2, errfun(x1, x2),out2);
%fprintf('call-mosek:            time: %5.2f, err-to-cvx-mosek: %3.2e, fval: %15.14f\n', t3, errfun(x1, x3),out3);
%fprintf('call-gurobi:           time: %5.2f, err-to-cvx-mosek: %3.2e, fval: %15.14f\n', t4, errfun(x1, x4),out4);
%fprintf('l1_projectiongradient: time: %5.2f, err-to-cvx-mosek: %3.2e, fval: %15.14f\n', t5, errfun(x1, x5),out5);
%fprintf('l1_subgradient:        time: %5.2f, err-to-cvx-mosek: %3.2e, fval: %15.14f\n', t6, errfun(x1, x6),out6);
%fprintf('l1_smoothgradient:     time: %5.2f, err-to-cvx-mosek: %3.2e, fval: %15.14f\n', t7, errfun(x1, x7),out7);
%fprintf('l1_fastsmoothgradient:     time: %5.2f, err-to-cvx-mosek: %3.2e, fval: %15.14f\n', t8, errfun(x1, x8),out8);
%fprintf('l1_proximalgradient:     time: %5.2f, err-to-cvx-mosek: %3.2e, fval: %15.14f\n', t9, errfun(x1, x9),out9);
%fprintf('l1_fastproximalgradient:     time: %5.2f, err-to-cvx-mosek: %3.2e, fval: %15.14f\n', t10, errfun(x1, x10),out10);
%fprintf('l1_augmentedLagrangian_dual:     time: %5.2f, err-to-cvx-mosek: %3.2e, fval: %15.14f\n', t11, errfun(x1, x11),out11);
%fprintf('l1_admm_dual:     time: %5.2f, err-to-cvx-mosek: %3.2e, fval: %15.14f\n', t12, errfun(x1, x12),out12);
%fprintf('l1_admm_primal:     time: %5.2f, err-to-cvx-mosek: %3.2e, fval: %15.14f\n', t13, errfun(x1, x13),out13);
fprintf('l1_AdaGrad:     time: %5.2f, err-to-cvx-mosek: %3.2e, fval: %15.14f\n', t14, errfun(x1, x14),out14);
fprintf('l1_Adam:     time: %5.2f, err-to-cvx-mosek: %3.2e, fval: %15.14f\n', t15, errfun(x1, x15),out15);
fprintf('l1_RMSProp:     time: %5.2f, err-to-cvx-mosek: %3.2e, fval: %15.14f\n', t16, errfun(x1, x16),out16);
fprintf('l1_momentum:     time: %5.2f, err-to-cvx-mosek: %3.2e, fval: %15.14f\n', t17, errfun(x1, x17),out17);


%{
 figure()
 subplot(3,2,1)
 plot(x1(1:1000));
 title('cvx-mosek')
 subplot(3,2,2)
 plot(x2(1:1000));
 title('cvx-gurobi')
 subplot(3,2,3)
 plot(x3(1:1000));
 title('mosek')
 subplot(3,2,4)
 plot(x4(1:1000));
 title('gurobi')
 subplot(3,2,5)
 plot(x5(1:1000));
 title('projection gradient')
 subplot(3,2,6)
 plot(x6(1:1000));
 title('subgradient')
%}