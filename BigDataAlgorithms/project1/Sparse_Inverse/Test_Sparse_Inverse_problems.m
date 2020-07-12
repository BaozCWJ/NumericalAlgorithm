% function Test_lSparse_Inverse_problems
% max logdet(X)-Tr(SX)-rho||X||_1

clear all; 
rng('default')
n = 30;
rho = 0.1; %10, 0.1, 0.001
P = 30; %30 60, 90 ,120, 200  model2 isn't sensitive to P
model = 2; % Choose one model to generate S

if model == 1
    S = model1(n);
else
    S = model2(n, P);
end



tic;
[X1, out1] = cvx_solver(S, rho);
t1 = toc;
dual_gap1 = n-trace(S*X1)  - rho*norm(vec(X1), 1);

tic;
[X2, out2] = subgradient(S, rho);
dual_gap2 = n-trace(S*X2)  - rho*norm(vec(X2), 1);
t2 = toc;


fprintf('cvx:        time: %5.2f, rho: %3.3e, fval: %5.4f, dual gap: %5.14f\n', t1, rho,out1,dual_gap1);
fprintf('subgradient:       time: %5.2f, rho: %3.3e, fval: %5.4f, dual gap: %5.14f\n', t2, rho,out2,dual_gap2);





