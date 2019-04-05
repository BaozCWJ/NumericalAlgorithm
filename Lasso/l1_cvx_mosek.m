function [x1, out1] = l1_cvx_mosek(x0, A, b, mu, opts1)
%% l1_cvx_mosek Solving Lasso using CVX by calling calling mosek

n = 1024;
cvx_begin quiet
    cvx_precision high
    cvx_solver(opts1)  
    variable x(n)
    minimize sum((A*x - b).^2)/2 +mu*norm(x,1)
cvx_end
x1 = x;
out1 = cvx_optval;