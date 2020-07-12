function [x2, out2] = l1_cvx_gurobi(x0, A, b, mu, opts2)
%% l1_cvx_mosek Solving Lasso using CVX by calling calling gurobi

n = 1024;
cvx_begin quiet
    cvx_precision high
    cvx_solver(opts2)
    variable x(n)
    minimize norm(A*x-b,1) +mu*norm(x,1)
cvx_end
x2 = x;
out2 = cvx_optval;