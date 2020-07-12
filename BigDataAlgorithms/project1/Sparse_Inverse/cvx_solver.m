% cvx solver
function [X1, out1] = cvx_solver(S, rho)

% OUTPUT: X1: the solution
%         out: the optimality conditions (need to derive the dual problem)
% s = size(X0);
n = length(S);
% cvx_solver mosek
cvx_begin quiet
    variable X(n,n) semidefinite
    maximize log_det(X) - trace(S*X) - rho* norm(vec(X), 1)
cvx_end
X1 = X;
out1 = log(det(X1)) - trace(S*X1) - rho* norm(vec(X1), 1);
end