function [x4, out4] = l1_gurobi(x0, A, b, mu, opts4)
%% l1_mosek Solving Lasso by directly calling gurobi

%   Prolem: min 0.5 * ||Ax-b||_2^2 + mu*||x||_1
%   We first reformulate it as a QP problem:
%       min 0.5 * (A(x^+ - x^-)-b)'* (A(x^+ - x^-)-b) + mu* (ones(n,1)'* x^+ + ones(n,1)'*x^-)
%   s.t.    x^+ >= 0
%           x^- >= 0
%% Model Setup

[m, n] = size(A);
% set up Q
At = [A -A];
model.Q = 0.5*sparse(At'*At);

% set up linear part
c = mu*ones(2*n,1)- (b'*At)';
model.obj =c;
P =eye(2*n);
model.A = sparse(P);
l1 = zeros(2*n,1);
model.rhs =full(l1);
model.sense = '>';

params.method = 2;

% optimize the problem
res = gurobi(model, params);
% show the primal solution
x4 = res.x(1:n,1)-res.x(n+1:2*n,1);
out4 = 0.5 * sum_square(A * x4 - b) + mu * norm(x4, 1);
end

