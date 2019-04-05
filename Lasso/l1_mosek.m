function [ x3, out3 ] = l1_mosek(x0, A, b, mu, opts3)
%% l1_mosek Solving Lasso by directly calling mosek

%   Prolem: min 0.5 * ||Ax-b||_2^2 + mu*||x||_1
%   We first reformulate it as a QP problem:
%       min 0.5 * y'* y + mu* (ones(n,1)'* x^+ + ones(n,1)'*x^-)
%   s.t.    A(x^+ - x^-) - y = b
%           x^+ >= 0
%           x^- >= 0
%% Model Setup

[m, n] = size(A);
q = diag([zeros(2*n,1);ones(m,1)]);
c = mu * [ones(2*n,1);zeros(m,1)];
a = sparse([A, -A, -eye(m)]);

blc = b;
buc = b;
blx = [zeros(2*n,1);-inf(m,1)];
bux = [];

%% Optimize the problem

[res] = mskqpopt(q,c,a,blc,buc,blx,bux);
x3 = res.sol.itr.xx(1:n) - res.sol.itr.xx((n+1):2*n);
out3 = res.sol.itr.pobjval;
end