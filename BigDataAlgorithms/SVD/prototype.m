function [error, sing_val, sing_vec] = prototype(A, r, q)
% Implement Prototype algorithm in the paer:
% "N. Halko, P. G. Martinsson, and J. A. Tropp, Finding Structure with 
% Randomness: Probabilistic Algorithms for Constructing Approximate Matrix 
% Decompositions,SIAM Rev., 53(2), 217288." 
%
% Input: 
% A: the m by n matrix
% r: the number of largest singular values
% q: the number of power iteration
%
% Output:
% error: the actual error, ||A-QQ'A||
% sing_val: the r-largest approximate singular values of A
% sing_vec: r singular values corresponding the singular vectors

[m, n] = size(A);

% Stage A
Omega=randn(n,2*r);

Y = ((A*A')^q)*A*Omega;  % m*2r
[Q,~] = qr(Y);
Q = Q(:,1:2*r); % m * 2r

% Stage B
B = Q'*A;  % 2r*n
[Uhat,Sigma,Vhat]=svd(B); % B = Uhat*Sigma*Vhat'
U = Q*Uhat; % m*2r

% output
error = norm(A-Q*Q'*A);
sing_val_2r = vec(diag(Sigma));
sing_val = sing_val_2r(1:r);
sing_vec = U(:,1:r);
sing_vec1 = Vhat(:, 1:r); % The singular vectors corresponding the r-largest singular values
end
