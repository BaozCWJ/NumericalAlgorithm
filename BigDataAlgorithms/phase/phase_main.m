n = 200 ;%20?60?100?200
m = 800;%100?200?400?800
% Data generation
x = randn(n,1) + 1i*randn(n,1); % True Signal
A = 1/sqrt(2)*randn(m,n) + 1i/sqrt(2)*randn(m,n); % measurement matrix
y = abs(A*x).^2; % measurements

% WF solver
max_iter= 2500;                           % Max number of iterations
tau0 = 330;                         % Time constant for step size
opts = [max_iter, tau0];
[z, errs] = phase_WF(A, y, x, opts);
T = max_iter;
% Reuslts
fprintf('Relative error after initialization: %f\n', errs(1))
fprintf('Relative error after %d iterations: %f\n', T, errs(T+1))
 
figure, semilogy(0:T,errs, 'linewidth', 2) 
xlabel('Iteration'), ylabel('Relative error (log10)')
title('Relative error vs. iteration count')
