%generate random matrix A
clear all;
m = 2048;
n = 512;
p = 20;
A = randn(m,p)*randn(p,n); % m * n
% the reference values of singular values from matlab svd
d = svd(A);

% set r = {5,10,15,20}
[error1,s1, U1] = prototype(A, 5, 1);
[error2,s2, U2] = prototype(A, 10, 1);
[error3,s3, U3] = prototype(A, 15, 1);
[error4,s4, U4] = prototype(A, 20, 1);


%show the difference between prototype and matlab svd
figure
plot(1:5, d(1:5), '+', 1:5, s1, 'o', 'LineWidth', 2)
legend('svd', 'prototype')
title('the 5-largest singular values')
figure
plot(1:10, d(1:10), '+', 1:10, s2,'o', 'LineWidth', 2)
legend('svd', 'prototype')
title('the 10-largest singular values')
figure
plot(1:15, d(1:15), '+',1:15, s3,'o', 'LineWidth', 2)
legend('svd', 'prototype')
title('the 15-largest singular values')
figure
plot(1:20, d(1:20,1), '+',1:20, s4,'o', 'LineWidth', 2)
legend('svd', 'prototype')
title('the 20-largest singular values')
