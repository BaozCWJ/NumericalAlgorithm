function [x12, out12] = l1_admm_dual(x0, A, b, mu, opts12)
%% Solving Lasso using the ADMM for dual problem
bigmu = mu*1e5;
m = size(A, 1);
n = size(A, 2);

maxITER =30; 
epsilon = 1e-6;
AA=A*A';
gamma=(-1+sqrt(5))/2;
x = x0;
beta = 1e-2;
z = zeros(n,1);
[L, D] = ldl(eye(m) + AA* beta);
%% iteration
while bigmu >= mu
    k = 1;
    while k < maxITER
        x0=x;
        z0=z;
        y = L'\(D\(L\(b -  A *(x - beta*z0))));
        temp=x0/beta+A'*y;
        z=sign(temp).*min(abs(temp),bigmu);
        x=x0+gamma*(A'*y-z)*beta;
        if norm(x0-x) < epsilon
            break;
        end
        k = k + 1;
    end
    bigmu = bigmu / 10;
end

x12 =  x;
out12 = 0.5*sum_square(A*x12 - b) + mu * sum(abs(x12));
end



