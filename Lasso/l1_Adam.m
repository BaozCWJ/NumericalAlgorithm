function [x15, out15] = l1_Adam(x0, A, b, mu, opts15)
m = size(A, 1);
n = size(A, 2);
bigmu = 1e5*mu;
epsilon = 1e-8;
maxIter = 200;
Ab = A' * b;
AA = A' * A;
x = x0;
r= zeros(n,1);
v =zeros(n,1);
s=v;
delta=1e-8;
alpha =8e-2;
rho1=0.95;
rho2 = 0.999;

while bigmu >= mu
    for k = 1:maxIter      
        x0 = x;
        g = AA*x0 - Ab + bigmu * sign(x0); 
        s=rho1*s+(1-rho1)*g;
        r=rho2*r+(1-rho2)*g.*g;
        s_hat=s/(1-rho1^k);
        r_hat=r/(1-rho2^k);
        x = x -s_hat./(sqrt(r_hat)+delta)*alpha;
        if norm(x0-x) < epsilon
            break;
        end
    end
    bigmu = bigmu / 10; 
end
x15 = x;
out15 = 0.5*sum_square(A*x15 - b) + mu * sum(abs(x15));
end
