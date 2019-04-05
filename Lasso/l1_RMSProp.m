function [x16, out16] = l1_RMSProp(x0, A, b, mu, opts16)
m = size(A, 1);
n = size(A, 2);
bigmu = 1e5*mu;
epsilon = 1e-8;
alpha =6e-4;
maxIter = 300; 
Ab = A' * b;
AA = A' * A;
x = x0;
r= zeros(n,1);
delta=1e-8;
rho = 0.999999;
while bigmu >= mu
    for k = 1:maxIter      
        x0 = x ;
        g = AA*x0 - Ab + bigmu * sign(x0); 
        r=rho*r+(1-rho)*g.*g;
        x = x- g./(delta+sqrt(r))*alpha;
        if norm(x0-x) < epsilon
            break;
        end
    end
    bigmu = bigmu / 10; 
end
x16 = x;
out16 = 0.5*sum_square(A*x16 - b) + mu * sum(abs(x16));
end

function L=lasso(A,b,x,mu)
L=0.5*sum_square(A*x - b) + mu * sum(abs(x));
end
