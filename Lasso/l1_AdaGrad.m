function [x14, out14] = l1_AdaGrad(x0, A, b, mu, opts14)
m = size(A, 1);
n = size(A, 2);
bigmu = 1e5*mu;
epsilon = 1e-8;
alpha =0.8;
maxIter = 400;
Ab = A' * b;
AA = A' * A;
x = x0;
r= zeros(n,1);
delta=1e-8;
while bigmu >= mu
    for k = 1:maxIter      
        x0=x;
        g = AA*x0 - Ab + bigmu * sign(x0); 
        r=r+g.*g;
        x = x - g./(delta+sqrt(r))*alpha;
        if norm(x0-x) < epsilon
            break;
        end
    end
    bigmu = bigmu / 10; 
end
x14 = x;
out14 = 0.5*sum_square(A*x14 - b) + mu * sum(abs(x14));
end
