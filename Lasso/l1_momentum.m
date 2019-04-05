function [x17, out17] = l1_momentum(x0, A, b, mu, opts17)
m = size(A, 1);
n = size(A, 2);
bigmu = 1e5*mu;
epsilon = 1e-8;
beta = 3e-4;
alpha =1.1*beta;
maxIter = 400;

Ab = A' * b;
AA = A' * A;
x = x0;
v =zeros(n,1);

while bigmu >= mu
    for k = 1:maxIter      
        x0 = x;
        g = AA*x0 - Ab + bigmu * sign(x0); 
        v = alpha*v - beta*g;
        x = x + v;
        if norm(x0-x) < epsilon
            break;
        end
    end
    bigmu = bigmu / 10; 
end
x17 = x;
out17 = 0.5*sum_square(A*x17 - b) + mu * sum(abs(x17));
end
