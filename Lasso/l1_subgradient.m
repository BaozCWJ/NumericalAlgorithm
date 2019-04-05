function [x6, out6] = l1_subgradient(x0, A, b, mu, opts6)
%% l1_subgradient Solving Lasso using the subgradient method
%  nonfixed stepsize + continuation + stop criteria
bigmu = mu*1e5;

maxITER = 1000; 
epsilon = 1e-8;

x = x0;
while bigmu >= mu
    k = 1;
    alpha = 3.4e-4;
    while k < maxITER
        x0 = x;
        g = A' * (A * x - b)  + bigmu * sign(x);
        x = x - alpha*(1+2/sqrt(k))* g;
        if mod(k,50)==0
            alpha=alpha*0.8;
        end
        if norm(x0-x) < epsilon
            break;
        end
        k = k + 1;
    end
    bigmu = bigmu / 10;
end
x6 = x;
out6 = 0.5*sum_square(A*x6 - b) + mu * sum(abs(x6));
end

