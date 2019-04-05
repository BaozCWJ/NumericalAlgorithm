function [x7, out7] = l1_smoothgradient(x0, A, b, mu, opts7);
%% Solving smoothed Lasso using the gradient method for the smoothed primal problem
bigmu = mu*1e5;

maxITER = 1000; 
epsilon = 1e-7;
lambda=2.5e-8;
x = x0;
while bigmu >= mu
    k = 1;
    alpha = 3.4e-4;
    while k < maxITER
        x0 = x;
        g = A' * (A * x - b)  + bigmu * smoothgradient(x,lambda);
        x = x - alpha* g;
        if mod(k,40)==0
            alpha=alpha*0.9;
        end
        if norm(x0-x) < epsilon
            break;
        end
        k = k + 1;
    end
    bigmu = bigmu / 10;
end
x7 = x;
out7 = 0.5*sum_square(A*x7 - b) + mu * sum(abs(x7));

end

function smg=smoothgradient(x,lambda);
filter=abs(x)<lambda;
x(filter)=x(filter)/lambda;
x(~filter)=sign(x(~filter));
smg=x;
end

