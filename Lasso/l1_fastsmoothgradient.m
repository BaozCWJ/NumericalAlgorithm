function [x8, out8] = l1_fastsmoothgradient(x0, A, b, mu, opts8)
%% l1_fastsmoothgradient Solving Lasso using the fast gradient method for the smoothed primal problem
%  diminishing stepsize + continuation + stop criteria
bigmu = mu*1e5;

maxITER = 800; 
epsilon = 1e-7;
lambda=1e-8;
x = x0;
v = x0;
while bigmu >= mu
    k = 1;
    alpha = 3e-4;
    while k < maxITER
        theta=2/(1+k);
        v0=v;
        x0 = x;
        y=(1-theta)*x0+theta*v0;
        g = A' * (A * y - b)  + bigmu * smoothgradient(y,lambda);
        x = y - alpha* g;
        if mod(k,50)==0
            alpha=alpha*0.8;
        end
        if norm(x0-x) < epsilon
            break;
        end
        k = k + 1;
        v=x0+(x-x0)/theta;
    end
    bigmu = bigmu / 10;
end
x8 = x;
out8 = 0.5*sum_square(A*x8 - b) + mu * sum(abs(x8));
end

function smg=smoothgradient(x,lambda);
filter=abs(x)<lambda;
x(filter)=x(filter)/lambda;
x(~filter)=sign(x(~filter));
smg=x;
end
