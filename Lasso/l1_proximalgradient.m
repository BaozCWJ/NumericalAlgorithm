function [x9, out9] = l1_proximalgradient(x0, A, b, mu, opts9)
%% l1_proximalgradient Solving Lasso using the proximal gradient method for the primal problem
%  diminishing stepsize + continuation + stop criteria
bigmu = mu*1e5;

maxITER = 400; 
epsilon = 1e-7;

x = x0;

while bigmu >= mu
    k = 1;
    alpha = 3.4e-4;
    while k < maxITER
        if mod(k,50)==0
            alpha=alpha*0.9;
        end
        x0 = x;
        g = A' * (A * x0 - b);
        y = x0 - alpha*(1+1/sqrt(k))* g;
        x=proxgradient(y,alpha*(1+1/sqrt(k))*bigmu);
        if norm(x0-x) < epsilon
            break;
        end
        k = k + 1;
    end
    bigmu = bigmu / 10;
end
x9 = x;
out9 = 0.5*sum_square(A*x9 - b) + mu * sum(abs(x9));
end

function prox=proxgradient(x,lambda);
filter1=x>lambda;
filter2=x<(-lambda);
filter3=abs(x)<=lambda;
x(filter1)=x(filter1)-lambda;
x(filter2)=x(filter2)+lambda;
x(filter3)=x(filter3)*0;
prox=x;
end