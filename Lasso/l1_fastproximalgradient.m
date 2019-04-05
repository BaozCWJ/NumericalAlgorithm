function [x10, out10] = l1_fastproximalgradient(x0, A, b, mu, opts10)
%% l1_fastproximalgradient Solving Lasso using the fast proximal gradient method for the primal problem
%  diminishing stepsize + continuation + stop criteria
bigmu = mu*1e5;

maxITER = 60; 
epsilon = 1e-6;

x = x0;
v = x0;
while bigmu >= mu
    k = 1;
    alpha = 3.4e-4;
    while k < maxITER
        if mod(k,50)==0
            alpha=alpha*0.9;
        end
        theta=2/(1+k);
        v0=v;
        x0 = x;
        y=(1-theta)*x0+theta*v0;
        g=A' * (A * y - b);
        x=proxgradient(y-alpha*(1+1/sqrt(k))*g,alpha*(1+1/sqrt(k))*bigmu);
        v=x0+(x-x0)/theta;
        if norm(x0-x) < epsilon
            break;
        end
        k = k + 1;
    end
    bigmu = bigmu / 10;
end
x10 = x;
out10 = 0.5*sum_square(A*x10 - b) + mu * sum(abs(x10));
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