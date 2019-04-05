function [x13, out13] = l1_admm_primal(x0, A, b, mu, opts13)
%% Solving Lasso using the ADMM with linearization for primal problem
bigmu = mu*1e5;
m = size(A, 1);
n = size(A, 2);
AA=A'*A;
Ab=A'*b;
beta = 200;
maxITER =150; 
epsilon = 1e-8;
gamma=(-1+sqrt(5))/2;
x = x0;
z = zeros(n,1);
u=0;
%% iteration
while bigmu >= mu
    k = 1;
    while k < maxITER
        x0=x;
        z0=z;
        u0=u;
        alpha=1e-1;
        grad=(AA*x-Ab+beta*(x-z0+u0));
        L=lagrangian(A,b,x,z0,u0,beta);
        while lagrangian(A,b,x-alpha*grad,z0,u0,beta)>L + alpha * grad' * grad * 0.1
            alpha=0.5*alpha;
        end
        x=x-alpha*grad;
        z=proxgradient(x+u0,bigmu/beta);
        u=u0+gamma*(x-z);
        if norm(x0-x) < epsilon
            break;
        end
        k = k + 1;
    end
    bigmu = bigmu / 10;
end

x13 =  x;
out13 = 0.5*sum_square(A*x13 - b) + mu * sum(abs(x13));
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

function L=lagrangian(A,b,x,z,u,beta)
    L=0.5*sum_square(A*x - b)+0.5*beta*sum_square(x-z+u);
end


