function [x4, out4] = l1_admm_primal(x0, A, b, mu, opts4)
%% Solving Lasso using the ADMM with linearization for primal problem
bigmu = mu*1e4;
m = size(A, 1);
n = size(A, 2);

maxITER =40; 
epsilon = 1e-6;
x=x0;
y=zeros(m,1);
beta = 1e-2;
alpha = (-1+sqrt(5))/2;
u = zeros(m,1);

%% iteration
while bigmu >= mu
    k = 1;
    while k < maxITER
        x0=x;
        y0=y;
        u0=u;
        x=argminL_x(A,b,x0,y0,u0,beta,bigmu,8);
        y=argminL_y(A,b,x,y0,u0,beta,bigmu,8);
        u=u0+(A*x-y-b)*alpha*beta*(1+1/k);
        if norm(x0-x) < epsilon
            break;
        end
        k = k + 1;
    end
    bigmu = bigmu / 10;
end

x4 =  x;
out4 = sum(abs(A*x4 - b)) + mu * sum(abs(x4));
end


function L=lagrangian(A,b,x,y,u,beta,mu)
    L=mu*sum(abs(x))+sum(abs(y))+beta*0.5*sum_square(A*x-y - b)+u'*(A*x-y-b)
end

function prox=proxgradient(x,lambda)
filter1=x>lambda;
filter2=x<(-lambda);
filter3=abs(x)<=lambda;
x(filter1)=x(filter1)-lambda;
x(filter2)=x(filter2)+lambda;
x(filter3)=x(filter3)*0;
prox=x;
end

function x=argminL_x(A,b,x0,y0,u0,beta,mu,maxITER)
k=1;
gamma = 2e-1;
while k < maxITER
    x=x0;
    grad=A'*u0+beta*A'*(A*x-y0-b);
    gamma_t = gamma*(1/k);
    temp = x-gamma_t*grad;
    x0 = proxgradient(temp,mu*gamma_t);
    k = k+1;

end
x=x0;
end

function y=argminL_y(A,b,x0,y0,u0,beta,mu,maxITER)
k=1;
gamma = 2e-1;
while k < maxITER
    y=y0;
    grad=u0+beta*(y+b-A*x0);
    gamma_t = gamma*(1/k);
    temp = y-gamma_t*grad;
    y0 = proxgradient(temp,gamma_t);
    k = k+1;

end
y=y0;
end
