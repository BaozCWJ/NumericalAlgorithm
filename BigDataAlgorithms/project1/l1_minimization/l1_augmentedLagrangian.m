function [x3, out3] = l1_augmentedLagrangian(x0, A, b, mu, opts3)
%% Solving l1_minimization using the augmented Lagrangian method for primal problem
bigmu = mu*1e4;
m = size(A, 1);
n = size(A, 2);

maxITER =17; 
epsilon = 1e-6;
z = zeros(m+n,1);
z(1:n)=x0;
beta = 1.2e-2;
alpha = (-1+sqrt(5))/2;
y = zeros(m,1);
AI=[A -eye(m)];

%% iteration
while bigmu >= mu
    k = 1;
    while k < maxITER
        z0 = z;
        y0=y;
        z=argminL(A,b,y0,z0,beta,bigmu,7);
        y=y0+(AI*z-b)*alpha*beta*(1+1/sqrt(k));
        if norm(z0(1:n)-z(1:n)) < epsilon
            break;
        end
        k = k + 1;
    end
    bigmu = bigmu / 10;
end

x3 =  z(1:n);
out3 = sum(abs(A*x3 - b)) + mu * sum(abs(x3));
end


function L=lagrangian(A,b,y,z,beta,mu)
    AI = [A -eye(m)];
    L=mu*sum(abs(z(1:n)))+sum(abs(z(n+1:2*n)))+beta*0.5*sum_square(AI*z - b)+y'*(AI*z-b)
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

function z=argminL(A,b,y0,z0,beta,mu,maxITER)
m = size(A, 1);
n = size(A, 2);
k=1;
AI = [A -eye(m)];
gamma = 2e-1;
while k < maxITER
    z=z0;
    grad=AI'*y0+beta*AI'*(AI*z-b);
    gamma_t = gamma*(1/k);
    temp = z-gamma_t*grad;
    z0(1:n) = proxgradient(temp(1:n),mu*gamma_t);
    z0(n+1:n+m) = proxgradient(temp(n+1:n+m),gamma_t);
    k = k+1;

end
z=z0;
end