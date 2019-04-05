function [x11, out11] = l1_augmentedLagrangian_dual(x0, A, b, mu, opts11)
%% Solving Lasso using the augmented Lagrangian method for dual problem
bigmu = mu*1e5;
m = size(A, 1);
n = size(A, 2);

maxITER =18; 
epsilon = 1e-6;
gamma=(-1+sqrt(5))/2;
x = x0;
beta = 1e-2;
y = zeros(m,1);
%% iteration
while bigmu >= mu
    k = 1;
    while k < maxITER
        x0 = x;
        y0=y;
        y=argminL(A,b,y0,x0,beta,bigmu,3);
        temp=x0/beta+A'*y;
        z=sign(temp).*min(abs(temp),bigmu);
        x=x0+gamma*(A'*y-z)*beta;
        if norm(x0-x) < epsilon
            break;
        end
        k = k + 1;
    end
    bigmu = bigmu / 10;
end

x11 =  x;
out11 = 0.5*sum_square(A*x11 - b) + mu * sum(abs(x11));
end


function [x] = softThreshold(x0, mu)
    x = sign(x0) .* max(abs(x0) - mu, 0);
end

function L=lagrangian(A,b,y,x,beta,mu)
    temp=softThreshold(A'*y+x/beta,mu);
    L=-b'*y+0.5*y'*y+0.5*beta*temp'*temp;
end

function y=argminL(A,b,y0,x,beta,mu,maxITER)
m = size(A, 1);
L=lagrangian(A,b,y0,x,beta,mu);
k=1;
while k < maxITER
    temp=softThreshold(A'*y0+x/beta, mu);
    grad=y0 - b + beta*A * temp;
    direc=grad;
    alpha=1;
    L=lagrangian(A,b,y0,x,beta,mu);
    while lagrangian(A,b,y0-alpha*direc,x,beta,mu)>L - alpha * direc' * grad * 0.1
        alpha=0.5*alpha;
    end
    y0=y0-alpha*direc;
    k=k+1;
end
y=y0;
end