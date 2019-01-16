import numpy as np
from sympy import *


#Powell Badly Scaled Funciton
def Powell_func(x):
    if len(x)==2:
        r1=10000*x[0]*x[1]-1
        r2=np.exp(-x[0])+np.exp(-x[1])-1.0001
        return r1**2+r2**2
    else:
        print('inappropriate input size')

def Powell_grad(x):
    if len(x)==2:
        g0=2e4*(1e4*x[0]*x[1]-1)*x[1]-2*np.exp(-x[0])*(np.exp(-x[0])+np.exp(-x[1])-1.0001)
        g1=2e4*(1e4*x[0]*x[1]-1)*x[0]-2*np.exp(-x[1])*(np.exp(-x[0])+np.exp(-x[1])-1.0001)
        return np.array([g0,g1])
    else:
        print('inappropriate input size')

def Powell_Hess(x):
    if len(x)==2:
        H=np.zeros((2,2))
        H[0,0]=1e8*x[1]**2 + (-1.0001 + np.exp(-x[1]) + np.exp(-x[0]))*np.exp(-x[0]) + np.exp(-2*x[0])
        H[1,1]=1e8*x[0]**2 + (-1.0001 + np.exp(-x[1]) + np.exp(-x[0]))*np.exp(-x[1]) + np.exp(-2*x[1])
        H[0,1]=4e8*x[1]*x[0] - 2e4 + 2*np.exp(-x[0])*np.exp(-x[1])
        return H+H.T
    else:
        print('inappropriate input size')


#Discrete boundary value Function
def Discrete_func(x):
    n=len(x)
    h=1/(1+n)
    r=np.zeros(n)
    r[0]=2*x[0]-x[1]+h**2*pow(x[0]+h+1,3)/2
    r[n-1] = 2 * x[n-1] - x[n-2] + h ** 2 * pow(x[n-1] + n*h + 1, 3) / 2
    for i in range(1,n-1):
        r[i]=2*x[i]-x[i-1]-x[i+1]+h**2*pow(x[i]+(i+1)*h+1,3)/2
    return r.dot(r)

def Discrete_grad(x):
    n=len(x)
    h=1/(1+n)
    r=np.zeros(n)
    g=np.zeros(n)
    r[0]=2*x[0]-x[1]+h**2*pow(x[0]+h+1,3)/2
    r[n-1] = 2 * x[n-1] - x[n-2] + h ** 2 * pow(x[n-1] + n*h + 1, 3) / 2
    for i in range(1,n-1):
        r[i]=2*x[i]-x[i-1]-x[i+1]+h**2*pow(x[i]+(i+1)*h+1,3)/2
    g[0]=r[0]*(4+3*h**2*pow(x[0]+h+1,2))-2*r[1]
    g[n-1]=r[n-1]*(4+3*h**2*pow(x[n-1]+n*h+1,2))-2*r[n-2]
    for i in range(1,n-1):
        g[i]=r[i]*(4+3*h**2*pow(x[i]+(i+1)*h+1,2))-2*r[i-1]-2*r[i+1]
    return g

def Discrete_Hess(x0):
    n=len(x0)
    h=1/(1+n)
    x=symarray('x',n)
    r = symarray('r', n)
    r[0]= 2 * x[0] - x[1] + h ** 2 * pow(x[0] + h + 1, 3) / 2
    r[n-1] = 2 * x[n - 1] - x[n - 2] + h ** 2 * pow(x[n - 1] + n * h + 1, 3) / 2
    for i in range(1, n - 1):
        r[i] = 2 * x[i] - x[i - 1] - x[i + 1] + h ** 2 * pow(x[i] + (i + 1) * h + 1, 3) / 2
    H=hessian(r.dot(r),x)
    return H

#Extended Powell sigular Function
def Extended_func(x):
    n=len(x)
    if n%4==0:
        r = np.zeros(n)
        k=n//4
        for i in range(k):
            r[4*i]=x[4*i]+10*x[4*i+1]
            r[4*i+1]=sqrt(5)*(x[4*i+2]-x[4*i+3])
            r[4*i+2]=(x[4*i+1]-2*x[4*i+2])**2
            r[4*i+3]=sqrt(10)*(x[4*i]-x[4*i+3])**2
        return r.dot(r)
    else:
        print('inappropriate input size')

def Extended_grad(x):
    n=len(x)
    if n%4==0:
        g=np.zeros(n)
        k=n//4
        for i in range(k):
            g[4*i]=2*(x[4*i]+10*x[4*i+1])+40*pow(x[4*i]-x[4*i+3],3)
            g[4*i+1]=20*(x[4*i]+10*x[4*i+1])+4*pow(x[4*i+1]-2*x[4*i+2],3)
            g[4*i+2]=10*(x[4*i+2]-x[4*i+3])-8*pow(x[4*i+1]-2*x[4*i+2],3)
            g[4*i+3]=-10*(x[4*i+2]-x[4*i+3])-40*pow(x[4*i]-x[4*i+3],3)
        return g
    else:
        print('inappropriate input size')

def Extended_Hess(x0):
    n=len(x0)
    if n%4==0:
        x = symarray('x', n)
        r = symarray('r', n)
        k=n//4
        for i in range(k):
            r[4*i]=x[4*i]+10*x[4*i+1]
            r[4*i+1]=sqrt(5)*(x[4*i+2]-x[4*i+3])
            r[4*i+2]=(x[4*i+1]-2*x[4*i+2])**2
            r[4*i+3]=sqrt(10)*(x[4*i]-x[4*i+3])**2
        H=hessian(r.dot(r),x)
        return H
    else:
        print('inappropriate input size')