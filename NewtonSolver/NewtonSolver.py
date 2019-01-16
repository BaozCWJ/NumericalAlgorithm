import numpy as np
import scipy.sparse as spr
from scipy.sparse.linalg import spsolve
from sympy import *


#substitute symbolic expression to numerical result
def SymToNum(expr,x,x0):
    n=len(x0)
    s=expr
    for i in range(n):
        s=s.subs({x[i]:x0[i]})
    return s

#backtracking line search(using Armijo condition)
def LineSearch(func,grad,x,d,alpha0,alpha_min,rho):
    alpha=alpha0
    decay=0.9
    g=grad(x)
    while func(x+alpha*d)>func(x)+rho*alpha*g.dot(d) and alpha>alpha_min:
        alpha=alpha*decay
    return alpha

#Damped Newton via linesearch
def Damped_Newton(func,grad,Hess,x0,epsilon,alpha0):
    k=0
    alpha_min=1e-3
    rho=1e-3
    if len(x0)>5:
        x = symarray('x', len(x0))
        H_sym = Hess(x0)
        while func(x0)>epsilon:
            H=spr.csc_matrix(np.array(SymToNum(H_sym,x,x0)).astype(np.float32))
            d = -spsolve(H, grad(x0))
            t=LineSearch(func,grad,x0,d,alpha0,alpha_min,rho)
            x0=x0+t*d
            k=k+1
    else:
        while func(x0) > epsilon:
            d = -np.linalg.solve(Hess(x0), grad(x0))
            t = LineSearch(func, grad, x0, d,alpha0,alpha_min,rho)
            x0 = x0 + t * d
            k = k + 1
    print('iteration=',k)
    print('soluton:',x0)
    print('optimal value=',func(x0))

#Altenative dircetion Newton
def Refine_Newton(func,grad,Hess,x0,epsilon,alpha0):
    k=0
    alpha_min=1e-3
    rho=1e-3
    if len(x0)>5:
        x = symarray('x', len(x0))
        H_sym = Hess(x0)
        while func(x0)>epsilon:
            g=grad(x0)
            H=spr.csc_matrix(np.array(SymToNum(H_sym,x,x0)).astype(np.float32))
            d = -spsolve(H, g)
            if g.dot(d)>0:
                d=-g
            t=LineSearch(func,grad,x0,d,alpha0,alpha_min,rho)
            x0=x0+t*d
            k=k+1
    else:
        while func(x0) > epsilon:
            g=grad(x0)
            d = -np.linalg.solve(Hess(x0), g)
            if g.dot(d)>0:
                d=-g
            t = LineSearch(func, grad, x0, d,alpha0,alpha_min,rho)
            x0 = x0 + t * d
            k = k + 1
    print('iteration=',k)
    print('soluton:',x0)
    print('optimal value=',func(x0))


#Symmetric Rank One Quasi Newton Method
def SR1_quasi_Newton(func,grad,x,epsilon,alpha0):
    k=0
    alpha_min=1e-8
    rho=1/3
    H=np.eye(len(x))
    g=grad(x)
    while func(x) > epsilon:
        x0=x
        g0=g
        d = -H.dot(g)
        if g.dot(d)>0:
            d=-g
        t = LineSearch(func, grad, x, d,alpha0,alpha_min,rho)
        x = x + t * d
        g=grad(x)
        y=g-g0
        s=x-x0
        u=s-H.dot(y)
        H=H+np.tensordot(u,u,axes=0)/u.dot(y)
        k = k + 1
    print('iteration=',k)
    print('soluton:',x)
    print('optimal value=',func(x))


#Davidon-Fletcher-Powell Quasi Newton
def DFP_quasi_Newton(func,grad,x,epsilon,alpha0):
    k=0
    alpha_min=1e-8
    rho=1/3
    H=np.eye(len(x))
    g=grad(x)
    while func(x) > epsilon:
        x0=x
        g0=g
        d = -H.dot(g)
        if g.dot(d)>0:
            d=-g
        t = LineSearch(func, grad, x, d,alpha0,alpha_min,rho)
        x = x + t * d
        g=grad(x)
        y=g-g0
        s=x-x0
        H=H+np.tensordot(s,s,axes=0)/s.dot(y)-H.dot(np.tensordot(y,y,axes=0)).dot(H)/y.dot(np.dot(H,y))
        k = k + 1
    print('iteration=',k)
    print('soluton:',x)
    print('optimal value=',func(x))

#Broyden-Fletcher-Goldfarb-Shanno Quasi Newton
def BFGS_quasi_Newton(func,grad,x,epsilon,alpha0):
    k=0
    alpha_min=1e-8
    rho=1/3
    H=np.eye(len(x))
    g=grad(x)
    while func(x) > epsilon:
        x0=x
        g0=g
        d = -H.dot(g)
        if g.dot(d)>0:
            d=-g
        t = LineSearch(func, grad, x, d,alpha0,alpha_min,rho)
        x = x + t * d
        g=grad(x)
        y=g-g0
        s=x-x0
        H=H+(1+y.dot(np.dot(H,y))/s.dot(y))*np.tensordot(s,s,axes=0)/s.dot(y)-(H.dot(np.tensordot(y,s,axes=0))+np.tensordot(s,y,axes=0).dot(H))/s.dot(y)
        k = k + 1
        if np.linalg.norm(g,2)<=epsilon:
            break
    print('iteration=',k)
    print('soluton:',x)
    print('optimal value=',func(x))