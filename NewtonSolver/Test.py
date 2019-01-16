import time
from TestFunction import *
from NewtonSolver import *


epsilon=1e-8

print('=====Powell Badly Scaled Funciton=====')
x0=np.array([0,1])

print('Damped Newton')
s=time.time()
#Damped_Newton(Powell_func,Powell_grad,Powell_Hess,x0,epsilon,alpha0=10)
print('time=',time.time()-s)

print('Refine Newton')
s=time.time()
#Refine_Newton(Powell_func,Powell_grad,Powell_Hess,x0,epsilon,alpha0=10)
print('time=',time.time()-s)

print('SR1 Quasi Newton')
s=time.time()
SR1_quasi_Newton(Powell_func,Powell_grad,x0,epsilon,alpha0=1)
print('time=',time.time()-s)

print('DFP Quasi Newton')
s=time.time()
DFP_quasi_Newton(Powell_func,Powell_grad,x0,epsilon,alpha0=2)
print('time=',time.time()-s)

print('BFGS Quasi Newton')
s=time.time()
#BFGS_quasi_Newton(Powell_func,Powell_grad,x0,epsilon,alpha0=1)
print('time=',time.time()-s)



'''
print('=====Discrete boundary value Function=====')
D_num=10
x1=np.zeros(D_num)
h=1/(D_num+1)
for i in range(D_num):
    x1[i]=(i+1)*h*((i+1)*h-1)

print('Damped Newton')
s=time.time()
Damped_Newton(Discrete_func,Discrete_grad,Discrete_Hess,x1,epsilon,alpha0=10)
print('time=',time.time()-s)

print('Refine Newton')
s=time.time()
Refine_Newton(Discrete_func,Discrete_grad,Discrete_Hess,x1,epsilon,alpha0=10)
print('time=',time.time()-s)

print('SR1 Quasi Newton')
s=time.time()
SR1_quasi_Newton(Discrete_func,Discrete_grad,x1,epsilon,alpha0=1)
print('time=',time.time()-s)

print('DFP Quasi Newton')
s=time.time()
DFP_quasi_Newton(Discrete_func,Discrete_grad,x1,epsilon,alpha0=2)
print('time=',time.time()-s)

print('BFGS Quasi Newton')
s=time.time()
BFGS_quasi_Newton(Discrete_func,Discrete_grad,x1,epsilon,alpha0=1)
print('time=',time.time()-s)



print('=====Extended Powell sigular Function=====')
E_num=12
x2=np.zeros(E_num)
for i in range(E_num//4):
    x2[4*i]=3
    x2[4*i+1]=-1
    x2[4*i+3]=1

print('Damped Newton')
s=time.time()
Damped_Newton(Extended_func,Extended_grad,Extended_Hess,x2,epsilon,alpha0=10)
print('time=',time.time()-s)

print('Refine Newton')
s=time.time()
Refine_Newton(Extended_func,Extended_grad,Extended_Hess,x2,epsilon,alpha0=10)
print('time=',time.time()-s)

print('SR1 Quasi Newton')
s=time.time()
SR1_quasi_Newton(Extended_func,Extended_grad,x2,epsilon,alpha0=1)
print('time=',time.time()-s)

print('DFP Quasi Newton')
s=time.time()
DFP_quasi_Newton(Extended_func,Extended_grad,x2,epsilon,alpha0=2)
print('time=',time.time()-s)

print('BFGS Quasi Newton')
s=time.time()
BFGS_quasi_Newton(Extended_func,Extended_grad,x2,epsilon,alpha0=1)
print('time=',time.time()-s)
'''