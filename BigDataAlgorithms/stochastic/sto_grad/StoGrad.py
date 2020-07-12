import numpy as np
import math

def loss(X,Y,w,l1_norm):
    n = len(Y)
    l=l1_norm*sum(abs(w))
    temp = -Y*X.dot(w)
    l += sum(np.log(1+np.exp(temp)))/n
    return l


def grad(X,Y,w,l1_norm):
    n = len(Y)
    g = l1_norm*np.sign(w)
    exp = np.exp(-Y*X.dot(w))
    temp = (-exp)/(1+exp)
    g += (temp*Y).dot(X)/n
    return g


def Adagrad(X,Y,l1_norm,epsilon,alpha,batch_size,max_iter):
    loss_his = []
    dataset_size = len(Y)
    w = np.ones(len(X[0]))/len(X[0])
    r = np.zeros(len(X[0]))
    delta = 1e-8
    for ii in range(max_iter):
        np.random.seed(ii)
        batch_idx = np.random.choice(range(dataset_size), size=batch_size, replace=False)
        X_batch = X[batch_idx,:]
        Y_batch = Y[batch_idx]
        g = grad(X_batch,Y_batch,w,l1_norm)
        r += g*g
        w = w - (alpha/(delta+np.sqrt(r)))*g
        l = loss(X,Y,w,l1_norm)
        if ii%100 ==0: print(ii,l)
        loss_his.append(l)
        if len(loss_his)>1 and abs(loss_his[-2]-loss_his[-1])<epsilon:
            return loss_his
    return loss_his

def Adam(X,Y,l1_norm,epsilon,alpha,beta_1,beta_2,batch_size,max_iter):
    loss_his = []
    dataset_size = len(Y)
    w = np.ones(len(X[0]))/len(X[0])
    r = np.zeros(len(X[0]))
    s = np.zeros(len(X[0]))
    delta = 1e-10
    for ii in range(max_iter):
        np.random.seed(ii)
        batch_idx = np.random.choice(range(dataset_size), size=batch_size, replace=False)
        X_batch = X[batch_idx,:]
        Y_batch = Y[batch_idx]
        g = grad(X_batch,Y_batch,w,l1_norm)
        s = beta_1*s+(1-beta_1)*g
        r = beta_2*r+(1-beta_2)*g*g
        s_hat = s/(1-pow(beta_1,ii+1))
        r_hat = r/(1-pow(beta_2,ii+1))
        w = w - alpha/(delta+np.sqrt(r_hat))*s_hat
        l = loss(X,Y,w,l1_norm)
        if ii%100 ==0: print(ii,l)
        loss_his.append(l)
        if len(loss_his)>1 and abs(loss_his[-2]-loss_his[-1])<epsilon:
            return loss_his
    return loss_his

def RMSProp(X,Y,l1_norm,epsilon,alpha,beta,batch_size,max_iter):
    loss_his = []
    dataset_size = len(Y)
    w = np.ones(len(X[0]))/len(X[0])
    r = np.zeros(len(X[0]))
    delta = 1e-6
    for ii in range(max_iter):
        batch_idx = np.random.choice(range(dataset_size), size=batch_size, replace=False)
        X_batch = X[batch_idx,:]
        Y_batch = Y[batch_idx]
        g = grad(X_batch,Y_batch,w,l1_norm)
        r = beta*r + (1-beta)*g*g
        w = w - (alpha/(delta+np.sqrt(r)))*g
        l = loss(X,Y,w,l1_norm)
        if ii%100 ==0: print(ii,l)
        loss_his.append(l)
        if len(loss_his)>1 and abs(loss_his[-2]-loss_his[-1])<epsilon:
            return loss_his
    return loss_his
