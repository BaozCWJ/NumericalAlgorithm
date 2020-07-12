from StoGrad import *
from data import *
import matplotlib.pyplot as plt

flag=FLAGS()

def plot_loss(loss_his1,loss_his2,l1_norm,dataset):
    n1 = len(loss_his1)
    n2 = len(loss_his2)
    plt.figure()
    alg1,=plt.plot(range(n1),loss_his1)
    alg2,=plt.plot(range(n2),loss_his2)
    plt.legend([alg1,alg2],['Adagrad','Adam'])
    plt.xlabel('iterations')
    plt.ylabel('training loss')
    plt.title(r"%s  $\lambda$=%f"%(dataset,l1_norm))
    plt.ylim(-0.1,max(max(loss_his1),max(loss_his2))*1.2)
    plt.show()


#MNIST Covertype
dataset="Covertype"
X_mnist,Y_mnist = get_dataset(dataset)

loss_his1=Adagrad(X_mnist,Y_mnist,flag.l1_norm,flag.epsilon,flag.lr,flag.batch_size,flag.max_iter)
loss_his2=Adam(X_mnist,Y_mnist,flag.l1_norm,flag.epsilon,flag.lr,flag.beta_1,flag.beta_2,flag.batch_size,flag.max_iter)
plot_loss(loss_his1,loss_his2,flag.l1_norm,dataset)
print(len(loss_his1),loss_his1[-1])
print(len(loss_his2),loss_his2[-1])
