import os
import numpy as np
from sklearn.datasets import fetch_openml,fetch_covtype

class FLAGS(object):
        def __init__(self):
                self.batch_size = 500
                self.epsilon = 1e-4
                self.l1_norm = 10 #10 1 0.1 0.001
                self.lr = 1e-3 # MNIST: 1e-4, Covertype:1e-3
                self.max_iter = 10000
                self.beta_1 = 0.9
                self.beta_2 = 0.999

os.makedirs("./mnist",exist_ok=True)
os.makedirs("./covertype",exist_ok=True)


def get_dataset(name):
        if name =='MNIST':
                dataset = fetch_openml('mnist_784',data_home = "./mnist")
                X = dataset['data']
                Y = np.array(dataset['target'],dtype=np.float32)
                Y = 2*(Y%2)-1
                #print(dataset['data'].shape)
                #print(dataset['target'].shape)
                return X,Y
        elif name == "Covertype":
                dataset = fetch_covtype(data_home = "./covertype")
                X = dataset['data']
                Y = np.array(dataset['target'],dtype=np.float32)
                index = (Y==2)
                Y[index] = 1.0
                Y[~index] = -1.0
                #print(dataset['data'].shape)
                #print(dataset['target'].shape)
                return X,Y
        else:
                raise('Not such dataset')
