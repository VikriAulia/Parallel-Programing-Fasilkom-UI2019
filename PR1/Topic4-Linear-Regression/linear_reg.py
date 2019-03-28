import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import random as rnd
from mpi4py import MPI
from datetime import datetime

# Initialize data.
def init_data():
    df = pd.read_csv('winequality-white.csv', header=None)
    numpy_matrix = df.values
    feature = numpy_matrix[1:, :11]
    target = numpy_matrix[1:, 11]
    X = feature.astype(np.float)
    Y = target.astype(np.float)
    return X, Y

X, Y = init_data()
Y = np.array([Y]).transpose()
comm = MPI.COMM_WORLD
procid = comm.Get_rank()
size = comm.Get_size()
numworkers = size - 1
theta = np.zeros((X.shape[1], 1)) 

# initialize theta
def init_theta(n):
    theta = []
    for i in range(n):
        theta.append(rnd.random())
    return np.array(theta)

# linear regression using "mini-batch" gradient descent 
# function to compute hypothesis / predictions 
def hypothesis(X, theta):
    return np.dot(X, theta) 

# function to compute gradient of error function w.r.t. theta 
def gradient(X, y, theta): 
    h = hypothesis(X, theta) 
    grad = np.dot(X.transpose(), (h - y)) 
    return grad     

# function to compute the error for current values of theta 
def cost(X, y, theta): 
    h = hypothesis(X, theta) 
    J = np.dot((h - y).transpose(), (h - y)) 
    J /= 2
    return J[0] 

# function to create a list containing mini-batches 
def create_mini_batches(X, y, batch_size): 
    mini_batches = [] 
    data = np.hstack((X, y)) 
    
    np.random.shuffle(data) 
    n_minibatches = data.shape[0] // batch_size 
    i = 0
  
    for i in range(n_minibatches + 1): 
        mini_batch = data[i * batch_size:(i + 1)*batch_size, :] 
        X_mini = mini_batch[:, :-1] 
        Y_mini = mini_batch[:, -1].reshape((-1, 1)) 
        mini_batches.append((X_mini, Y_mini)) 
    if data.shape[0] % batch_size != 0: 
        mini_batch = data[i * batch_size:data.shape[0]] 
        X_mini = mini_batch[:, :-1] 
        Y_mini = mini_batch[:, -1].reshape((-1, 1)) 
        mini_batches.append((X_mini, Y_mini)) 
    return mini_batches 


print(procid)

 
batch_size = 100
learning_rate = 0.001

if procid == 0:
    errlist = []
    start = datetime.now()
    
    for i in range(numworkers):
        mini_batch = create_mini_batches(X, Y, batch_size)
        comm.send(mini_batch, dest=i+1, tag=1)

    for i in range(numworkers):
        err = comm.recv(source=i+1, tag=1)
        errlist.append(err)

    then = datetime.now()
    print("time: ", (then - start).total_seconds())
    

if procid >= 1:
    error_list = []
    minbatch = comm.recv(source=0, tag=1)
    
    for mb in minbatch: 
        X_mini, y_mini = mb 
        theta = theta - learning_rate * gradient(X_mini, y_mini, theta) 
        error_list.append(cost(X_mini, y_mini, theta))
    comm.send(error_list, dest=0, tag=1)


# theta, error_list = gradientDescent(X, Y) 
# print("Bias = ", theta[0]) 
# print("Coefficients = ", theta[1:]) 
