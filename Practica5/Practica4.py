import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize

def Parte1():
    data = loadmat('ex5data1.mat')
    # data.keys()
    X, y = data['X'], data['y']
    Xval, yval = data['Xval'], data['yval']
    Xtest, ytest = data['Xtest'], data['ytest']

def Hipothesys(X, theta):
    return X.dot(theta)

def Cost(theta, X, y, l_rate):
    return ((Hipothesys(X, theta) - y.ravel()).dot(Hipothesys(X, theta) - y.ravel())/(len(y)*2)
            + l_rate/(2*len(y))*(np.square(theta[1:])).sum())


Parte1()
