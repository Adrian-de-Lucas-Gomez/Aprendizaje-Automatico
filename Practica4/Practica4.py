import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize
import displayData as disp
import checkNNGradients as checkNNG
import pandas as pd

def parte1():

    data= loadmat('ex4data1.mat')

    y= data['y']
    X = data['X']

    # sample = np.random.choice(X.shape[0], 100)
    # disp.displayData(X[sample])
    # plt.show()

    weights = loadmat('ex4weights.mat')
    theta1 = weights['Theta1']
    theta2 = weights['Theta2']

    y_matrix = pd.get_dummies(np.array(y).ravel()).values
    A=coste(theta1, theta2, X, y_matrix)
    D=0
    

def sigmoide(x):
    s = 1 / (1 + np.exp(-x))
    return s

def redNeuronalPaLante(X, theta1, theta2):
    
    xSize = X.shape[0]
    # Capa de entrada
    a1 = np.hstack([np.ones((xSize,1)), X])    
    # Capa oculta    
    z2 = theta1.dot(a1.T)
    a2 = np.hstack([np.ones((xSize,1)),sigmoide(z2.T)])
    # Capa de salida
    z3 = theta2.dot(a2.T)
    a3 = sigmoide(z3) 
    return (a1, z2, a2, z3, a3)

def coste(theta1, theta2, X, Y):

	H = redNeuronalPaLante(X, theta1,theta2)[-1] #Sacas el resultado
	cost = (-1/(len(X))) * np.sum((np.log(H.T)*Y + np.log(1-H).T*(1-Y)))
	return cost
    
parte1()