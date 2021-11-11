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
    B = costeReg(theta1, theta2, X, y_matrix, 1)
    print(B)

    diff = checkNNG.checkNNGradients(backprop, 1)


def sigmoideDer(x):
    return( sigmoide(x) * (1.0 - sigmoide(x)))

def backprop(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, reg ):

    theta1 = params_rn[:(num_ocultas * (num_entradas + 1))].reshape(num_ocultas,(num_entradas + 1))
    theta2 = params_rn[(num_ocultas * (num_entradas + 1)):].reshape(num_etiquetas,(num_ocultas + 1))
    
    y_matrix = pd.get_dummies(np.array(y).ravel()).values

    a1, z2, a2, z3, a3 = redNeuronalPaLante(theta1, theta2, X)
    costeR = costeReg(theta1, theta2, X, y_matrix, 1)

    d3 = a3.T - y_matrix
    d2 = theta2[:,1:].T.dot(d3.T)*sigmoideDer(z2)

    delta1 = d2.dot(a1)
    delta2 = d3.T.dot(a2)

    theta1Copy = np.c_[np.zeros((theta1.shape[0],1)),theta1[:,1:]]
    theta2Copy = np.c_[np.zeros((theta2.shape[0],1)),theta2[:,1:]]

    m = len(X)

    theta1Grad = delta1/m + (theta1Copy* reg)/m
    theta2Grad = delta2/m + (theta2Copy* reg)/m

    return(costeR, np.r_[theta1Grad.ravel(), theta2Grad.ravel()])

def sigmoide(x):
    s = 1 / (1 + np.exp(-x))
    return s

def redNeuronalPaLante(X, theta1, theta2):
    
    xSize = X.shape[0]
    # Capa de entrada
    a1 = np.hstack([np.ones((xSize,1)), X])    
    # Capa oculta    
    z2 = np.dot(a1, theta1.T)
    a2 = np.hstack([np.ones((xSize,1)),sigmoide(z2)])
    # Capa de salida
    z3 = np.dot(a2, theta2.T) 
    a3 = sigmoide(z3) 
    return (a1, z2, a2, z3, a3)

def coste(theta1, theta2, X, Y):

	H = redNeuronalPaLante(X, theta1,theta2)[-1] #Sacas el resultado
	cost = (-1/(len(X))) * np.sum((np.log(H.T)*Y + np.log(1-H).T*(1-Y)))
	return cost

def costeReg(theta1, theta2, X, Y, lambo):
    costeN = coste(theta1, theta2, X, Y)
    costeR = costeN + (lambo/(2*len(X)) * (np.sum(np.square(theta1[:,1:])) + np.sum(np.square(theta2[:,1:]))))
    return costeR

parte1()