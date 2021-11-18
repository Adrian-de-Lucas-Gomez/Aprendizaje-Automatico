import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as minimize
import displayData as disp
import checkNNGradients as checkNNG
import pandas as pd

# def parte1():

#     data= loadmat('ex4data1.mat')

#     y= data['y']
#     X = data['X']

#     # sample = np.random.choice(X.shape[0], 100)
#     # disp.displayData(X[sample])
#     # plt.show()

#     weights = loadmat('ex4weights.mat')
#     theta1 = weights['theta1']
#     theta2 = weights['theta2']

#     y_matrix = pd.get_dummies(np.array(y).ravel()).values
#     A=coste(theta1, theta2, X, y_matrix)
#     B = costeReg(theta1, theta2, X, y_matrix, 1)
#     print(B)

#     diff = checkNNG.checkNNGradients(back_propagate, 1)

def parte1():
    datos = loadmat('ex4data1.mat')
    X = datos['X']
    y = datos['y']
    y = np.ravel(y)

    m = len(y)
    input_size= np.shape(X)[1]
    num_hidden = 25
    num_labels = 10

    yAux = (y-1)
    y_onehot = np.zeros((m, num_labels))
    for i in range(m):
        y_onehot[i][yAux[i]] = 1

    # sample = np.random.choice(X.shape[0], 100)
    # disp.displayData(X[sample])
    # plt.show()

    weights = loadmat('ex4weights.mat')
    theta1 = weights['Theta1']
    theta2 = weights['Theta2']

    print("Coste sin regularizar: " + str(coste(X, y_onehot, theta1, theta2))[:5])
    print("Coste regularizado con lambda=1: " + str(costeReg(X, y_onehot, theta1, theta2, 1))[:5])
    print()

    lambo = 1
    print("--Comprobación del gradiente--")
    diff = checkNNG.checkNNGradients(redNeuronalPatras, lambo)
    print("Menor diferencia: " + str(min(diff)))
    print("Mayor diferencia: " + str(max(diff)))
    print()

    #Aprendizaje de los parámetros
    theta1 = pesosAleatorios(input_size, num_hidden)
    theta2 = pesosAleatorios(num_hidden, num_labels)
    params_rn = np.concatenate((np.ravel(theta1), np.ravel(theta2)))
    
    #n_iters = 70
    optimizeResult = minimize.minimize(fun=redNeuronalPatras, x0=params_rn, args=(input_size, num_hidden, num_labels, X, y_onehot, lambo), method='TNC', jac=True, options={'maxiter': 70})
    evaluacion = comprobar(optimizeResult, input_size, num_hidden, num_labels, X, y)
    print("Evaluación del entrenamiento de la red: {}%".format(str(evaluacion*100)[:5]))


def coste(X, Y, theta1, theta2):

	a1, a2, H = redNeuronalPaLante(X, theta1,theta2) #Sacas el resultado

	op1= -1/(len(X))
	op2 = Y * np.log(H)
	op3 = (1-Y) * np.log(1-H)

	cost = op1 * np.sum(op2 + op3)
	return cost

def costeReg(X, Y, theta1, theta2, lambo):
    costeN = coste(X, Y, theta1, theta2)
    costeR = costeN + (lambo/(2*len(X)) * (np.sum(np.square(theta1[:,1:])) + np.sum(np.square(theta2[:,1:]))))
    return costeR


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
    z3 = np.dot(a2, theta2.T) 
    a3 = sigmoide(z3) #Es la hipotesis

    return (a1, a2, a3)

def redNeuronalPatras(params_rn, n_input, n_hidden, n_labels, X, y, lambdita):
    theta1 = np.reshape(params_rn[:n_hidden * (n_input+1)], (n_hidden, (n_input+1)))
    theta2 = np.reshape(params_rn[n_hidden * (n_input+1):], (n_labels, (n_hidden+1)))
    
    m = len(X)
    A1, A2, H = redNeuronalPaLante(X, theta1, theta2)

    Delta1 = np.zeros_like(theta1)
    Delta2 = np.zeros_like(theta2)

    for t in range(m):
        a1t = A1[t, :]
        a2t = A2[t, :]
        ht = H[t, :]
        yt = y[t]

        d3t = ht - yt
        d2t = np.dot(theta2.T, d3t) * (a2t * (1 - a2t))

        Delta1 = Delta1 + np.dot(d2t[1:, np.newaxis], a1t[np.newaxis, :])
        Delta2 = Delta2 + np.dot(d3t[:, np.newaxis], a2t[np.newaxis, :])

    
    #Gradientes
    G1= Delta1 / m
    G2 = Delta2 / m

    #Lambdas
    lambo1 = lambdita * theta1 / m
    lambo2 = lambdita * theta2 / m

    lambo1[:, 0] = 0
    lambo2[:, 0] = 0

    G1 += lambo1
    G2 += lambo2

    gradiente = np.concatenate((np.ravel(G1), np.ravel(G2)))

    #Coste
    coste = costeReg(X, y, theta1, theta2, lambdita)

    return coste, gradiente
    
def comprobar(resOpt, n_input, n_hidden, n_labels, X, y):
    theta1 = np.reshape(resOpt.x[:n_hidden * (n_input + 1)] , (n_hidden, (n_input+1)))
    theta2 = np.reshape(resOpt.x[n_hidden * (n_input + 1):] , (n_labels, (n_hidden+1)))

    A1, A2, H = redNeuronalPaLante(X, theta1, theta2)

    aux = np.argmax(H, axis=1)
    aux += 1

    return np.sum(aux == y) / np.shape(H)[0]

def pesosAleatorios(L_ini, L_out):
    E_ini = 0.12
    return np.random.random((L_out, L_ini + 1)) * (2*E_ini) - E_ini

parte1()