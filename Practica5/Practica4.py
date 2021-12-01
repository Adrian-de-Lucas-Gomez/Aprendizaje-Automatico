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
    lRate = 1
    theta = np.array([1, 1]).ravel()
    Xstacked = np.c_[np.ones((len(X),1)),X]
    c = Cost(theta, Xstacked, y, lRate)
    print("Coste con λ=1 y θ=[1; 1]: " , c)
    g = Gradiente(theta, Xstacked, y, lRate)
    print("Gradiente entre ",min(g) ," y ", max(g))

    lRate = 0
    minFun = minimize(fun = Cost, x0 = theta, args=(Xstacked, y, lRate))

    # plt.figure()
    # plt.scatter(X, y, marker = '$☺$', c="green", s = 100, linewidths=0.1)
    # lineX = np.linspace(min(X), max(X), 1000)
    # lineY = np.c_[np.ones((1000,1)), lineX].dot(minFun.x)
    # plt.plot(lineX, lineY, '-', c="purple")
    # plt.show()

    #Parte2

    data = loadmat('ex5data1.mat')
    # data.keys()
    X, y = data['X'], data['y']
    Xval, yval = data['Xval'], data['yval']
    Xtest, ytest = data['Xtest'], data['ytest']
    lRate = 1
    theta = np.array([1, 1]).ravel()
    Xstacked = np.c_[np.ones((len(X),1)),X]

    m = len(y)
    errorEntre = np.empty(m)
    errorValida = np.empty(m)

    Xvalidar = np.c_[np.ones((len(Xval), 1)), Xval]
    theta = np.array([1,1])
    lRate = 0

    for i in range(1,m+1):
        fmin = minimize(fun = Cost, x0 = theta, args = (Xstacked[:i], y[:i], lRate))
        errorEntre[i-1] = CalculaError(fmin.x, Xstacked[:i], y[:i])
        errorValida[i-1] = CalculaError(fmin.x, Xvalidar, yval)

    # plt.figure()
    # plt.plot(range(1, m+1), errorEntre, c = "red")
    # plt.plot(range(1, m+1), errorValida, c = "green")
    # plt.show()

    #PARTE 3

    gradoPolinomio = 8
    mediaX = np.mean(X, axis = 0)
    sigmaX = np.std(X, axis = 0)
    Xnorm = normalizar(genPolynomial(X,gradoPolinomio), mediaX, sigmaX)
    Xnorm = np.c_[np.ones((len(Xnorm), 1)), Xnorm]
    lRate = 0
    theta = np.zeros(gradoPolinomio + 1)
    
    fmin = minimize(fun = Cost, x0 = theta, args = (Xnorm, y, lRate))
        

    plt.figure()
    plt.scatter(X, y, marker = '$♥$', c="orange", s = 100, linewidths=0.1)
    graphX = normalizar(np.arrange(min(X), max(X), 0.05), mediaX, sigmaX)
    graphY = np.c_[np.ones((len(Xnorm), 1)), Xnorm]
    graphY = graphY.dot(fmin.x) 

    plt.plot(graphX, graphY, '-', c= "purple")
    plt.show()

def Hypothesys(X, theta):
    return X.dot(theta)

def Cost(theta, X, y, lRate):
    return ((Hypothesys(X, theta) - y.ravel()).dot(Hypothesys(X, theta) - y.ravel())/(len(y)*2)
            + lRate/(2*len(y))*(np.square(theta[1:])).sum())

def Gradiente(theta, X, Y, lRate):
    stacked = np.hstack(([0], theta[1:]))
    op1 = X.T.dot(Hypothesys(X, theta) - Y.ravel())
    op2 = len(Y) + lRate * stacked/len(Y)
    return (op1 / op2)

def CalculaError(theta, X, Y):
    op1 = (Hypothesys(X,theta) - Y.ravel())
    op2 = op1.dot(Hypothesys(X, theta) - Y.ravel())
    return op2 / (len(Y) * 2)

def genPolynomial(X,p):
    ret = np.empty((len(X), p))
    for i in range(p):
        ret[:,i] = (X**(i+1)).ravel()
    return ret

def normalizar(X, media, sigma):
    return (X-media)/sigma

Parte1()
