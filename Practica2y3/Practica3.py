import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize as opt
from scipy.io import loadmat

def Parte1():
    data= loadmat('ex3data1.mat')

    y= data['y']
    X = data['X']

    sample = np.random.choice(X.shape[0], 10)
    plt.imshow(X[sample, :].reshape(-1, 20).T)
    for s in sample:
        print(str(y[s]) + ";")

    # print(yBoolean[0][100])
    # print(yBoolean[1][500])
    # print(yBoolean[9][4990])
    
    Theta = oneVsAll(X, y, 10, 0.1)
    XStacked = np.hstack([np.ones((len(y), 1)),X])
    print('Se ha clasificado correctamente el ', end='')
    print("%.2f" %(precision(XStacked, y.ravel(), Theta)*100), end='')
    print('% de los ejemplos de entrenamiento')

    plt.show()

def Parte2():
    data= loadmat('ex3data1.mat')

    y= data['y']
    X = data['X']
    
    XStacked = np.hstack([np.ones((len(y), 1)),X])
    
    redNeuronal(XStacked, y)

def redNeuronal(X, Y):
    pesos = loadmat('ex3weights.mat')
    theta1 = pesos ['Theta1']
    theta2 = pesos ['Theta2']

    hl2 = theta1.dot(X.T)   #Capa oculta
    act2 = sigmoide(hl2)    #Activación

    act2 = np.vstack([np.ones((len(act2[0]))), act2])

    ol = theta2.dot(act2)   #Capa de salida
    act3 = sigmoide(ol)
    
    print("La red neuronal clasifica correctamente el ", end='')
    print(precisionRedNeuronal(Y, act3) * 100, end='')
    print("% de los ejemplos.")

def oneVsAll(X, y, num_etiquetas, reg):
    
    X_stacked = np.hstack([np.ones((len(y),1)), X])

    ret = np.zeros([num_etiquetas, len(X_stacked[0])])

    for etiqueta in range(num_etiquetas):
        labels = ((y== etiqueta + 1 )* 1 ).ravel()#conseguimos un array de booleanos en la etiqueta que nos iteresa y luego los convertimos a unos 
        ret[etiqueta] = opt.fmin_tnc(func = costeRegularizado, x0 = ret[etiqueta], 
        fprime=gradienteRegularizado,args=(X_stacked,labels,reg))[0]
    
    return ret

def precision(X, Y, Theta):
    n = len(Y)
    predicciones = np.empty(n)
    for i in range(n):
        predicciones[i] = (np.argmax(sigmoide(np.dot(X[i], Theta.T)))+1)
    return np.mean(Y == predicciones)

def precisionRedNeuronal(Y, finalActv):
    n = len(Y)
    predicciones = np.empty(n)
    for i in range(n):
        predicciones[i] = (np.argmax(finalActv[:,i])+1)
    return np.mean(Y.ravel() == predicciones)

def sigmoide(x):
    ## cuidado con x > 50 devuelve 1
    ##
    s = 1 / (1 + np.exp(-x))
    return s

#---------------------------------------------------------------------------------------------------

def coste(theta, X, Y):

	H = sigmoide(np.matmul(X, theta))

	cost = (-1/(len(X))) * (np.dot(np.log(H), Y) +
							np.dot(np.log(1 - H + 1e-6), (1 - Y)))

	return cost


def costeRegularizado(theta, X, Y, lambo):
	m = np.shape(X)[0]
	return coste(theta, X, Y ) + lambo * np.sum(theta**2)/(2*m)


def gradiente(theta, x, y):
    H = sigmoide(np.dot(x, theta))
    return np.dot((H - y), x) / len(y)


def gradienteRegularizado(theta,X, Y, lambo):
	grad = (gradiente(theta,X, Y))
	a = grad[0]
	reg = lambo*theta / len(Y)
	reg[0] = a
	return grad + reg 

#---------------------------------------------------------------------------------------------------



Parte2()
