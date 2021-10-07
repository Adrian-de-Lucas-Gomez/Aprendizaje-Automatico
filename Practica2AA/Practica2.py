import numpy as np
from numpy.lib import math
from pandas.io.parsers import read_csv
from matplotlib import pyplot as plt
import scipy.optimize as opt

valores = read_csv("ex2data1.csv", header = None).to_numpy()

admitidos = np.where(valores[:,2] == 0)
afortunados = np.where(valores[:,2] == 1)
plt.scatter(valores[admitidos,0], valores[admitidos,1], marker='o',c='red')
plt.scatter(valores[afortunados,0], valores[afortunados,1], marker='+',c='blue')


def sigmoide(a):
    r = 1 / (1+ np.exp(-a))  
    return r

def coste(theta, X, Y):

    H = sigmoide(np.matmul(X, theta))

    cost = (-1/(len(X))) * (np.dot(Y, np.log(H)) +
                            np.dot((1-Y), np.log(1-H)))

    return cost

def gradiente(theta,XT, Y):
    H = sigmoide(np.matmul(XT, theta))

    grad = (1/len(Y)) * np.matmul(XT.T, H - Y)
    return grad

X = valores[:,:-1]
Y = valores[:,2]
m = np.shape(X)[0]
n = np.shape(X)[1]
X = np.hstack([np.ones([m, 1]),X])

theta = np.array([0,0,0])

# print(np.exp([1,2,3,4]))
# print(valores[:,1].shape)

# print(coste(theta,X,Y))
# print(gradiente(theta,X,Y))

result = opt.fmin_tnc(func = coste, x0 = theta, fprime = gradiente, args = (X,Y))
theta_opt = result[0]
print(theta_opt)

xPintar = valores[admitidos,0]

min_x = min(xPintar)
max_x = max(xPintar)
min_y = theta[0] + theta[1] * min_x
max_y = theta[0] + theta[1] * max_x

plt.plot([min_x, max_x], [min_y, max_y])
#plt.plot([50, 100], [50, 100])
plt.show()
