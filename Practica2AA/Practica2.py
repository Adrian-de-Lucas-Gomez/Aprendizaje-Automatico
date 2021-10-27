import numpy as np
from numpy.lib import math
from pandas.io.parsers import read_csv
from matplotlib import pyplot as plt
import scipy.optimize as opt
from sklearn.preprocessing import PolynomialFeatures

def evaluaLogistica(X,Y,theta):
	b = sigmoide( np.dot(X,theta))>=0.5
	correctos = np.sum((sigmoide( np.dot(X,theta))>=0.5)==Y)
	print(f"porcentaje de casos predichos correctmente {correctos / np.shape(X)[0] * 100}%")

def pinta_frontera_recta(X, Y, theta):
	x1_min, x1_max = X[:, 1].min(), X[:, 1].max()
	x2_min, x2_max = X[:, 2].min(), X[:, 2].max()

	xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
	np.linspace(x2_min, x2_max))

	h = sigmoide(np.c_[np.ones((xx1.ravel().shape[0], 1)),
	xx1.ravel(),
	xx2.ravel()].dot(theta))
	h = h.reshape(xx1.shape)

	# el cuarto parámetro es el valor de z cuya frontera se
	# quiere pintar
	plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='green')
	#plt.show()


def sigmoide(a):
	r = 1 / (1+ np.exp(-a))  
	return r

def costeRegularizado(theta, X, Y, lambo):

	H = sigmoide(np.matmul(X, theta))

	cost = (-1/(len(X))) * (np.dot(Y, np.log(H)) +
							np.dot((1-Y), np.log(1-H))) +((lambo/2*(len(X))*(theta**2).sum()))


def coste(theta, X, Y):

	H = sigmoide(np.matmul(X, theta))

	cost = (-1/(len(X))) * (np.dot(Y, np.log(H)) +
							np.dot((1-Y), np.log(1-H)))

	return cost

def gradiente(theta,XT, Y):
	H = sigmoide(np.matmul(XT, theta))

	grad = (1/len(Y)) * np.matmul(XT.T, H - Y)
	return grad

def gradienteRegularizado(theta,XT, Y, lambo):
	H = sigmoide(np.matmul(XT, theta))

	grad = (1/len(Y)) * np.matmul(XT.T, H - Y) + (lambo/2*(len(XT)*theta))
	return grad

def primeraParte():

	valores = read_csv("ex2data1.csv", header = None).to_numpy()

	admitidos = np.where(valores[:,2] == 0)
	sus = np.where(valores[:,2] == 1)
	plt.scatter(valores[admitidos,0], valores[admitidos,1], marker='o',c='red')
	plt.scatter(valores[sus,0], valores[sus,1], marker='+',c='blue')


	X = valores[:,:-1]
	Y = valores[:,-1]
	m = np.shape(X)[0]
	n = np.shape(X)[1]
	X = np.hstack([np.ones([m, 1]),X])

	theta = np.zeros(np.shape(X)[1])

	result = opt.fmin_tnc(func = coste, x0 = theta, fprime = gradiente, args = (X,Y))
	theta_opt = result[0]
	print(theta_opt)

	pinta_frontera_recta(X, Y , theta_opt)

	evaluaLogistica(X,Y, theta_opt)

	plt.show()
	plt.close()

def parte2():
	plt.figure()
	valores = read_csv("ex2data2.csv", header = None).to_numpy()

	admitidos = np.where(valores[:,2] == 0)
	sus = np.where(valores[:,2] == 1)
	plt.scatter(valores[admitidos,0], valores[admitidos,1], marker='o',c='red')
	plt.scatter(valores[sus,0], valores[sus,1], marker='+',c='blue')

	X = valores[:,:-1]
	Y = valores[:,-1]
	m = np.shape(X)[0]
	n = np.shape(X)[1]
	X = np.hstack([np.ones([m, 1]),X])
	theta = np.zeros(np.shape(X)[1])
	result = opt.fmin_tnc(func = coste, x0 = theta, fprime = gradiente, args = (X,Y))
	theta_opt = result[0]

	poly = PolynomialFeatures(6)

	lambdita = 1

	print(costeRegularizado(X,Y,theta,lambdita))

	#plot_decisionboundary(X, Y, theta_opt, poly)

	plt.show()

def plot_decisionboundary(X, Y, theta, poly):
	plt.figure()
	x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
	x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
	xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
	np.linspace(x2_min, x2_max))
	h = sigmoide(poly.fit_transform(np.c_[xx1.ravel(),
	xx2.ravel()]).dot(theta))
	h = h.reshape(xx1.shape)
	plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='g')

parte2()