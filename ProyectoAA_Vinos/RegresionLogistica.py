import numpy as np
from numpy.lib import math
from pandas.io.parsers import read_csv
from matplotlib import colors, pyplot as plt
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
	m = np.shape(X)[0]
	return coste(theta, X, Y ) + lambo * np.sum(theta**2)/(2*m)

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
	grad = (gradiente(theta,XT, Y))
	a = grad[0]
	reg = lambo*theta / len(Y)
	reg[0] = a
	return grad + reg 

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

	# print(np.exp([1,2,3,4]))
	# print(valores[:,1].shape)

	# print(coste(theta,X,Y))
	# print(gradiente(theta,X,Y))

	result = opt.fmin_tnc(func = coste, x0 = theta, fprime = gradiente, args = (X,Y))
	theta_opt = result[0]
	print(theta_opt)


	#plt.plot([min_x, max_x], [min_y, max_y])
	#plt.plot([50, 100], [50, 100])

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
	#X = np.hstack([np.ones([m, 1]),X])

	poly = PolynomialFeatures(6)
	x_ft= poly.fit_transform(X)
	theta = np.zeros(np.shape(x_ft)[1])

	ini = 0
	fin = 2
	lambditas = np.linspace(ini,fin,5)

	colorines = ["red", "darkorange", "yellow", "green", "aqua"]

	for l, col in zip(lambditas,colorines):
		result = opt.fmin_tnc(func = costeRegularizado, x0 = theta, fprime = gradienteRegularizado, args = (x_ft,Y,l))
		theta_opt = result[0]
		plot_decisionboundary(X,Y,theta_opt,poly, l, col)


	#plot_decisionboundary(X, Y, theta_opt, poly)
	plt.text(0,-1.0,("from {0} to {1}".format(ini,fin) ))
	plt.show()

def plot_decisionboundary(X, Y, theta, poly, lanbda,color):
	x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
	x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
	xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
	np.linspace(x2_min, x2_max))
	h = sigmoide(poly.fit_transform(np.c_[xx1.ravel(),
	xx2.ravel()]).dot(theta))
	h = h.reshape(xx1.shape)
	plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors=color)


parte2()