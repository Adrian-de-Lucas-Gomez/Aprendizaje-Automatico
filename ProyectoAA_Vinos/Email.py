import numpy as np

import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.sparse.construct import random
import scipy.optimize as opt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, accuracy_score

import pandas as pd
from pandas.io.parsers import read_csv
import sklearn.model_selection as ms

def costeRegularizado(theta, X, Y, lambo):
	m = np.shape(X)[0]
	return coste(theta, X, Y ) + lambo * np.sum(theta**2)/(2*m)

def coste(theta, X, Y):

	H = expit(np.matmul(X, theta))

	cost = (-1/(len(X))) * (np.dot(Y, np.log(H)) +
							np.dot((1-Y), np.log(1-H)))

	return cost

def gradiente(theta,XT, Y):
	H = expit(np.matmul(XT, theta))

	grad = (1/len(Y)) * np.matmul(XT.T, H - Y)
	return grad

def gradienteRegularizado(theta,XT, Y, lambo):
	grad = (gradiente(theta,XT, Y))
	a = grad[0]
	reg = lambo*theta / len(Y)
	reg[0] = a
	return grad + reg 


wines = read_csv('winequality-red.csv')

print("Rows, columns: " + str(wines.shape))

X = wines.drop('quality', axis = 1)

X = wines.drop('quality', axis = 1)

bins = (2, 5.5, 9)
group_names = ['bad','good']
wines['quality'] = pd.cut(wines['quality'], bins = bins, labels = group_names)
y = wines['quality']

counts = wines['quality'].value_counts()

nbad = counts['bad']
ngood = counts['good']

print(f"bad {nbad}, good {ngood}")

X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_valid, y_train, y_valid = ms.train_test_split(X_train, y_train, test_size=0.25, random_state=1)

#Distribucion de los tamaños para loc conjuntos de entrenamiento, validacion y prueba
print(len(X_train), len(X_test), len(X_valid))

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
X_valid = sc.fit_transform(X_valid)

################# REGRESION LOGISTICA ###################

print("########### REGRESION LOGISTICA ###########")

m = np.shape(X_train)[0]
X_train_s = np.hstack([np.ones([m, 1]),X_train])

m = np.shape(X_valid)[0]
X_valid_s = np.hstack([np.ones([m, 1]),X_valid])

theta = np.zeros(np.shape(X_train_s)[1])

#Rs = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
Rs = [0.01]

for r in Rs:
    result = opt.fmin_tnc(
        func = costeRegularizado, x0 = theta, fprime = gradienteRegularizado,
        args = (X_train_s, y_train, 1))
    theta_opt = result[0]
    print(theta_opt)