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


def comienzo():
    wines = read_csv('winequality-red.csv')

    print("Rows, columns: " + str(wines.shape))

    #print(wines.head())
    
    #print(wines.info())
    
    #Comprobamos que no haya parametros null
    #print(wines.isnull().sum())

    #print(wines.describe())

    ########################## Preparacion de lo datos del DataSet ##################################

    #Quitamos la columna de valores porque es lo que queremos determinar nosotros
    X = wines.drop('quality', axis = 1)

    bins = (2, 5.5, 9)
    group_names = ['bad','good']
    wines['quality'] = pd.cut(wines['quality'], bins = bins, labels = group_names)
    y = wines['quality']

    counts = wines['quality'].value_counts()

    nbad = counts['bad']
    ngood = counts['good']

    print(f"bad {nbad}, good {ngood}")

    #Quitamos los parametros menos relevantes evitando ruido y mejorando la precision
    # X = X.drop('residual sugar', axis = 1)
    # X = X.drop('fixed acidity', axis = 1)
    # X = X.drop('free sulfur dioxide', axis = 1)
    # X = X.drop('citric acid', axis = 1)
    #################################################################################

    #print(X.head())

    X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.2, random_state=1)
    X_train, X_valid, y_train, y_valid = ms.train_test_split(X_train, y_train, test_size=0.25, random_state=1)
    
    #Distribucion de los tamaños para loc conjuntos de entrenamiento, validacion y prueba
    print(len(X_train), len(X_test), len(X_valid))

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)
    X_valid = sc.fit_transform(X_valid)

    ################# SVM ###############################

    Cs = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    # sigmas = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

    # bestAcc = 0
    # bestC = 0
    # bestSig = 0

    # for c in Cs :
    #     for s in sigmas:

    #         svc = SVC(kernel='rbf', C=c, gamma=1/(2 * s**2))
    #         svc.fit(X_train, y_train)
    #         pred_svc = svc.predict(X_valid)
    #         #print(classification_report(y_test, pred_svc))
    #         accuracy = accuracy_score(y_valid, pred_svc)
    #         if bestAcc < accuracy:
    #             bestAcc = accuracy
    #             bestC = c
    #             bestSig = s

    # print("########### SVM ############")
    
    # svc = SVC(kernel='rbf', C=bestC, gamma=1/(2 * bestSig**2))
    # svc.fit(X_train, y_train)

    # pred_svc = svc.predict(X_test)
    # print(classification_report(y_test, pred_svc))

    # print()

    
    ################# RANDOM FOREST CLASIFICATION ###############################

    # randForest = RandomForestClassifier(random_state=0)
    # randForest.fit(X_train, y_train)
    # pred_rfc = randForest.predict(X_test)
    # accuracy = accuracy_score(y_test, pred_rfc)

    # print("########### RANDOM FOREST ###########")
    # print(classification_report(y_test, pred_rfc))

    ####### Utilidad de los atributos

    # feat_importances = pd.Series(randForest.feature_importances_, index= X.columns)
    # feat_importances.nlargest(25).plot(kind='barh',figsize=(10,10))
    # plt.show()


    ################# LOGISTIC REGRESION ###################

    print("########### LOGISTIC REGRESION ###########")
    print(f"max accuracy in training with validation is {5}") 
    
    #Ponemos la primera columna con 1 para facilitar las cuentas vectoriales
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

    maxAccuracy = 0
    bestC = 0

    #Busqueda de la mejor configuracion de C

    for Ci in Cs:     
        logReg = LogisticRegression(random_state=1, solver='lbfgs', C=Ci)
        logReg.fit(X_train, y_train)
        pred_sgd = logReg.predict(X_valid)
        accuracy = accuracy_score(y_valid, pred_sgd)

        if accuracy > maxAccuracy:
            maxAccuracy = accuracy
            bestC = Ci
    
    print("########### LOGISTIC REGRESION ###########")
    print(f"max accuracy in training with validation is {maxAccuracy}")         

    logReg = LogisticRegression(random_state=1, solver='lbfgs', C=bestC)
    logReg.fit(X_valid, y_valid)
    pred_sgd = logReg.predict(X_test)

    print(classification_report(y_test, pred_sgd))
    

    
    #Evolucion de una de las variables y  relacion con la valoracion del vino
    #fig = plt.figure(figsize = (10,6))
    #sns.barplot(x = 'quality', y = 'fixed acidity', data = wines)

    #Vista general de la correlacion del valor de los parametros y la calidad del vino
    # corr = wines.corr()
    # plt.subplots(figsize=(15,10))
    # sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))

    # wines.hist(bins=25,figsize=(10,10))

    # bins = (2, 6.5, 8)
    # group_names = ['bad','good']
    # wines['quality'] = pd.cut(wines['quality'], bins = bins, labels = group_names)

    # label_quality = LabelEncoder() 

    # wines['quality'] = label_quality.fit_transform(wines['quality'])
    # wines['quality'].value_counts()
    # sns.countplot(wines['quality'])

    # plt.show()

comienzo()