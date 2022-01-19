import numpy as np

import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.sparse.construct import random
import scipy.optimize as opt
import seaborn as sns

import checkNNGradients as checkNNG

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import pandas as pd
from pandas.io.parsers import read_csv
import sklearn.model_selection as ms

def costeRegularizado(theta, X, Y, lambo):
	m = np.shape(X)[0]
	return coste(theta, X, Y ) + lambo * np.sum(theta**2)/(2*m)

def coste(theta, X, Y):

	H = expit(np.matmul(X, theta))

	cost = (-1/(len(X))) * (np.dot(Y, np.log(H+ 0.00000001)) +
							np.dot((1-Y), np.log(1-H + 0.00000001)))

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


##############################################################################################################
def sigmoide(a):
	r = 1 / (1+ np.exp(-a))  
	return r


def costeNN(X, Y, theta1, theta2):

    a1, a2, H = redNeuronalPaLante(X, theta1,theta2) #Haces la pasada por la red neuronal

    #Queda mas claro dividido por partes
    op1= -1/(len(X))    
    op2 = Y.dot(np.log(H))
    op3 = (1-Y).dot(np.log(1-H))

    cost = op1 * np.sum(op2 + op3)
    return cost

def costeNNReg(X, Y, theta1, theta2, lambo):
    costeN = costeNN(X, Y, theta1, theta2)
    costeR = costeN + (lambo/(2*len(X)) * (np.sum(np.square(theta1[:,1:])) + np.sum(np.square(theta2[:,1:]))))
    return costeR

def redNeuronalPaLante(X, theta1, theta2):
    
    xSize = X.shape[0]
    # Capa de entrada
    a1 = np.hstack([np.ones((xSize,1)), X])    
    # Capa oculta    
    z2 = theta1.dot(a1.T)
    a2 = np.hstack([np.ones((xSize,1)), sigmoide(z2.T)])
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
    coste = costeNNReg(X, y, theta1, theta2, lambdita)

    return coste, gradiente
    
def precisionChecker(resOpt, n_input, n_hidden, n_labels, X, y):
    theta1 = np.reshape(resOpt.x[:n_hidden * (n_input + 1)] , (n_hidden, (n_input+1)))
    theta2 = np.reshape(resOpt.x[n_hidden * (n_input + 1):] , (n_labels, (n_hidden+1)))

    A1, A2, H = redNeuronalPaLante(X, theta1, theta2)

    aux = np.argmax(H, axis=1)
    aux += 1
    #calculamos cuantos se han identificado correctamente y lo dividimos por los casos de prueba
    return np.sum(aux == y) / np.shape(H)[0]

def randomWeights(L_ini, L_out, E_ini):
    return np.random.random((L_out, L_ini + 1)) * (2*E_ini) - E_ini


#############################################################################################################################




def comienzo():
    wines = read_csv('winequality-red.csv')

    print("Rows, columns: " + str(wines.shape))

    #print(wines.head())
    
    #print(wines.info())
    
    #Comprobamos que no haya parametros null
    #print(wines.isnull().sum())

    #print(wines.describe())

    ########################## Preparacion de lo datos del DataSet ##################################
    bins = (2, 5.5, 9)
    group_names = [0,1]
    wines['quality'] = pd.cut(wines['quality'], bins = bins, labels = group_names)
    
    #Quitamos la columna de valores y las etiquetas porque es lo que queremos determinar nosotros
    X = wines.drop('quality', axis = 1)
    X = X.values

    y = wines['quality']
    y = y.astype(int).values

    counts = wines['quality'].value_counts()

    nbad = counts[0]
    ngood = counts[1]

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

    # print("########### LOGISTIC REGRESION ###########")
    
    # #Ponemos la primera columna con 1 para facilitar las cuentas vectoriales
    # m = np.shape(X_train)[0]
    # X_train_s = np.hstack([np.ones([m, 1]),X_train])

    # m = np.shape(X_valid)[0]
    # X_valid_s = np.hstack([np.ones([m, 1]),X_valid])
    
    # theta = np.zeros(np.shape(X_train_s)[1])
    
    # #Rs = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    # Rs = [0.01]

    # for r in Rs:
    #     result = opt.fmin_tnc(
    #         func = costeRegularizado, x0 = theta, fprime = gradienteRegularizado,
    #         args = (X_train_s, y_train, 1))
    #     theta_opt = result[0]
    #     print(theta_opt)

    # maxAccuracy = 0
    # bestC = 0

    # #Busqueda de la mejor configuracion de C

    # for Ci in Cs:     
    #     logReg = LogisticRegression(random_state=1, solver='lbfgs', C=Ci)
    #     logReg.fit(X_train, y_train)
    #     pred_sgd = logReg.predict(X_valid)
    #     accuracy = accuracy_score(y_valid, pred_sgd)

    #     if accuracy > maxAccuracy:
    #         maxAccuracy = accuracy
    #         bestC = Ci

    # print(f"max accuracy in training with validation is {maxAccuracy}")         

    # logReg = LogisticRegression(random_state=1, solver='lbfgs', C=bestC)
    # logReg.fit(X_valid, y_valid)
    # pred_sgd = logReg.predict(X_test)

    # print(classification_report(y_test, pred_sgd))
    
    
    ################# REDES NEURONALES ###################

    print("########### REDES NEURONALES ###########")

    input_size= np.shape(X)[1]
    out_size = 2 #bueno o malo
    num_hidden = np.arange(5,50,5)
    #num_hidden = [50]
    num_labels = 2

    lambdas = np.arange(0,2,0.06)
    #lambdas = [0.1]

    record = 0
    bestL = 0
    bestHid = 0

    for hid in num_hidden:
        for l in lambdas:

            #Elejimos valores aleatorios para las thetas
            theta1 = randomWeights(input_size, hid, 0.12)
            theta2 = randomWeights(hid, out_size, 0.12)
            #Los guardamos para probar con ellos
            params_rn = np.concatenate((np.ravel(theta1), np.ravel(theta2)))
            
            num_Iterations = 100     #Vueltas dadas para tratar de optimizar
            optimizeResult = opt.minimize(
                fun=redNeuronalPatras,
                x0=params_rn,
                args=(input_size, hid, num_labels, X_train, y_train, l),
                method='TNC',
                jac=True,
                options={'maxiter': num_Iterations})

            evaluations = precisionChecker(optimizeResult, input_size, hid, num_labels, X_train, y_train)

            #porcentaje =  float(str(evaluations*100)[:5])

            A1, A2, H =  redNeuronalPaLante(X_train, theta1, theta2)
            
            #H = np.array(np.argmax(H, axis=0))

            predValues = np.zeros(np.shape(H)[0])

            i = 0

            for v in H:
                if(v[0]>v[1]):
                    predValues[i] = 0
                else:
                    predValues[i] = 1
                i += 1

            nn_matrix = confusion_matrix(y_train, predValues)
            print("Confusion matrix:")
            print(nn_matrix)

            porcentaje = ((nn_matrix[0][0] + nn_matrix[1][1]) / np.shape(H)[0]) * 100

            if porcentaje > record:
                record = porcentaje
                bestL = l
                bestHid = hid

            print(f"Precision de la red con lambda {l} y {hid} neuronas ocultas")
            print(f"{porcentaje}%\n")

    print(f"La mejor configuracion de la red fue con lambda {bestL} y {bestHid} capas ocultas")
    print( str(record) + "%\n")

    
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