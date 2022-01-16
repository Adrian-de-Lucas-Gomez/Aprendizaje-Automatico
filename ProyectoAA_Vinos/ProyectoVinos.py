import numpy as np

import matplotlib.pyplot as plt
from scipy.sparse.construct import random
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, accuracy_score

import pandas as pd
from pandas.io.parsers import read_csv
import sklearn.model_selection as ms

def comienzo():
    wines = read_csv('winequality-red.csv')

    print("Rows, columns: " + str(wines.shape))

    #print(wines.head())
    
    #print(wines.info())
    
    #Comprobamos que no haya parametros null
    #print(wines.isnull().sum())

    #print(wines.describe())

    ################# SVM ###############################

    #Now seperate the dataset as response variable and feature variabes
    X = wines.drop('quality', axis = 1)

    #bins = (2, 6.5, 9)
    bins = (2, 4, 6, 9)
    group_names = ['bad','normal', 'good']
    wines['quality'] = pd.cut(wines['quality'], bins = bins, labels = group_names)
    y = wines['quality']

    counts = wines['quality'].value_counts()

    nbad = counts['bad']
    nnormal = counts['normal']
    ngood = counts['good']

    print(f"bad {nbad}, good {ngood}, normal {nnormal}")

    sizesTest = {0.2, 0.3, 0.4}

    maxAccuracy = 0
    bestTestSize = 0

    for tSize in sizesTest:
        X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size = tSize, random_state = 0)           

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.fit_transform(X_test)

        svc = SVC()
        svc.fit(X_train, y_train)
        pred_svc = svc.predict(X_test)
        accuracy = accuracy_score(y_test, pred_svc)

        if accuracy > maxAccuracy:
            maxAccuracy = accuracy
            bestTestSize = tSize            
            bestmodel = pred_svc
            print(f"new max SVM acc {maxAccuracy}, testSize {bestTestSize}")


    print("########### SVM ############")        
    print(f"max accuracy {maxAccuracy}, best TestSize {bestTestSize}")
    print(classification_report(y_test, pred_svc))
    print()

    
    ################# RANDOM FOREST CLASIFICATION ###############################


    sizesTest = {0.2, 0.3, 0.4}

    maxAccuracy = 0
    bestTestSize = 0

    for tSize in sizesTest:

        X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size = tSize, random_state = 0)           

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.fit_transform(X_test)

        model2 = RandomForestClassifier(random_state=0)
        model2.fit(X_train, y_train)
        y_pred2 = model2.predict(X_test)
        #print(classification_report(y_test, y_pred2))

        if accuracy> maxAccuracy:
            maxAccuracy = accuracy
            bestTestSize = tSize
            bestModel = y_pred2
            print(f"new max rForest acc {maxAccuracy}, testSize {bestTestSize}")

    print("########### RANDOM FOREST ###########")
    print(f"max accuracy {maxAccuracy}, best TestSize {bestTestSize}")
    print(classification_report(y_test, pred_svc))

    ####### COSAS DE ENSEÑAR

    # feat_importances = pd.Series(model2.feature_importances_, index= X.columns)
    # feat_importances.nlargest(25).plot(kind='barh',figsize=(10,10))
    # plt.show()

    
    #Evolucion de una de las variables y  relacion con la valoracion del vino
    #fig = plt.figure(figsize = (10,6))
    #sns.barplot(x = 'quality', y = 'fixed acidity', data = wines)

    #Vista general de la correlacion del valor de los parámetros y la calidad del vino
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