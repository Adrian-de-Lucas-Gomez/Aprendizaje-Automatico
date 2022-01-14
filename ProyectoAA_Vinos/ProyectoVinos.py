import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder

import pandas as pd
from pandas.io.parsers import read_csv

def comienzo():
    wines = read_csv('winequality-red.csv')

    print("Rows, columns: " + str(wines.shape))

    #wines.head()

    #Evolucion de una de las variables y  relacion con la valoracion del vino
    #fig = plt.figure(figsize = (10,6))
    #sns.barplot(x = 'quality', y = 'fixed acidity', data = wines)

    #Vista general de la correlacion del valor de los parámetros y la calidad del vino
    corr = wines.corr()
    plt.subplots(figsize=(15,10))
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))

    wines.hist(bins=25,figsize=(10,10))

    bins = (2, 6.5, 8)
    group_names = ['bad','good']
    wines['quality'] = pd.cut(wines['quality'], bins = bins, labels = group_names)

    label_quality = LabelEncoder() 

    wines['quality'] = label_quality.fit_transform(wines['quality'])
    wines['quality'].value_counts()
    sns.countplot(wines['quality'])

    plt.show()

comienzo()