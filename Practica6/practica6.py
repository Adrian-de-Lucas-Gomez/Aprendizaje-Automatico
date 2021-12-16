from re import X
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.svm import SVC
from process_email import email2TokenList
import codecs
from get_vocab_dict import getVocabDict
import glob
import sklearn.model_selection as ms
import time



def visualize_boundary(X, y, svm):
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    x1, x2 = np.meshgrid(x1, x2)
    yp = svm.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape)
    pos = (y == 1).ravel()
    neg = (y == 0).ravel()
    plt.figure()
    plt.scatter(X[pos, 0], X[pos, 1], color='black', marker='+')
    plt.scatter(X[neg, 0], X[neg, 1], color='yellow',
                edgecolors='black', marker='o')
    plt.contour(x1, x2, yp)
    plt.show()


def parte1():
    data = loadmat('ex6data1.mat')
    X, y = data['X'], data['y'].ravel()
    #svm = SVC(kernel = 'linear', C=1)
    svm = SVC(kernel='linear', C=100)
    svm.fit(X, y.ravel())
    visualize_boundary(X, y, svm)


def parte2():
    data = loadmat('ex6data2.mat')
    X, y = data['X'], data['y'].ravel()
    sigma = 0.1
    C = 1
    svm = SVC(kernel='rbf', C=C, gamma=1/(2 * sigma**2))
    svm.fit(X, y.ravel())
    visualize_boundary(X, y, svm)


def Spam(corpusDict):
    allFiles = glob.glob('spam/*.txt')
    Xspam = np.zeros((len(allFiles), len(corpusDict)))
    Yspam = np.ones(len(allFiles))

    i = 0
    for file in allFiles:
        emails = codecs.open(file, 'r', encoding='utf-8', errors = 'ignore').read()
        tokens = email2TokenList(emails)
        palabras = filter(None, [corpusDict.get(x) for x in tokens])
        for palabra in palabras:
            Xspam[i, palabra-1] = 1
        i+=1

    Xspam.shape
    
def EasyHam(corpusDict):
    allFiles = glob.glob('easy_ham/*.txt')
    XeasyHam = np.zeros((len(allFiles), len(corpusDict)))
    easyHam = np.ones(len(allFiles))

    j = 0
    for file in allFiles:
        emails = codecs.open(file, 'r', encoding='utf-8', errors = 'ignore').read()
        tokens = email2TokenList(emails)
        palabras = filter(None, [corpusDict.get(x) for x in tokens])
        for palabra in palabras:
            XeasyHam[j, palabra-1] = 1
        j+=1
    
    XeasyHam.shape

def HardHam(corpusDict):
    allFiles = glob.glob('easy_ham/*.txt')
    XhardHam = np.zeros((len(allFiles), len(corpusDict)))
    hardHam = np.ones(len(allFiles))

    k = 0
    for file in allFiles:
        emails = codecs.open(file, 'r', encoding='utf-8', errors = 'ignore').read()
        tokens = email2TokenList(emails)
        palabras = filter(None, [corpusDict.get(x) for x in tokens])
        for palabra in palabras:
            XhardHam[k, palabra-1] = 1
        k+=1

    XhardHam.shape


def parte3():
    data = loadmat('ex6data3.mat')
    X, y = data['X'], data['y'].ravel()
    Xval, yVal = data['Xval'], data['yval'].ravel()
    sigmas = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    Cs = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    scores  = np.zeros((len(Cs), len(sigmas)))

    i = 0
    j = 0
    for c,i in zip(Cs,range(len(Cs))):
        for sigma,j in zip(sigmas,range(len(sigmas))):
            svm = SVC(kernel='rbf', C=c, gamma=1/(2 * sigma**2))
            svm.fit(X,y.ravel())
            scores[i,j] = svm.score(Xval, yVal)           

    print("Error minimo ", 1 - scores.max())
    cOpt = Cs[scores.argmax()//len(sigmas)]
    sigmaOpt = sigmas[scores.argmax() % len(sigmas)]
    print("C optimo {}    sigma optimo {}".format(cOpt, sigmaOpt) )
    svm = SVC(kernel='rbf', C=cOpt, gamma=1/(2 * sigmaOpt**2))
    svm.fit(X,y.ravel())
    visualize_boundary(X, y, svm)

 
    # Parte 4: Deteccion de SPAM----------------------------------------------------------------------

    corpusDict = getVocabDict()

    Spam(corpusDict)
    EasyHam(corpusDict)
    HardHam(corpusDict)

    Xtrain, Xtest, Ytrain, Ytest = ms.train_test_split(X, y, test_size=0.2, random_state=1)
    Xtrain, Xvalid, Ytrain, Yvalid = ms.train_test_split(Xtrain, Ytrain, test_size=0.25, random_state=1)
    
    #activamos un contador
    start= time.process_time()

    sigmas = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    Cs = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

    correct = np.empty((len(Cs), len(sigmas)))

    for c,i in zip(Cs,range(len(Cs))):
        for sigma,j in zip(sigmas,range(len(sigmas))):
            svm = SVC(kernel='rbf', C=c, gamma=1/(2 * sigma**2))
            svm.fit(Xtrain,Ytrain)
            correct[i,j]= svm.score(Xvalid, Yvalid)

    COpt = Cs[correct.argmax()//len(Cs)] 
    SigmaOpt = sigmas[correct.argmax() % len(sigmas)]

    svm = SVC(kernel='rbf', C=COpt, gamma=1/(2 * SigmaOpt**2))
    svm.fit(Xtrain, Ytrain)

    #Paramos el contador
    stop = time.process_time()
    
    Score = svm.score(Xtest, Ytest)
    TimeElapsed = stop - start

    print("\n")
    print("DETECCION DE SPAWM\n")

    print("Error minimo ", 1 - Score.max())
    print("C optimo {}    sigma optimo {}".format(COpt, SigmaOpt) )
    print("Porcentaje aciertos {}%".format(Score*100))
    print("Tiempo en ejecicion {}segundos".format(TimeElapsed))

parte3()

