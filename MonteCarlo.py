import numpy as np
from matplotlib import pyplot as plt

def main():
    plt.figure()
    plt.show()
    print (integra_mc_Array(funcionMasDos, 2, 4))

def integra_mc_Array(fun, a, b, num_puntos=10000):
    #No tengo ni zorra d que hacer
    secuenciaAleatoria = np.random.randint(a, b, num_puntos)

    secuenciaFuncion = np.arange(a, b, num_puntos)

    funcionMasDos(secuenciaFuncion) #Deberia de devorverlo con los valores calculados

    #maximo = secuenciaFuncion[9999] #El ultimo valor en este caso es el mas grande
    maximo = np.amax(secuenciaFuncion)

    dentro = secuenciaAleatoria < secuenciaFuncion  #array de booleanos que muestra los valores por debajo de la funcion

    Numdebajo = np.sum(dentro)  #Suma los unos de los true y asi sabemos cuantos hay debajo

    integral = Numdebajo/10000 * (b-a) * maximo     #Calculo de la integral

    return integral


def funcionMasDos(x):
    x = x + 2

def integra_mc_Bucle(fun, a, b, num_puntos=10000):

    secuenciaAleatoria = np.random.randint(a, b, num_puntos)

    secuenciaFuncion = np.arange(a, b, num_puntos)

    for number in secuenciaFuncion:
        number = number + 2
    
    #Por hacer...


