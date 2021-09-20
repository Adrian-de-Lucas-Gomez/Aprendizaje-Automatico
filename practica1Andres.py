import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def dot_product(x1, x2):
    tic = time.process_time()
    dot = 0
    for i in range(len(x1)):
        dot += x1[i] * x2[i]
    toc = time.process_time()
    return 1000 * (toc - tic)

def fast_dot_product(x1, x2):
    tic = time.process_time()
    dot = np.dot(x1, x2)
    toc = time.process_time()
    return 1000 * (toc - tic)

def compara_tiempos():
    print("adwdaw")
    sizes = np.linspace(100, 1000000, 100)
    times_dot = []
    times_fast = []

    for size in sizes:
        x1 = np.random.uniform(1,100,int(size))
        x2 = np.random.uniform(1,100,int(size))
        times_dot+= [dot_product(x1,x2)]
        times_fast += [fast_dot_product(x1,x2)]

    plt.figure()
    plt.scatter(sizes, times_dot, c = 'red', label='bucle')
    plt.scatter(sizes, times_fast, c = 'blue', label='vector')
    plt.legend()
    plt.show()

def integra_mc(fun, a, b, num_puntos = 10):
    
    X = np.random.uniform(a,b,size= (num_puntos))
    Y = np.random.uniform(-10,10,size= (num_puntos))
    plt.plot(X,Y)
    plt.show()

    return 


def pruebaInt():

    # X = np.linspace(0,100, 100)
    # Y = funcionInetegrable(X)

    # plt.plot(X,Y)

    #plt.show()

    integra_mc(funcionInetegrable, 0, 10)

    return

def funcionInetegrable(x):
    return x- 100*x*x + x*x*x

pruebaInt()