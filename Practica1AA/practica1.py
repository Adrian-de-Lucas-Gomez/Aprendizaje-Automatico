from os import makedirs
import numpy as np
from matplotlib import pyplot as plt
from pandas.io.parsers import read_csv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def carga_csv(file):
    valores = read_csv(file, header=None).to_numpy()
    return valores

def parte1():
    datos = carga_csv("ex1data1.csv")
    X = datos[:,0]
    Y = datos[:,1]

    m = len(X)
    alpha = 0.01
    theta_0 = theta_1 = 0
    for _ in range(1500):
        sum_0 = sum_1 = 0
        for i in range(m):
            sum_0 += (theta_0 + theta_1 * X[i]) - Y[i]        
            sum_1 += ((theta_0 + theta_1 * X[i]) - Y[i]) * X[i]
        theta_0 = theta_0 - (alpha/m) * sum_0
        theta_1 = theta_1 - (alpha/m) * sum_1

        plt.plot(X,Y, "x")  
        min_x = min(X)
        max_x = max(X)
        min_y = theta_0 + theta_1 * min_x
        max_y = theta_0 + theta_1 * max_x
        
    plt.plot([min_x,max_x],[min_y,max_y])
    plt.show()

def coste(X, Y, theta0, theta1):
    m = len(X)
    sum = 0
    for i in range(m):
        sum += (theta0 + theta1*X[i] - Y[i])**2
    return sum

def pruebaSombrero():
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)

    X, Y= np.meshgrid(X,Y)
    R= np.sqrt(X**2 +Y**2)
    Z = np.sin(R)

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

def make_data(t0_range, t1_range, X, Y):
    """Genera las matrices X,Y,Z para generar un plot en 3D
    """
    step = 0.1
    Theta0 = np.arange(t0_range[0], t0_range[1], step)
    Theta1 = np.arange(t1_range[0], t1_range[1], step)
    Theta0, Theta1 = np.meshgrid(Theta0, Theta1)
    # Theta0 y Theta1 tienen las misma dimensiones, de forma que
    # cogiendo un elemento de cada uno se generan las coordenadas x,y
    # de todos los puntos de la rejilla
    Coste = np.empty_like(Theta0)
    for ix, iy in np.ndindex(Theta0.shape):
        Coste[ix, iy] = coste(X, Y, Theta0[ix, iy], Theta1[ix, iy])

    return Theta0,Theta1,Coste

def parte1_1():
    datos = carga_csv("ex1data1.csv")
    X = datos[:,0]
    Y = datos[:,1]

    m = len(X)
    alpha = 0.01
    theta_0 = theta_1 = 0
    for _ in range(1500):
        sum_0 = sum_1 = 0
        for i in range(m):
            sum_0 += (theta_0 + theta_1 * X[i]) - Y[i]        
            sum_1 += ((theta_0 + theta_1 * X[i]) - Y[i]) * X[i]
        theta_0 = theta_0 - (alpha/m) * sum_0
        theta_1 = theta_1 - (alpha/m) * sum_1

    THETA_0,THETA_1,COSTE = make_data([-10,10],[-1,4],X,Y)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(THETA_0,THETA_1,COSTE,cmap=cm.jet)
    ax.set_xlabel("θ0")
    ax.set_ylabel("θ1")
    ax.set_zlabel("Coste")
    plt.show()
    plt.clf()

    plt.plot(theta_0, theta_1, marker='x',markersize=10,markeredgecolor='red')
    plt.contour(THETA_0,THETA_1,COSTE,np.logspace(-2,5,20))
    plt.show()
    plt.clf()

