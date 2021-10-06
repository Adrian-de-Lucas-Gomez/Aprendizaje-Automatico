#jupyter-nbconvert --to PDFviaHTML example.ipynb
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
    X = datos[:, 0]
    Y = datos[:, 1]

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

        plt.plot(X, Y, "x")
        min_x = min(X)
        max_x = max(X)
        min_y = theta_0 + theta_1 * min_x
        max_y = theta_0 + theta_1 * max_x

    plt.plot([min_x, max_x], [min_y, max_y])
    plt.show()


def costeIterativo(X, Y, theta0, theta1):
    m = len(X)
    sum = 0
    for i in range(m):
        sum += (theta0 + theta1*X[i] - Y[i])**2
    return sum

def coste(X, Y, Theta):
    H = np.dot(X, Theta)
    Aux = (H - Y) ** 2
    return Aux.sum() / (2 * len(X))

def pruebaSombrero():
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)

    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X**2 + Y**2)
    Z = np.sin(R)

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


def make_data(t0_range, t1_range, X, Y):
    step = 0.1
    Theta0 = np.arange(t0_range[0], t0_range[1], step)
    Theta1 = np.arange(t1_range[0], t1_range[1], step)
    Theta0, Theta1 = np.meshgrid(Theta0, Theta1)
    Coste = np.empty_like(Theta0)
    for ix, iy in np.ndindex(Theta0.shape):
        Coste[ix, iy] = costeIterativo(X, Y, Theta0[ix, iy], Theta1[ix, iy])

    return Theta0, Theta1, Coste


def parte1_1():
    datos = carga_csv("ex1data1.csv")
    X = datos[:, 0]
    Y = datos[:, 1]

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

    THETA_0, THETA_1, COSTE = make_data([-10, 10], [-1, 4], X, Y)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(THETA_0, THETA_1, COSTE, cmap=cm.jet)
    ax.set_xlabel("θ0")
    ax.set_ylabel("θ1")
    ax.set_zlabel("Coste")
    plt.show()
    plt.clf()

    plt.plot(theta_0, theta_1, marker='x',
             markersize=10, markeredgecolor='red')
    plt.contour(THETA_0, THETA_1, COSTE, np.logspace(-2, 5, 20))
    plt.show()
    plt.clf()


def normMatriz(matriz):
    matriz_norm = np.empty_like(matriz, dtype=np.float32)
    mu = np.empty_like(matriz[0], dtype=np.float32)
    sigma = np.empty_like(matriz[0], dtype=np.float32)

    mu = np.mean(matriz, axis=0)
    sigma = np.std(matriz, axis=0)

    matriz_norm = (matriz-mu) / sigma

    return matriz_norm, mu, sigma


def descenso_gradiente(X,Y,alpha):
    nIteraciones = 100    
    theta = np.zeros(np.shape(X)[1])
    costes = np.zeros(nIteraciones)

    for i in range(nIteraciones):
        costes[i] = coste(X,Y, theta)
        aux = gradiente(X, Y, theta, alpha)
        theta = aux

    return theta, costes


def gradiente(X, Y, Theta, alpha):
    NuevaTheta = Theta
    m = np.shape(X)[0]
    n = np.shape(X)[1]
    H = np.dot(X, Theta)
    return Theta-(alpha/m) * np.dot(np.transpose(X), (H-Y))

def gradienteIterativo(X, Y, Theta, alpha):
    NuevaTheta = Theta
    m = np.shape(X)[0]
    n = np.shape(X)[1]
    H = np.dot(X, Theta)
    Aux = (H - Y)
    for i in range(n):
        Aux_i = Aux * X[:, i]
        NuevaTheta[i] -= (alpha / m) * Aux_i.sum()
    return NuevaTheta

def parte2():
    datos = carga_csv("ex1data2.csv")
    datos_norm, media, desviacion = normMatriz(datos)
    X = datos_norm[:, :-1]
    Y = datos_norm[:, -1]
    m = np.shape(X)[0]
    n = np.shape(X)[1]
    X = np.hstack([np.ones([m, 1]), X])

    plt.figure()
    alpha = 0.001
    thetas, costes = descenso_gradiente(X,Y,alpha)
    plt.scatter(np.arange(np.shape(costes)[0]),costes,c='red',label="0.001 ")
    alpha = 0.003
    thetas, costes = descenso_gradiente(X,Y,alpha)
    plt.scatter(np.arange(np.shape(costes)[0]),costes,c='orange',label='0.003')
    alpha = 0.01
    thetas, costes = descenso_gradiente(X,Y,alpha)
    plt.scatter(np.arange(np.shape(costes)[0]),costes,c='yellow',label='0.01')
    alpha = 0.03
    thetas, costes = descenso_gradiente(X,Y,alpha)
    plt.scatter(np.arange(np.shape(costes)[0]),costes,c='green',label='0.01')
    alpha = 0.1
    thetas, costes = descenso_gradiente(X,Y,alpha)
    plt.scatter(np.arange(np.shape(costes)[0]),costes,c='blue',label='0.1')
    alpha = 0.3
    thetas, costes = descenso_gradiente(X,Y,alpha)
    plt.scatter(np.arange(np.shape(costes)[0]),costes,c='purple',label='0.3')
    alpha = 0.5
    thetas, costes = descenso_gradiente(X,Y,alpha)
    plt.scatter(np.arange(np.shape(costes)[0]),costes,c='black',label='0.5')
        
    plt.savefig("DescensoGradiente")
    plt.savefig("res")
    pies2Nor= (1650 - media[0]) / desviacion[0]    
    habsNor = (3 - media[1]) / desviacion[1]
    precioPredecido = (thetas[0]+thetas[1]*pies2Nor+thetas[2]*habsNor)*desviacion[2] + media[2]
    print("precio casa de 1650pies^2 y 3 habs (descenso grad): ", precioPredecido)


#np.transpose      np.matmul      np.linalg.pinv (es hacer la inversa)
# (XT * X )^-1 * XT * Y
def ec_normal(matriz, precios):
    first_term = np.linalg.pinv(np.matmul(np.transpose(matriz),matriz))
    second_term = np.transpose(matriz)
    third_term = precios

    theta = np.matmul(np.matmul(first_term , second_term), third_term)

    return theta

def parte2_2():
    datos = carga_csv("ex1data2.csv")
    X = datos[:, :-1]
    Y = datos[:, -1]
    m = np.shape(X)[0]
    n = np.shape(X)[1]
    X = np.hstack([np.ones([m, 1]), X])
    theta = ec_normal(X, Y)

    precioPredecido= theta[1]*1650 + theta[2]*3 + theta[0]    

    print("precio casa con 1650pies^2 y 3 habs", precioPredecido)
    return

#[89597.90954355   139.21067402 -8738.01911255]
# O1*x + 02*y + O3
#1600,3,329900
#
def normal(X, Y):
    x_transpose = np.transpose(X)
    x_transpose_dot_x = x_transpose.dot(X)
    term_1 = np.linalg.pinv(x_transpose_dot_x)
    term_2 = x_transpose.dot(Y)
    return term_1.dot(term_2)

#[-9.98019634e-17  8.84765988e-01 -5.31788197e-02]
# prueba()
parte2()
parte2_2()
