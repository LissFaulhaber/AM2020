import numpy as np

def RegCost (theta, X, y, lamb):
    
    
    #tamanho do conjunto de treinamento
    m = len(y)
    #cálculo de custo da regressão linear
    cost=(np.sum((X.dot(theta) - y)**2))
    #cálculo do termo de regularização
    reg = (np.sum(np.square(theta[1:]))*lamb)
    #cálculo do custo
    J = (cost+reg) / (2*m)
    
    return J

def gdReg(theta, X, y, lamb):
    m = len(y)
    
    #calcula o valor da hipótese para a combinação de x e theta
    h = X.dot(theta)
    #calcula a diferença entre o valor da hipótese gerada e o y correspondente
    loss = h - y
    #cálculo da regularização, retorna 0 na primeira posição do vetor
    reg = (lamb*np.r_[[[0]], theta[1:]])
    #calcula o gradiente da função, ou seja, se o valor de theta está maior ou menor que o desejado
    gradient = ((X.T.dot(loss)) + reg)/ m

    return gradient

def minimize (theta, X, y, lamb):
    epochs = 5000
    alpha = 0.001
    cost = np.zeros(epochs)
    
    for i in range(epochs):
        gradient = gdReg(theta, X, y, lamb)
        theta = theta - (alpha*gradient)
        cost[i] = RegCost(theta, X, y, lamb=lamb)
    
    return cost[-1], theta