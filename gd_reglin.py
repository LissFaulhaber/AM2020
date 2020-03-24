import numpy as np
from custo_reglin_uni import custo_reglin_uni

#como o valor inicial de theta é definido na própria função, esta se aplica apenas a casos de função univariada
def gd_reglin_uni(X, y, alpha, epochs, theta = np.array([0,0], ndmin = 2).T):
    
    #tamanho do conjunto de treinamento
    m = len(y)
    
    #calcula o custo para cada valor (epochs) a ser adotado por theta
    cost = np.zeros(epochs)
    #range de saltos a serem usados em theta
    for i in range(epochs):
        #calcula o valor da hipótese para a combinação de x e theta
        h = X.dot(theta)
        #calcula a diferença entre o valor da hipótese gerada e o y correspondente
        loss = h - y
        #calcula o gradiente da função, ou seja, se o valor de theta está maior ou menor que o desejado
        gradient = X.T.dot(loss) / m
        #calcula novo valor para theta
        theta = theta - (alpha * gradient)
        #calcula novo custo da função
        cost[i] = custo_reglin_uni(X, y, theta = theta)

    return cost[-1], theta

#de mesmo modo, esta função atenderá apenas a funções com 03 variáveis (features)
def gd(X, y, alpha, epochs, theta=np.array([0,0,0], ndmin = 2).T):

    m = len(y)

    cost = np.zeros(epochs)

    for i in range(epochs):
        h = X.dot(theta)
        loss = h - y
        gradient = X.T.dot(loss) / m
        theta = theta - (alpha * gradient)
        cost[i] = custo_reglin_uni(X, y, theta=theta)

    return cost[-1], theta
