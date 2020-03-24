import numpy as np

def RegCost (lamb, X, y, theta):
    
    
    #tamanho do conjunto de treinamento
    m = len(y)
    #cálculo de custo da regressão linear
    cost=(np.sum((X.dot(theta) - y)**2))
    #cálculo do termo de regularização
    reg = (np.sum(np.square(theta[1:]))*lamb)
    #cálculo do custo
    J = (cost+reg) / (2*m)
    
    return J

def gdReg(lamb, X, y, theta):
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