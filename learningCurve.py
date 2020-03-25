import numpy as np
from linearRegCostFunction import RegCost, minimize

def learningCurve (X, y, Xval, yval, lamb):
    #valor inicial para theta
    theta0 = np.zeros((X.shape[1],1))
    #inicialização de vetores que receberão os valores de erros
    erro = np.zeros((len(y), 1))
    erro_val = np.zeros((len(y), 1))
    
    for i in range (len(y)):
        #calcula o melhor theta para cada combinação de X[1:i+1]
        cost, theta = minimize(theta0, X[1:i+1],y[1:i+1], lamb)
        #calcula o erro de cada combinação de X e theta
        erro[i] = RegCost(theta, X[1:i+1],y[1:i+1],lamb)
        #calcula o erro do conjunto de validação para cada theta encontrado
        erro_val[i] = RegCost(theta, Xval, yval, lamb)
        
    return erro, erro_val    
    