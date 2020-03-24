import numpy as np
from sigmoide import sigmoide

def costFunctionReg(theta, X, y, lamb):
    #transforma os valores de theta, X e y em matrix
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    
    #primeira parcela da função de custo para Reg. Logística, caso y=0, então grad0 = 0
    grad0 = np.multiply(-y, np.log(sigmoide(X * theta.T)))
    #segunda parcela da função de custo, caso y=1, então grad1 = 0
    grad1 = np.multiply((1 - y), np.log(1 - sigmoide(X * theta.T)))
    
    #termo de regularização
    reg = np.sum(np.square(theta[:,1:].T))*lamb
    
    #calcula o valor do custo J, de acordo com as parcelas grad0 e grad1 e o tamanho do conjunto de dados
    J = (np.sum(grad0 - grad1) / (len(y)))+ reg / (len(y)*2)
    
        
    return J

def gdFunction(theta, X, y, lamb):
    #transforma os valores de theta, X e y em matrix
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    #contagem de thetas
    parametros = int(theta.ravel().shape[1])
    #cálculo do gradiente descendente
    grad = np.zeros(parametros)

    erro = sigmoide(X * theta.T) - y

    for i in range(parametros):
        term = np.multiply(erro, X[:,i])
        if i == 0:
            grad[i] = np.sum(term) / len(y)
        else:
            gdreg = (lamb / len(y))*theta.T[i]
            grad[i] = (np.sum(term) / len(y)) + gdreg 
        
    return grad


