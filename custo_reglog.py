import numpy as np
from sigmoide import sigmoide


def custo_reglog(theta, X, y):
    #transforma os valores de theta, X e y em matrix
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    #primeira parcela da função de custo para Reg. Logística, caso y=0, então grad0 = 0
    grad0 = np.multiply(-y, np.log(sigmoide(X * theta.T)))
    #segunda parcela da função de custo, caso y=1, então grad1 = 0
    grad1 = np.multiply((1 - y), np.log(1 - sigmoide(X * theta.T)))
    #calcula o valor do custo J, de acordo com as parcelas grad0 e grad1 e o tamanho do conjunto de dados
    return np.sum(grad0 - grad1) / (len(X))
