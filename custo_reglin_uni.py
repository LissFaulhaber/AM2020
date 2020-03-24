import numpy as np
#por ter o valor de theta definido no momento em que se chama a função, ela pode ser aplicada tanto para função univariada e multivariada, pois irá varrer todo o vetor correspondente a X e o vetor de theta inserido na chamada da função.
def custo_reglin_uni(X, y, theta):

    # Quantidade de exemplos de treinamento
    m = len(y)

    # Computar a função do custo J
    # custo J = (soma (valores de x * valores de theta - valor de y)^2) / dobro do tamanho do conjunto de treinamento 
    J = (np.sum((X.dot(theta) - y)**2)) / (2 * m)

    return J