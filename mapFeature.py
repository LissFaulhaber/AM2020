import numpy as np
#biblioteca com pacotes de manipulação de variáveis
from sklearn.preprocessing import PolynomialFeatures

def mapFeature(X, indice):
    #determina o grau que será utilizado no cálculo das novas variáveis
    poly = PolynomialFeatures(degree = indice)
    #realiza a transformação das variáveis, até o grau escolhido
    mapa = poly.fit_transform(X)
    
    return mapa
    