import numpy as np

def poly_features (X, indice):
    #inicia a matriz que receberá as novas features, tendo X linhas e indice colunas
    X_poly = np.zeros([X.size,indice])
    #apenas para facilitar o cálculo dentro do loop
    fator = np.arange(1, (indice+1), 1)
    for j in range(X.size):
        for i in range(indice):
            X_poly[j][i] = X[j]**fator[i]
    
    return X_poly

   