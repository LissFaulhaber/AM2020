import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sigmoide import sigmoide
from sklearn.preprocessing import PolynomialFeatures

def plotDecision (data, X, theta, filename):
    
    #cálculo para fronteira de decisão
    x1_min, x1_max = X[:,1].min(), X[:,1].max(),
    x2_min, x2_max = X[:,2].min(), X[:,2].max(),
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    poly=PolynomialFeatures(6)
    h = sigmoide(poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(theta))
    h = h.reshape(xx1.shape)
    
     # gerando o gráfico de dispersão dos dados

    positivo = data[data['Aceito'].isin([1])]
    negativo = data[data['Aceito'].isin([0])]

    fig, ax = plt.subplots(figsize=(8,6))
    ax.axis([-1, 1.5, -0.8, 1.2])
    ax.scatter(positivo['Teste 1'], positivo['Teste 2'], s=50, c='k', marker='+', label='y=1')
    ax.scatter(negativo['Teste 1'], negativo['Teste 2'], s=50, c='y', marker='o', label='y=0')
    ax.contour(xx1, xx2, h, 10, linewidths=1, colors='g')
    ax.legend()
    ax.set_xlabel('Microchip Test 1')
    ax.set_ylabel('Microchip Test 2')

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    plt.savefig(filename)
    plt.show()