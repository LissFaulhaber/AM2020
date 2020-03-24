import numpy as np

#Z-Score norm
def normalizar_caracteristica(examData, labels):
    #média dos valores X
    mean_examData = np.mean(examData, axis=0)
    #desvio padrão dos valores X
    std_examData = np.std(examData, axis=0)
    #cálculo de normalização de X
    examData_norm = (examData - mean_examData) / std_examData

    #média de y
    mean_labels = np.mean(labels)
    #desvio padrão de y
    std_labels = np.std(labels)
    labels_norm = (labels - mean_labels) / std_labels
    
    # Incluir o valor de 1 em x, pois theta0 = 1
    examData_norm = np.c_[np.ones((examData_norm.shape[0], 1)), examData_norm]

    return examData_norm , labels_norm