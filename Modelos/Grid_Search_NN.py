import numpy as np
from pandas.core.common import random_state
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import AUC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


#FUNCIÓN PARA GRIDSEARCH
def grid_search_model(input_shape = 43, optimizer = 'adam'):
    """
        Crea un modelo de red neuronal con una arquitectura específica y configuración de optimización
        para realizar una búsqueda de hiperparámetros mediante GridSearch.

        Parámetros:
        input_shape : Número de características de entrada al modelo.
        Define la dimensión de la capa de entrada.

        optimizer : Optimizador utilizado para entrenar el modelo.

        Retorna:
        model : Modelo de red neuronal compilado, con la arquitectura y optimizador especificados.
        """
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(input_shape,)))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizer, loss='binary_crossentropy',
                    metrics=['accuracy','recall', 'precision', AUC(name='auc')])

    return model