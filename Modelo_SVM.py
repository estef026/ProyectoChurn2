import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class ModeloSVM:
    def __init__(self, datos, objetivo):
        """
        Inicializa la clase ModeloSVM con datos y objetivo.

        Parámetros:
        - datos: DataFrame con las características.
        - objetivo: Series o array con la variable de salida.
        """
        self.datos = datos
        self.objetivo = objetivo
        self.X_entrenamiento, self.X_prueba, self.y_entrenamiento, self.y_prueba = train_test_split(
            datos, objetivo, test_size=0.2, random_state=42
        )

    def entrenar(self):
        """
        Entrena el modelo SVM utilizando GridSearchCV para optimizar los hiperparámetros.
        """
        # Definimos los hiperparámetros que queremos optimizar
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf', 'poly']
        }

        # Creamos el clasificador SVM
        svm = SVC()

        # Utilizamos GridSearchCV para encontrar la mejor combinación de hiperparámetros
        self.busqueda_grid = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy')
        self.busqueda_grid.fit(self.X_entrenamiento, self.y_entrenamiento)

    def evaluar(self):
        """
        Evalúa el modelo SVM en el conjunto de prueba y devuelve la precisión.

        Retorna:
        - exactitud: Precisión del modelo en el conjunto de prueba.
        """
        # Realizamos predicciones en el conjunto de prueba
        y_pred = self.busqueda_grid.predict(self.X_prueba)

        # Calculamos la exactitud
        exactitud = accuracy_score(self.y_prueba, y_pred)
        return exactitud

    def predecir(self, nuevos_datos):
        """
        Realiza predicciones en nuevos datos.

        Parámetros:
        - nuevos_datos: DataFrame o array con los datos para predecir.

        Retorna:
        - predicciones: Array con las predicciones del modelo.
        """
        predicciones = self.busqueda_grid.predict(nuevos_datos)
        return predicciones
