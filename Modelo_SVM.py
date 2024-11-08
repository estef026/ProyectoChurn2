import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score
import seaborn as sns
import matplotlib.pyplot as plt


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
        self.modelo_entrenado = None

    def svm_default(self, probability=True):
        """
        Devuelve un clasificador SVM con un parámetro predeterminado.

        Parámetros:
        - probability: Booleano que indica si el clasificador debe calcular probabilidades.

        Retorna:
        - SVC con los parámetros predeterminados.
        """
        return SVC(probability=probability)

    def entrenar(self, modelo=None, use_grid_search=True, param_grid=None):
        """
        Entrena el modelo SVM, con la opción de usar un estimador personalizado o GridSearchCV.

        Parámetros:
        - modelo: Estimador SVM personalizado (si se proporciona).
        - use_grid_search: Booleano que indica si se debe usar GridSearchCV.
        - param_grid: Diccionario de hiperparámetros para GridSearchCV (solo si use_grid_search es True).
        """
        if modelo is None:
            modelo = self.svm_default()

        if use_grid_search:
            # Si no se proporcionan hiperparámetros, usar valores predeterminados
            if param_grid is None:
                param_grid = {
                    'C': [0.01, 0.1, 1, 10],
                    'kernel': ['linear', 'rbf', 'poly']
                }
            self.busqueda_grid = GridSearchCV(modelo, param_grid, cv=3, scoring='accuracy')
            self.busqueda_grid.fit(self.X_entrenamiento, self.y_entrenamiento)
            self.modelo_entrenado = self.busqueda_grid.best_estimator_
        else:
            # Entrena el modelo directamente sin grid search
            modelo.fit(self.X_entrenamiento, self.y_entrenamiento)
            self.modelo_entrenado = modelo


    def evaluar(self):
        """
        Evalúa el modelo SVM en el conjunto de prueba y devuelve la precisión.

        Retorna:
        - exactitud: Precisión del modelo en el conjunto de prueba.
        """
        if self.modelo_entrenado is None:
            raise Exception("El modelo no ha sido entrenado. Llame al método entrenar() primero.")

        # Realizamos predicciones en el conjunto de prueba
        self.y_pred = self.modelo_entrenado.predict(self.X_prueba)

        # Calculamos la exactitud
        recall = recall_score(self.y_prueba, self.y_pred)
        return recall

    def predecir(self, nuevos_datos):
        """
        Realiza predicciones en nuevos datos.

        Parámetros:
        - nuevos_datos: DataFrame o array con los datos para predecir.

        Retorna:
        - predicciones: Array con las predicciones del modelo.
        """
        if self.modelo_entrenado is None:
            raise Exception("El modelo no ha sido entrenado. Llame al método entrenar() primero.")

        predicciones = self.modelo_entrenado.predict(nuevos_datos)
        return predicciones

    def confusion_matrix(self):
        cm = confusion_matrix(self.y_prueba, self.y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Matriz de Confusión")
        plt.xlabel("Predicciones")
        plt.ylabel("Valores Verdaderos")
        plt.show()


    def classification_report(self):
        report = classification_report(self.y_prueba, self.y_pred)
        print("Informe de Clasificación:\n", report)
