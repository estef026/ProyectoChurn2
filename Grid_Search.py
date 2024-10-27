import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class ModelGridSearch:
    def __init__(self, model, param_grid, cv=5):
        """
        Inicializa la clase para realizar Grid Search.

        :param model: Modelo de scikit-learn a optimizar.
        :param param_grid: Diccionario de hiperparámetros a explorar.
        :param cv: Número de divisiones para la validación cruzada.
        """
        self.model = model
        self.param_grid = param_grid
        self.cv = cv
        self.grid_search = None

    def perform_grid_search(self, X, y):
        """Realiza el Grid Search para encontrar los mejores hiperparámetros."""
        self.grid_search = GridSearchCV(estimator=self.model, param_grid=self.param_grid, cv=self.cv, scoring='accuracy')
        self.grid_search.fit(X, y)
        print("Mejores hiperparámetros encontrados:", self.grid_search.best_params_)

    def get_best_model(self):
        """Retorna el mejor modelo encontrado durante el Grid Search."""
        return self.grid_search.best_estimator_

    def evaluate_best_model(self, X_test, y_test):
        """Evalúa el mejor modelo en un conjunto de prueba."""
        best_model = self.get_best_model()
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Precisión del mejor modelo: {accuracy:.2f}")
        return accuracy


# Ejemplo de uso de la clase ModelGridSearch
if __name__ == "__main__":
    # Cargar tus datos (X, y)
    # Ejemplo:
    # X = pd.read_csv('datos.csv')
    # y = X.pop('target_column')

    # Define el rango de hiperparámetros para probar
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }

    # Crear una instancia del modelo
    rf_model = RandomForestClassifier(random_state=42)

    # Crear una instancia de la clase ModelGridSearch
    grid_search_model = ModelGridSearch(model=rf_model, param_grid=param_grid)

    # Realizar la búsqueda de hiperparámetros
    grid_search_model.perform_grid_search(X, y)

    # Evaluar el mejor modelo si se dispone de un conjunto de prueba
    # X_test, y_test = ...
    # grid_search_model.evaluate_best_model(X_test, y_test)
