
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.exceptions import NotFittedError

#definición de la clase
class AdaBoostModel: #Inicia la definición de la clase
    def __init__(
            ##guarda el valor del parámetro random_state en n_stimators
            self, n_estimators=50, random_state=42): #metodo constructor
        self.n_estimators = n_estimators #guarda el valor del parámetro random_state
        self.random_state = random_state
        # Crea una instancia del clasificador AdaBoost con los parámetros especificados y la guarda en el atributo de instancia self.model
        self.model = AdaBoostClassifier(n_estimators=self.n_estimators, random_state=self.random_state, algorithm='SAMME')

    #entrenar el modelo
    #def del metodo train con parametros que usará de entrenamiento
    def train(self, X_train, y_train):
        # Entrenar el modelo
        self.model.fit(X_train, y_train) #entrenar el clasificador utilizando los datos de entrenamiento
        print("Modelo entrenado.")

    def predict(self, X_test):
        # Hacer predicciones
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        # Evaluar el modelo
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Precisión del modelo AdaBoost: {accuracy:.2f}")
        return accuracy

    def grid_search(self, X_train, y_train):
        # Definir los parámetros para el Grid Search
        param_grid = {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.1, 1.0],
            'random_state': [42]
        }

        # Configurar el Grid Search
        grid_search = GridSearchCV(AdaBoostClassifier(random_state=self.random_state), param_grid, cv=5,
                                   scoring='accuracy')

        # Ajustar el modelo
        grid_search.fit(X_train, y_train)

        print(f"Mejores parámetros: {grid_search.best_params_}")
        return grid_search.best_estimator_


# Crear instancia del modelo AdaBoost
modelo_adaboost = AdaBoostModel(n_estimators=50, random_state=42)

# Realizar el Grid Search para encontrar los mejores hiperparámetros
best_model = modelo_adaboost.grid_search(X_train, y_train)
modelo_adaboost.model = best_model
# Entrenar el modelo con los mejores parámetros
best_model.fit(X_train, y_train)
print("Modelo entrenado con mejores parámetros.")

# Evaluar el modelo
accuracy = modelo_adaboost.evaluate(X_test, y_test)

# Imprimir resultados
print(f"Precisión final del modelo AdaBoost en el conjunto de prueba: {accuracy:.2f}")
