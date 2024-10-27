
# ada_boost_model.py
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

class AdaBoostModel:
    def __init__(self, n_estimators=50, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = AdaBoostClassifier(n_estimators=self.n_estimators, random_state=self.random_state, algorithm='SAMME')

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        print("Modelo entrenado.")

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Precisión del modelo AdaBoost: {accuracy:.2f}")
        return accuracy

    def grid_search(self, X_train, y_train):
        param_grid = {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.1, 1.0],
            'random_state': [42]
        }
        grid_search = GridSearchCV(AdaBoostClassifier(random_state=self.random_state), param_grid, cv=5,
                                   scoring='accuracy')
        grid_search.fit(X_train, y_train)
        print(f"Mejores parámetros: {grid_search.best_params_}")
        return grid_search.best_estimator_
