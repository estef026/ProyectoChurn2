
# ada_boost_model.py
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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

    def confusion_matrix(self, X_test, y_test):
        y_pred = self.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Matriz de Confusión")
        plt.xlabel("Predicciones")
        plt.ylabel("Valores Verdaderos")
        plt.show()
        return cm

    def classification_report(self, X_test, y_test):
        y_pred = self.predict(X_test)
        report = classification_report(y_test, y_pred)
        print("Informe de Clasificación:\n", report)
        return report

