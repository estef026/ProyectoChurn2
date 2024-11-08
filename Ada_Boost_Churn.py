from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class AdaBoostChurn:
    """
    Clase para entrenar y evaluar un modelo de AdaBoost para predicción de churn.
    """

    def __init__(self, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=42):
        """
        Inicializa el modelo de AdaBoost con los hiperparámetros especificados.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.algorithm = algorithm
        self.random_state = random_state

        # Crear el modelo con los hiperparámetros
        self.model = AdaBoostClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            algorithm=self.algorithm,
            random_state=self.random_state
        )
        self.feature_names = None

    def fit(self, X_train, y_train, feature_names=None):
        """
        Entrena el modelo con los datos de entrenamiento.

        Parámetros:
        X_train: array-like, datos de entrenamiento preprocesados
        y_train: array-like, etiquetas de entrenamiento
        feature_names: list, nombres de las características (opcional)
        """
        self.feature_names = feature_names if feature_names is not None else [f'Feature_{i}' for i in range(X_train.shape[1])]
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """
        Evalúa el modelo con los datos de prueba.

        Retorna:
        dict con métricas de rendimiento
        """
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        metrics = {
            'classification_report': classification_report(y_test, y_pred),
            'roc_auc_score': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

        return metrics

    def predict(self, X_test):
        """
        Predice probabilidades para nuevos datos.
        """
        return self.model.predict(X_test)

    def predict_proba(self, X):
        """
        Predice probabilidades para nuevos datos.
        """
        return self.model.predict_proba(X)[:, 1]

    def plot_feature_importance(self, top_n=10):
        """
        Visualiza las características más importantes del modelo.
        """
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance.head(top_n), x='importance', y='feature')
        plt.title(f'Top {top_n} Características más Importantes')
        plt.tight_layout()
        plt.show()

        return feature_importance

    def plot_confusion_matrix(self, y_true, y_pred=None):
        """
        Visualiza la matriz de confusión.
        """
        if y_pred is None:
            y_pred = self.model.predict(y_true)
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matriz de Confusión')
        plt.ylabel('Valor Real')
        plt.xlabel('Valor Predicho')
        plt.tight_layout()
        plt.show()

    def get_params(self, deep=True):
        """
        Devuelve los parámetros del modelo.
        """
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'algorithm': self.algorithm,
            'random_state': self.random_state
        }

    def set_params(self, **params):
        """
        Actualiza los parámetros del modelo.
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
                if key in ['n_estimators', 'learning_rate', 'algorithm', 'random_state']:
                    self.model.set_params(**{key: value})
        return self
