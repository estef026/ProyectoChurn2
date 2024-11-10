from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class AdaBoostChurn:
    """
    Clase para entrenar y evaluar un modelo de AdaBoost para predicción de churn.
    """

    def __init__(self, n_estimators=50, learning_rate=1.0, algorithm='SAMME',  max_depth=3, random_state=42):
        """
        Inicializa el modelo de AdaBoost con los hiperparámetros especificados.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.algorithm = algorithm
        self.max_depth = max_depth
        self.random_state = random_state

        # Crear el estimador base, un DecisionTreeClassifier con el max_depth especificado
        base_estimator = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)

        # Crear el modelo de AdaBoost con el estimador base
        self.model = AdaBoostClassifier(
            estimator=base_estimator,  # Establecer el estimador base
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

    def evaluate(self, X_train, y_train, X_test, y_test):
        """
        Evalúa el modelo con los datos de entrenamiento y prueba.

        Retorna:
        dict con métricas de rendimiento para ambos conjuntos de datos
        """
        # Evaluar en el conjunto de entrenamiento
        y_train_pred = self.model.predict(X_train)
        y_train_pred_proba = self.model.predict_proba(X_train)[:, 1]

        train_metrics = {
            'classification_report_train': classification_report(y_train, y_train_pred),
            'roc_auc_score_train': roc_auc_score(y_train, y_train_pred_proba),
            'confusion_matrix_train': confusion_matrix(y_train, y_train_pred)
        }

        # Evaluar en el conjunto de prueba
        y_test_pred = self.model.predict(X_test)
        y_test_pred_proba = self.model.predict_proba(X_test)[:, 1]

        test_metrics = {
            'classification_report_test': classification_report(y_test, y_test_pred),
            'roc_auc_score_test': roc_auc_score(y_test, y_test_pred_proba),
            'confusion_matrix_test': confusion_matrix(y_test, y_test_pred)
        }

        # Combinar métricas de ambos conjuntos
        metrics = {**train_metrics, **test_metrics}

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

        plt.figure(figsize=(8, 6))
        sns.barplot(data=feature_importance.head(top_n), x='importance', y='feature')
        plt.tight_layout()
        plt.show()



    def plot_confusion_matrix(self, X, y_true, y_pred=None):
        """
        Visualiza la matriz de confusión.
        """
        if y_pred is None:
            # Predecir las clases usando las características
            y_pred = self.model.predict(X)

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
            'max_depth': self.max_depth,
            'random_state': self.random_state
        }

    def set_params(self, **params):
        """
        Actualiza los parámetros del modelo.
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
                if key in ['n_estimators', 'learning_rate', 'algorithm', 'max_depth', 'random_state']:
                    self.model.set_params(**{key: value})
        return self
