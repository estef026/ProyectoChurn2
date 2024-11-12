from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class GradientBoostingChurn:
    """
    Clase para entrenar y evaluar un modelo de Gradient Boosting para predicción de churn
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, subsample=1.0, random_state=42):
        """
        Inicializa el modelo de Gradient Boosting con los hiperparámetros especificados o valores predeterminados.
        Parámetros:
        - n_estimators: int, número de árboles en el modelo.
        - learning_rate: float, tasa de aprendizaje para ajustar el impacto de cada árbol.
        - max_depth: int, profundidad máxima de cada árbol.
        - subsample: float, fracción de muestras utilizadas para entrenar cada árbol.
        - random_state: int, semilla para la reproducibilidad.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.subsample = subsample
        # Inicializa el clasificador de Gradient Boosting con los hiperparámetros.
        self.model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
            subsample=self.subsample,
        )
        self.feature_names = None

    def update_params(self, params):
        """
        Actualiza los parámetros del modelo con un diccionario.

        Parámetros:
        params: dict, diccionario de hiperparámetros
        """
        # Actualiza los parámetros en el modelo interno de Gradient Boosting.
        self.model.set_params(**params)
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def fit(self, X_train, y_train, feature_names=None):
        """
        Entrena el modelo con los datos de entrenamiento

        Parámetros:
        X_train: array-like, datos de entrenamiento preprocesados
        y_train: array-like, etiquetas de entrenamiento
        feature_names: list, nombres de las características (opcional)
        """
        # Asigna los nombres de las características, si se proporcionan.
        self.feature_names = feature_names if feature_names is not None else [f'Feature_{i}' for i in range(X_train.shape[1])]
        self.model.fit(X_train, y_train)

    def evaluate(self, X_train, y_train, X_test, y_test):
        """
        Evalúa el modelo en el conjunto de entrenamiento y prueba,
        y retorna una tabla comparativa de métricas.

        Parámetros:
        - X_train: array-like, características del conjunto de entrenamiento.
        - y_train: array-like, etiquetas del conjunto de entrenamiento.
        - X_test: array-like, características del conjunto de prueba.
        - y_test: array-like, etiquetas del conjunto de prueba.
        Retorna:
        - metrics_df: DataFrame con métricas de rendimiento para entrenamiento y prueba.
        """
        #Predicciones y métricas para el conjunto de entrenamiento
        y_train_pred = self.model.predict(X_train)
        y_train_pred_proba = self.model.predict_proba(X_train)[:, 1]
        train_metrics = {
            'Set': 'Train',
            'ROC AUC Score': roc_auc_score(y_train, y_train_pred_proba),
            'Classification Report': classification_report(y_train, y_train_pred, output_dict=True),
            'Confusion Matrix': confusion_matrix(y_train, y_train_pred)
        }

        #Predicciones y métricas para el conjunto de prueba
        y_test_pred = self.model.predict(X_test)
        y_test_pred_proba = self.model.predict_proba(X_test)[:, 1]
        test_metrics = {
            'Set': 'Test',
            'ROC AUC Score': roc_auc_score(y_test, y_test_pred_proba),
            'Classification Report': classification_report(y_test, y_test_pred, output_dict=True),
            'Confusion Matrix': confusion_matrix(y_test, y_test_pred)
        }

        # Crear DataFrame para organizar los resultados de entrenamiento y prueba
        metrics_df = pd.DataFrame([train_metrics, test_metrics])

        return metrics_df

    def predict(self, X_test):
        """
        Predice probabilidades para nuevos datos
        Parámetros:
        - X_test: array-like, datos para los que se harán las predicciones.
        Retorna:
        - Predicciones de clase para los datos de entrada.
        """
        return self.model.predict(X_test)

    def predict_proba(self, X):
        """
        Predice probabilidades para nuevos datos
        Parámetros:
        - X: array-like, datos para los que se calcularán las probabilidades.
        Retorna:
        - Probabilidades de la clase positiva para los datos de entrada.
        """
        return self.model.predict_proba(X)[:, 1]

    def plot_feature_importance(self, top_n=10):
        """
        Visualiza las características más importantes del modelo
        Parámetros:
        - top_n: int, número de características principales a mostrar.
        """
        # Ordena las características por importancia y selecciona las principales
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        # Gráfico de barras para las características más importantes
        plt.figure(figsize=(8, 6))
        sns.barplot(data=feature_importance.head(top_n), x='importance', y='feature')
        plt.tight_layout()
        plt.show()



    def plot_confusion_matrix(self, y_true, y_pred=None):
        """
        Visualiza la matriz de confusión
        Parámetros:
        - y_true: array-like, valores reales de las etiquetas.
        - y_pred: array-like, valores predichos de las etiquetas (opcional).
        """
        if y_pred is None:
            y_pred = self.model.predict(y_true)
            # Genera y grafica la matriz de confusión
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
        Parámetros:
        - deep: booleano, indica si se incluyen todos los parámetros (para compatibilidad con sklearn).
        Retorna:
        - dict: Diccionario con los hiperparámetros del modelo.
        """
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'random_state': self.random_state
        }

    def set_params(self, **params):
        """
        Actualiza los parámetros del modelo.
        Parámetros:
        - params: dict, diccionario con los nuevos valores de los hiperparámetros.
        Retorna:
        - self: instancia actualizada de la clase.
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
                # También actualiza el modelo interno de GradientBoostingClassifier si es necesario
                if key in ['n_estimators', 'learning_rate', 'max_depth', 'random_state']:
                    self.model.set_params(**{key: value})
        return self
