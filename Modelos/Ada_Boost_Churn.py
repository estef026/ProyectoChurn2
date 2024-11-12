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
        Inicializa el modelo de AdaBoost con los hiperparámetros:
        Parámetros:
        - n_estimators: es el número de modelos débiles (árboles) que se entrenarán.
        - learning_rate: Tasa de aprendizaje que controla el peso de cada árbol.
        - algorithm: Algoritmo de boosting a utilizar (por defecto 'SAMME').
        - max_depth: Profundidad máxima del árbol base.
        - random_state: Semilla para garantizar reproducibilidad.
        """
        # Asignar hiperparámetros a atributos de instancia
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.algorithm = algorithm
        self.max_depth = max_depth
        self.random_state = random_state

        # Crear el estimador base, un DecisionTreeClassifier con el max_depth especificado
        base_estimator = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)

        # Crear el modelo de AdaBoost con el estimador base
        # Inicializar el modelo AdaBoost usando el estimador base y los hiperparámetros dados
        self.model = AdaBoostClassifier(
            estimator=base_estimator,  # Establecer el estimador base
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            algorithm=self.algorithm,
            random_state=self.random_state
        )
        # Inicializar el atributo para nombres de las características como None
        self.feature_names = None

    def fit(self, X_train, y_train, feature_names=None):
        """
        Entrena el modelo con los datos de entrenamiento.

        Parámetros:
        X_train: array-like, datos de entrenamiento preprocesados
        y_train: array-like, etiquetas de entrenamiento
        feature_names: list, nombres de las características (opcional)
        """
        # Asignar nombres de características si se proporcionan, de lo contrario asignar nombres genéricos
        self.feature_names = feature_names if feature_names is not None else [f'Feature_{i}' for i in range(X_train.shape[1])]
        # Entrena el modelo con los datos de entrenamiento
        self.model.fit(X_train, y_train)

    def evaluate(self, X_train, y_train, X_test, y_test):
        """
        Evalúa el modelo con los datos de entrenamiento y prueba.
        Parámetros:
        - X_train: Datos de entrenamiento.
        - y_train: Etiquetas de entrenamiento.
        - X_test: Datos de prueba.
        - y_test: Etiquetas de prueba.

        Retorna:
        dict con métricas de rendimiento para ambos conjuntos de datos
        """
        # Evalúa en el conjunto de entrenamiento
        # Predice etiquetas y probabilidades para el conjunto de entrenamiento
        y_train_pred = self.model.predict(X_train)
        y_train_pred_proba = self.model.predict_proba(X_train)[:, 1]
        # Calcula métricas para el conjunto de entrenamiento
        train_metrics = {
            'classification_report_train': classification_report(y_train, y_train_pred),
            'roc_auc_score_train': roc_auc_score(y_train, y_train_pred_proba),
            'confusion_matrix_train': confusion_matrix(y_train, y_train_pred)
        }

        # Evalúa en el conjunto de prueba
        # Predice etiquetas y probabilidades para el conjunto de prueba
        y_test_pred = self.model.predict(X_test)
        y_test_pred_proba = self.model.predict_proba(X_test)[:, 1]
        # Calcular métricas para el conjunto de prueba
        test_metrics = {
            'classification_report_test': classification_report(y_test, y_test_pred),
            'roc_auc_score_test': roc_auc_score(y_test, y_test_pred_proba),
            'confusion_matrix_test': confusion_matrix(y_test, y_test_pred)
        }

        # Combinar métricas de ambos conjuntos
        metrics = {**train_metrics, **test_metrics}
        # Retornar el diccionario con las métricas
        return metrics

    def predict(self, X_test):
        """
        Realiza predicciones de probabilidad de clase para nuevos datos.
        Parámetros:
        - X_test: array-like, datos de prueba.
        Retorna:
        - array: Predicciones de clase para X_test.
        """
        return self.model.predict(X_test)

    def predict_proba(self, X):
        """
        Predice probabilidades para nuevos datos.
        Parámetros:
        - X: array-like, datos para los que se desean las probabilidades.
        Retorna:
        - array: Probabilidad de la clase positiva para cada muestra en X.
        """
        return self.model.predict_proba(X)[:, 1]

    def plot_feature_importance(self, top_n=10):

        """
        Visualiza las características más importantes del modelo.
        Parámetros:
        - top_n: Número de características principales a mostrar (por defecto 10).
        """
        # Crea un DataFrame con las características y sus importancias
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        # Grafica las características principales en orden descendente
        plt.figure(figsize=(8, 6))
        sns.barplot(data=feature_importance.head(top_n), x='importance', y='feature')
        plt.tight_layout()
        plt.show()



    def plot_confusion_matrix(self, X, y_true, y_pred=None):
        """
        Visualiza la matriz de confusión.
        Parámetros:
        - X: array-like, datos de entrada.
        - y_true: array-like, etiquetas reales.
        - y_pred: array-like, etiquetas predichas (opcional).
        """
        if y_pred is None:
            # Predecir las clases usando las características
            # Si no se proporcionan predicciones, realizarlas con el modelo
            y_pred = self.model.predict(X)
        # Graficar la matriz de confusión
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
        - deep: bool, si se devuelven los parámetros de manera profunda (por defecto True).
        Retorna:
        - dict: Diccionario con los hiperparámetros del modelo.
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
        Parámetros:
        - params: Diccionario con los parámetros a actualizar.
        Retorna:
        - self: Instancia del modelo con parámetros actualizados.
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
                if key in ['n_estimators', 'learning_rate', 'algorithm', 'max_depth', 'random_state']:
                    self.model.set_params(**{key: value})
        return self
