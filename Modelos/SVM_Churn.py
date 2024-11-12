from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class SVMChurn:
    """
    Clase para entrenar y evaluar un modelo SVM (Support Vector Machine) para predicción de churn.
    """

    def __init__(self, C=1.0, kernel='rbf', gamma='scale', class_weight=None, random_state=42):
        """
        Inicializa el modelo de SVM con los hiperparámetros especificados.

        Parámetros:
        C: float, opción que controla la penalización por margen incorrecto.
        kernel: {'linear', 'poly', 'rbf', 'sigmoid'}, tipo de kernel a usar.
        gamma: {'scale', 'auto'} o float, función de activación.
        class_weight: dict o 'balanced', ajusta el peso de las clases.
        random_state: int, controla la aleatoriedad.
        """
        self.C = C # Regularización para el margen del SVM
        self.kernel = kernel # Tipo de kernel a usar (rbf, lineal, etc.)
        self.gamma = gamma # Función de activación (determina cómo los puntos están relacionados)
        self.class_weight = class_weight # Ajuste de pesos para clases desequilibradas
        self.random_state = random_state # Valor para controlar la aleatoriedad en el modelo

        # Crear el modelo SVM con los parámetros dados
        self.model = SVC(
            C=self.C,
            kernel=self.kernel,
            gamma=self.gamma,
            class_weight=self.class_weight,
            random_state=self.random_state,
            probability=True  # Habilitar el cálculo de probabilidades para la curva ROC AUC
        )
        self.feature_names = None # Atributo para almacenar los nombres de las características.

    def fit(self, X_train, y_train, feature_names=None):
        """
        Entrena el modelo con los datos de entrenamiento.

        Parámetros:
        X_train: array-like, datos de entrenamiento preprocesados.
        y_train: array-like, etiquetas de entrenamiento.
        feature_names: list, nombres de las características (opcional).
        """
        # Si se proporcionan los nombres de las características, se almacenan; de lo contrario, se genera un nombre genérico.
        self.feature_names = feature_names if feature_names is not None else [f'Feature_{i}' for i in range(X_train.shape[1])]
        # Entrena el modelo SVM utilizando los datos de entrenamiento.
        self.model.fit(X_train, y_train)

    def evaluate(self, X_train, y_train, X_test, y_test):
        """
        Evalúa el modelo en los conjuntos de entrenamiento y prueba.

        Retorna:
        dict con métricas de rendimiento para ambos conjuntos de datos
        """
        # Predicciones en el conjunto de entrenamiento
        y_train_pred = self.model.predict(X_train)
        y_train_pred_proba = self.model.predict_proba(X_train)[:, 1] # Probabilidades para la clase positiva
        # Métricas para el conjunto de entrenamiento
        train_metrics = {
            'classification_report_train': classification_report(y_train, y_train_pred), # Reporte de clasificación
            'roc_auc_score_train': roc_auc_score(y_train, y_train_pred_proba), # AUC-ROC
            'confusion_matrix_train': confusion_matrix(y_train, y_train_pred) # Matriz de confusión
        }

        # Evaluar en el conjunto de prueba
        y_test_pred = self.model.predict(X_test)
        y_test_pred_proba = self.model.predict_proba(X_test)[:, 1] # Probabilidades para la clase positiva
        # Métricas para el conjunto de prueba
        test_metrics = {
            'classification_report_test': classification_report(y_test, y_test_pred), # Reporte de clasificación
            'roc_auc_score_test': roc_auc_score(y_test, y_test_pred_proba), # AUC-ROC
            'confusion_matrix_test': confusion_matrix(y_test, y_test_pred) # Matriz de confusión
        }

        # Combina las métricas de entrenamiento y prueba
        metrics = {**train_metrics, **test_metrics}

        return metrics

    def predict(self, X_test):
        """
        Predice las clases para nuevos datos.
        """
        return self.model.predict(X_test) # Realiza las predicciones para el conjunto de prueba.

    def predict_proba(self, X):
        """
        Predice probabilidades para nuevos datos.
        """
        return self.model.predict_proba(X)[:, 1] # Devuelve las probabilidades para la clase positiva.

    def plot_feature_importance(self, top_n=10):
        """
        Visualiza las características más importantes del modelo. Para SVM, usamos los coeficientes
        del kernel lineal.
        """
        if self.kernel == 'linear':
            # Verifica si el modelo SVM utiliza un kernel lineal. La importancia de las características solo se puede calcular para un kernel lineal.
            coef = self.model.coef_.flatten() # Obtiene los coeficientes del modelo SVM con kernel lineal.
            # Crea un DataFrame para almacenar las características y sus importancias (coeficientes absolutos)
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                # Crea una columna llamada 'feature' en el DataFrame, que contiene los nombres de las características almacenadas previamente.
                'importance': abs(coef)
                # Crea una columna llamada 'importance' que contiene el valor absoluto de los coeficientes. El valor absoluto es utilizado
                # porque estamos interesados en la magnitud del impacto de cada característica, sin importar la dirección del coeficiente.
            }).sort_values('importance', ascending=False) # Ordena por importancia descendente
            # Visualiza las top_n características más importantes
            plt.figure(figsize=(10, 6))
            sns.barplot(data=feature_importance.head(top_n), x='importance', y='feature')
            plt.title(f'Top {top_n} Características más Importantes')
            plt.tight_layout()
            plt.show()


        else:
            print("La visualización de importancia solo está disponible para kernels lineales.")
            return None

    def plot_confusion_matrix(self, X, y_true, y_pred=None):
        """
        Visualiza la matriz de confusión.
        """
        if y_pred is None:
            # Si no se proporcionan las predicciones, se predicen las clases utilizando el modelo
            y_pred = self.model.predict(X)
        # Configura el gráfico para la matriz de confusión
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred) # Calcula la matriz de confusión
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues') # Muestra la matriz con un mapa de calor
        plt.title('Matriz de Confusión') # Título del gráfico
        plt.ylabel('Valor Real') # Etiqueta del eje Y
        plt.xlabel('Valor Predicho') # Etiqueta del eje X
        plt.tight_layout() # Ajuste de diseño
        plt.show() # Muestra el gráfico

    def get_params(self, deep=True):
        """
        Devuelve los parámetros del modelo.
        """
        return {
            'C': self.C,
            'kernel': self.kernel,
            'gamma': self.gamma,
            'class_weight': self.class_weight,
            'random_state': self.random_state
        } # Devuelve los parámetros del modelo en un diccionario.

    def set_params(self, **params):
        """
        Actualiza los parámetros del modelo.
        """
        for key, value in params.items():
            # Actualiza los parámetros internos del modelo si existen
            if hasattr(self, key):
                setattr(self, key, value)
                # Actualiza los parámetros correspondientes en el modelo SVM
                if key in ['C', 'kernel', 'gamma', 'class_weight', 'random_state']:
                    self.model.set_params(**{key: value})
        return self # Devuelve la instancia del objeto actualizado.
