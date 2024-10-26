import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


class GradientBoostingChurn:
    """
    Clase para predecir el abandono (churn) de clientes bancarios usando Gradient Boosting
    """

    def _init_(self):
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = None

    def preprocess_data(self, X, categorical_features=None):
        """
        Preprocesa nuevos datos usando los transformadores ajustados
        """
        X_processed = X.copy()

        # Procesar características categóricas
        if categorical_features:
            for column in categorical_features:
                if column in X_processed.columns:
                    if column not in self.label_encoders:
                        self.label_encoders[column] = LabelEncoder()
                        X_processed[column] = self.label_encoders[column].fit_transform(X_processed[column])
                    else:
                        X_processed[column] = self.label_encoders[column].transform(X_processed[column])

        # Procesar características numéricas
        numeric_features = X_processed.select_dtypes(include=['int64', 'float64']).columns
        X_processed[numeric_features] = self.scaler.transform(X_processed[numeric_features])

        return X_processed

    def fit(self, X, y, categorical_features=None, test_size=0.2, random_state=42):
        """
        Entrena el modelo con los datos proporcionados

        Parámetros:
        X: DataFrame con las características
        y: Series con la variable objetivo
        categorical_features: lista de nombres de columnas categóricas
        test_size: proporción de datos para prueba
        random_state: semilla aleatoria

        Retorna:
        dict con métricas de rendimiento del modelo
        """
        # Preprocesar datos
        self.feature_names = X.columns
        X_processed = self.preprocess_data(X, categorical_features)

        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=test_size, random_state=random_state
        )

        # Entrenar modelo
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=random_state
        )

        self.model.fit(X_train, y_train)

        # Evaluar modelo
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        # Calcular métricas
        metrics = {
            'classification_report': classification_report(y_test, y_pred),
            'roc_auc_score': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_importance': pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        }

        return metrics

    def predict_proba(self, X, categorical_features=None):
        """
        Predice la probabilidad de abandono para nuevos clientes

        Parámetros:
        X: DataFrame con los datos de los nuevos clientes
        categorical_features: lista de nombres de columnas categóricas

        Retorna:
        Array con probabilidades de abandono
        """
        if self.model is None:
            raise ValueError("El modelo debe ser entrenado antes de hacer predicciones")

        X_processed = self.preprocess_data(X, categorical_features)
        return self.model.predict_proba(X_processed)[:, 1]

    def plot_feature_importance(self, top_n=10):
        """
        Visualiza las características más importantes del modelo

        Parámetros:
        top_n: número de características principales a mostrar
        """
        if self.model is None:
            raise ValueError("El modelo debe ser entrenado antes de visualizar la importancia de características")

        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance.head(top_n), x='importance', y='feature')
        plt.title(f'Top {top_n} Características más Importantes')
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred):
        """
        Visualiza la matriz de confusión
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title('Matriz de Confusión')
        plt.ylabel('Valor Real')
        plt.xlabel('Valor Predicho')
        plt.show()