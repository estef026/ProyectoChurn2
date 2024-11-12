import numpy as np
from pandas.core.common import random_state
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import AUC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt



class ModeloRedNeuronal:
    def __init__(self, input_shape):
        self.input_shape = input_shape # Asigna la forma de entrada del modelo al atributo de la clase.

    def create_model(self, input_shape):
        self.model = Sequential() # Inicializa el modelo secuencial, donde se añadirán las capas.
        # Añade una capa densa con 128 neuronas, activación ReLU y especifica la forma de entrada
        self.model.add(Dense(128, activation='relu', input_shape=(input_shape,))) # Añade una capa de abandono (dropout) con un 10% de probabilidad para evitar el sobreajuste.
        self.model.add(Dropout(0.1))
        # Añade una segunda capa densa con 64 neuronas y activación ReLU
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.1)) # Añade otra capa de abandono (dropout) con un 10% de probabilidad.
        # Añade una capa de salida con una sola neurona y activación sigmoide para clasificación binaria
        self.model.add(Dense(1, activation='sigmoid'))
        # Compila el modelo especificando el optimizador RMSprop, la función de pérdida binary_crossentropy y las métricas a evaluar
        self.model.compile(optimizer=RMSprop(learning_rate=0.1), loss='binary_crossentropy', metrics=['recall', 'precision', AUC(name='auc'), 'accuracy'])
        self.model.summary() # Muestra el resumen del modelo, detallando la arquitectura y los parámetros.


    def create_best_model(self, input_shape):
        self.model = Sequential() # Inicializa el modelo secuencial nuevamente para crear otro modelo.
        self.model.add(Dense(128, activation='relu', input_shape=(input_shape,))) # Capa densa con 128 neuronas.
        self.model.add(Dropout(0.1)) # Capa de abandono (dropout) para evitar el sobreajuste.
        self.model.add(Dense(64, activation='relu')) # Capa densa con 64 neuronas.
        self.model.add(Dropout(0.1)) # Otra capa de abandono (dropout).
        self.model.add(Dense(1, activation='sigmoid')) # Capa de salida para clasificación binaria con sigmoide.
        # Compila el modelo con el optimizador Adam y una tasa de aprendizaje más baja (0.001)
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['recall', 'precision', AUC(name='auc'), 'accuracy'])
        self.model.summary() # Muestra el resumen del modelo.

    def train(self, X, y, test_size=0.2, epochs=50, batch_size=64):
        """
                Entrena el modelo con los datos proporcionados.
                Parámetros:
                - X: Características del conjunto de datos.
                - y: Etiquetas del conjunto de datos.
                - test_size: Proporción del conjunto de datos que se usará para validación.
                - epochs: Número de épocas de entrenamiento.
                - batch_size: Tamaño del lote para el entrenamiento.
                """
        # Divide los datos en conjuntos de entrenamiento y prueba utilizando el 20% para prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Entrena el modelo usando los datos de entrenamiento y validación
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                                 validation_data=(X_test, y_test)) # Pasa el conjunto de prueba para validación durante el entrenamiento.
        return history, X_test, y_test, X_train, y_train # Devuelve el historial de entrenamiento y los conjuntos de datos.

    def evaluate(self, X_test, y_test):
        """
                Evalúa el modelo en el conjunto de prueba.

                Parámetros:
                - X_test: Características del conjunto de prueba.
                - y_test: Etiquetas del conjunto de prueba.
                Retorna:
                - auc, loss, precision, recall, accuracy: Las métricas de evaluación del modelo.
                """
        # Evalúa el modelo directamente sin escalar los datos
        auc, loss, precision, recall, accuracy = self.model.evaluate(X_test, y_test)
        return auc, loss, precision, recall, accuracy

    def prediction(self, X_new):
        """
                Realiza predicciones con el modelo.

                Parámetros:
                - X_new: Características de los nuevos datos para los que se quiere hacer predicción.

                Retorna:
                - predictions: Predicciones del modelo (0 o 1).
                """
        # Realiza predicciones y convierte las probabilidades en clases (0 o 1) usando el umbral de 0.5
        predictions = (self.model.predict(X_new) > 0.5).astype("int32")
        return predictions # Devuelve las predicciones (0 o 1).

    def confusion_matrix(self, y_test, y_pred):
        """
                Genera la matriz de confusión y la visualiza.
                Parámetros:
                - y_test: Etiquetas reales del conjunto de prueba.
                - y_pred: Predicciones del modelo.
                Muestra:
                - Una matriz de confusión visualizada con un mapa de calor.
                """
        cm = confusion_matrix(y_test, y_pred) # Calcula la matriz de confusión entre las etiquetas reales y las predicciones.
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")  # Visualiza la matriz con un mapa de calor.
        plt.title("Matriz de Confusión")   #Título de la gráfica.
        plt.xlabel("Predicciones") #Etiqueta del eje X.
        plt.ylabel("Valores Verdaderos") # Etiqueta del eje Y.
        plt.show() #Muestra la gráfica.


    def classification_report(self, y_test, y_pred):
        """
                Genera un informe de clasificación con métricas como precisión, recall, f1-score, etc.
                Parámetros:
                - y_test: Etiquetas reales del conjunto de prueba.
                - y_pred: Predicciones del modelo.
                """
        report = classification_report(y_test, y_pred) # Genera el informe de clasificación.
        print("Informe de Clasificación:\n", report) # Imprime el informe de clasificación.

