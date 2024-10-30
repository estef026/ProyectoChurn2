import numpy as np
from pandas.core.common import random_state
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import AUC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt



class ModeloRedNeuronal:
    def __init__(self, input_shape):
        self.model = Sequential()
        self.model.add(Dense(128, activation='relu', input_shape=(input_shape,)))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['recall', 'precision', AUC(name='auc')])
        self.model.summary()


    def train(self, X, y, test_size=0.2, epochs=50, batch_size=32):
        # Dividir los datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Entrenar el modelo directamente sin escalar los datos
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                                 validation_data=(X_test, y_test))
        return history, X_test, y_test, X_train, y_train

    def evaluate(self, X_test, y_test):
        # Evaluar el modelo directamente sin escalar los datos
        auc, loss, precision, recall = self.model.evaluate(X_test, y_test)
        return auc, loss, precision, recall

    def prediction(self, X_new):
        # Realizar predicciones directamente sin escalar los datos
        predictions = (self.model.predict(X_new) > 0.5).astype("int32")
        return predictions

    def confusion_matrix(self, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Matriz de Confusión")
        plt.xlabel("Predicciones")
        plt.ylabel("Valores Verdaderos")
        plt.show()
        return cm

    def classification_report(self, y_test, y_pred):
        report = classification_report(y_test, y_pred)
        print("Informe de Clasificación:\n", report)
        return report