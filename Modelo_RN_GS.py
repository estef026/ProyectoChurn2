import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split


class ModeloRedNeuronal:
    def __init__(self, input_shape, neurons_1 = 12, neurons_2 = 8, learning_rate = 0.001):
        self.model = Sequential()
        self.model.add(Dense(neurons_1, activation='relu', input_shape=(input_shape,)))
        self.model.add(Dense(neurons_2, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy', 'precision', 'recall'])
        self.model.summary()

    def train(self, X, y, test_size=0.2, epochs=50, batch_size=32):
        # Dividir los datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Entrenar el modelo directamente sin escalar los datos
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                                 validation_data=(X_test, y_test))
        return history, X_test, y_test

    def evaluate(self, X_test, y_test):
        # Evaluar el modelo directamente sin escalar los datos
        loss, accuracy = self.model.evaluate(X_test, y_test)
        return loss, accuracy

    def predict(self, X_new):
        # Realizar predicciones directamente sin escalar los datos
        predictions = (self.model.predict(X_new) > 0.5).astype("int32")
        return predictions