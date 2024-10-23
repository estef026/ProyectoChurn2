import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import AUC


class ModeloRedNeuronal:
    def __init__(self, input_shape):
        self.model = Sequential()
        self.model.add(Dense(64, activation='relu', input_shape=(input_shape,)))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['recall', 'precision', AUC(name='auc')])
        self.model.summary()
        return self.model.summary()

    def train(self, X, y, test_size=0.2, epochs=50, batch_size=32):
        # Dividir los datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Entrenar el modelo directamente sin escalar los datos
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                                 validation_data=(X_test, y_test))
        return history, X_test, y_test, X_train, y_train

    def evaluate(self, X_test, y_test):
        # Evaluar el modelo directamente sin escalar los datos
        loss, precision, recall, auc = self.model.evaluate(X_test, y_test)
        return loss, precision, recall, auc

    def predict(self, X_new):
        # Realizar predicciones directamente sin escalar los datos
        predictions = (self.model.predict(X_new) > 0.5).astype("int32")
        return predictions