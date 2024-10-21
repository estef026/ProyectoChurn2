from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Modelo_RN:
    def __init__(self, datos, objetivo):
        self.datos = datos
        self.objetivo = objetivo
        self.X_entrenamiento, self.X_prueba, self.y_entrenamiento, self.y_prueba = train_test_split(
            datos, objetivo, test_size=0.2, random_state=42, stratify=True
        )

    def create_model(data_entrada):
        model = keras.Sequential([
            layers.Dense(128, activation="relu", input_dim= data_entrada, name="hidden-dense-128-layer-1"),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu", name="hidden-dense-64-layer-2"),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid", name="output-layer"),
        ])

        adam = tf.keras.optimizers.Adam()
        model.compile(loss="binary_crossentropy", optimizer=adam, metrics=["accuracy"])
        return model