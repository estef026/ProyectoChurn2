import subprocess
import sys

class LibraryInstaller:
    def __init__(self, libraries):
        self.libraries = libraries

    # Función para actualizar pip
    def update_pip(self):
        print("\nVerificando si hay una nueva versión de pip...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
            print("pip ha sido actualizado correctamente.")
        except subprocess.CalledProcessError:
            print("Hubo un error al intentar actualizar pip.")

    # Función para instalar librerías si no están instaladas
    def install_libraries(self):
        missing_libraries = []
        for lib in self.libraries:
            try:
                __import__(lib)  # Verificamos si la librería está instalada
                print(f"{lib} ya está instalada.")
            except ImportError:
                missing_libraries.append(lib)  # Si no está, la agregamos a la lista
                print(f"{lib} no está instalada.")

        if missing_libraries:
            print("\nInstalando las siguientes librerías:")
            print(", ".join(missing_libraries))
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_libraries)
                print("\nLas librerías han sido instaladas correctamente.")
            except subprocess.CalledProcessError:
                print("\nHubo un error al intentar instalar las librerías. Verifique el entorno.")
        else:
            print("\nTodas las librerías necesarias ya están instaladas.")

    # Función para importar las librerías necesarias
    def import_libraries(self):
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import scipy
        import openpyxl as op
        from numpy.lib.shape_base import column_stack
        from scipy.stats import skew
        from statistics import mean, stdev
        import os
        from sklearn.preprocessing import StandardScaler
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        from scikeras.wrappers import KerasClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import accuracy_score, classification_report

        # Si 'Sesgo' es un archivo o módulo local, asegúrate de importarlo
        try:
            import Sesgo
            from Sesgo import stats, biased_variables
        except ImportError:
            print("El módulo 'Sesgo' no se encuentra. Verifica que esté en tu proyecto.")

    # Función principal para ejecutar
    def execute(self):
        self.update_pip()           # Actualizar pip
        self.install_libraries()    # Instalar librerías faltantes
        self.import_libraries()     # Importar librerías necesarias
        print("\nTodo está instalado correctamente y actualizado. ¡Listo para trabajar!")

# Lista de librerías a instalar
libraries = [
    'pandas',
    'matplotlib',
    'seaborn',
    'numpy',
    'scipy',
    'openpyxl',
    'scikit-learn',
    'tensorflow',
    'imbalanced-learn',
    'pycaret',
    'scikeras',
    'Sesgo'
]

# Si "Sesgo" es una librería externa, agrégala aquí:
# 'Sesgo'

# Crear una instancia de la clase y ejecutar
installer = LibraryInstaller(libraries)
installer.execute()


