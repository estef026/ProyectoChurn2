from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score


class Tuner:
    def __init__(self, model, param_grid=None, score='accuracy', cv=5, use_random_search=False, n_iter=10):
        """
        Inicializa el tuner para ajustar hiperparámetros de un modelo.

        Parámetros:
        - model: El modelo base que se optimizará (ejemplo: AdaBoostModel).
        - param_grid: Diccionario de hiperparámetros a ajustar.
        - score: La métrica a optimizar en el GridSearch (por defecto 'accuracy').
        - cv: Número de folds para la validación cruzada (por defecto 5).
        - use_random_search: Si True, utiliza RandomizedSearchCV en lugar de GridSearchCV.
        - n_iter: Número de combinaciones a probar en RandomizedSearchCV.
        """
        self.model = model
        # Si no se proporciona un grid, usa parámetros predeterminados según el modelo
        self.param_grid = param_grid if param_grid else self.default_params(model)
        self.score = score
        self.cv = cv
        self.use_random_search = use_random_search
        self.n_iter = n_iter

        # Inicializa la búsqueda hiperparámetros usando GridSearchCV o RandomizedSearchCV
        if self.use_random_search: # Verifica si se ha especificado usar RandomizedSearchCV
            self.grid = RandomizedSearchCV(
                estimator=self.model, # Modelo base al cual se ajustarán los hiperparámetros
                param_distributions=self.param_grid,  # Espacio de búsqueda de hiperparámetros a probar
                n_iter=self.n_iter,  # Espacio de búsqueda de hiperparámetros a probar
                scoring=self.score, # Métrica para evaluar cada combinación de hiperparámetros
                cv=self.cv, # Número de folds para la validación cruzada
                n_jobs=-1, # Usa todos los núcleos disponibles para paralelizar la búsqueda
                random_state=42,  # Fija la semilla para asegurar reproducibilidad en los resultados
                verbose=1  # Muestra el progreso de la búsqueda en la consola
            )
        else:
            # Si use_random_search es False, utiliza GridSearchCV para búsqueda exhaustiva
            self.grid = GridSearchCV(
                estimator=self.model, # Modelo base al cual se ajustarán los hiperparámetros
                param_grid=self.param_grid, # Espacio de búsqueda de hiperparámetros (todas las combinaciones posibles)
                scoring=self.score, # Métrica para evaluar cada combinación de hiperparámetros
                cv=self.cv, # Número de folds para la validación cruzada
                n_jobs=-1, # Usa todos los núcleos disponibles para paralelizar la búsqueda
                verbose=1  # Muestra el progreso de la búsqueda en la consola
            )

    def default_params(self, model):
        """
                Devuelve un conjunto de hiperparámetros predeterminados para el modelo especificado.
                Parámetros:
                - model: objeto del modelo que se optimizará.
                Retorna:
                - dict: diccionario de parámetros predeterminados según el tipo de modelo.
                Lanza:
                - ValueError: si el modelo no tiene parámetros predeterminados definidos.
                """
        name = type(model).__name__
        print(f"Modelo: {name}")
        # Definición de un diccionario 'params' que contiene los espacios de búsqueda de hiperparámetros para diferentes modelos
        params = {
            # Hiperparámetros para el modelo 'AdaBoostChurn'
            'AdaBoostChurn': {
                'n_estimators': [50, 100, 200], # Número de árboles en el ensamble
                'learning_rate': [0.01, 0.1], # Tasa de aprendizaje que controla la contribución de cada clasificador débil
                'algorithm': ['SAMME'], # Algoritmo de impulso, aquí se utiliza 'SAMME' (opción para clasificación multicategórica)
                'estimator__max_depth': [1, 2, 3, 4], # Profundidad máxima de los árboles base (clasificadores débiles)
                'random_state': [42] # Semilla para reproducibilidad
            },
            # Hiperparámetros para el modelo 'GradientBoostingChurn'
            'GradientBoostingChurn': {
                'n_estimators': [ 150, 200, 300],  # Número de árboles en el ensamble
                'learning_rate': [ 0.05, 0.1, 5], # Tasa de aprendizaje que controla el impacto de cada árbol
                'max_depth': [ 5, 7, 9],  # Profundidad máxima de los árboles, controla la complejidad del modelo
                'subsample': [0.5, 1.0, 5]  # Porcentaje de muestras utilizadas en cada árbol, controla la variación
            },
            # Hiperparámetros para el modelo 'KerasClassifier'
            'KerasClassifier': {
                'batch_size': [32, 64, 128],  # Tamaño del lote para el entrenamiento de la red neuronal
                'epochs': [15, 10, 5], # Número de épocas o iteraciones de entrenamiento completas sobre los datos
                'optimizer': ['adam', 'rmsprop'], # Optimizador para actualizar los pesos durante el entrenamiento
                'optimizer__learning_rate': [0.001, 0.01, 0.1]  # Tasa de aprendizaje para el optimizador
            },
            # Hiperparámetros para el modelo 'SVMChurn'
            'SVMChurn': {
                'C': [0.1, 1, 10], # Parámetro de regularización; valores más altos tienden a ajustar más el modelo
                'kernel': ['linear', 'rbf'], # Tipo de función núcleo utilizada para transformar los datos
                'gamma': ['scale', 'auto'],  # Parámetro que controla la influencia de los puntos de datos en el modelo
                'class_weight': [None, 'balanced'],  # Ajuste de los pesos de las clases para manejar clases desbalanceadas
                'random_state': [42] # Semilla para asegurar reproducibilidad en la división de datos
            }
        }
        # Retorna los parámetros si el modelo está en el diccionario; si no, lanza un error
        if name in params:
            return params[name]
        else:
            raise ValueError(f"No hay parámetros predeterminados para el modelo: {name}")

    def tune(self, X, y):
        """
                Ejecuta la búsqueda de hiperparámetros usando el grid configurado y retorna los mejores parámetros.
                Parámetros:
                - X: array-like, matriz de características de entrada.
                - y: array-like, etiquetas de salida.
                Retorna:
                - dict: mejores hiperparámetros encontrados.
                """
        # Ajusta el modelo en los datos de entrenamiento usando validación cruzada
        self.grid.fit(X, y)
        print("Mejores parámetros:", self.grid.best_params_)
        return self.grid.best_params_

    def tune_nn(self, X, y):
        """
                Ajusta los hiperparámetros de un modelo de red neuronal (por ejemplo, KerasClassifier) y retorna los mejores parámetros.
                Parámetros:
                - X: array-like, matriz de características de entrada.
                - y: array-like, etiquetas de salida.
                Retorna:
                - dict: mejores hiperparámetros encontrados.
                """
        # Ajusta el modelo de red neuronal, suprimiendo el verbose
        self.grid.fit(X, y, verbose = 0)
        print("Mejores parámetros:", self.grid.best_params_)
        return self.grid.best_params_