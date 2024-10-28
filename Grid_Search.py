# grid_search.py

from sklearn.model_selection import GridSearchCV


class Tuner:
    def __init__(self, model, param_grid=None, score='accuracy', cv=5):
        """
        Inicializa el tuner para ajustar hiperparámetros de un modelo usando GridSearchCV.

        Parámetros:
        - model: El modelo base que se optimizará (ejemplo: AdaBoostClassifier).
        - param_grid: Diccionario de hiperparámetros a ajustar.
                      Si es None, se utilizan parámetros predefinidos según el modelo.
        - score: La métrica a optimizar en el GridSearch (por defecto 'accuracy').
        - cv: Número de folds para la validación cruzada (por defecto 5).
        """
        self.model = model  # Modelo a tunear
        # Establece los parámetros a ajustar, utilizando parámetros predeterminados si no se proporciona ninguno
        self.param_grid = param_grid if param_grid else self._default_params(model)
        self.score = score  # Métrica a utilizar para evaluar el modelo
        self.cv = cv  # Número de folds para validación cruzada
        # Inicializa GridSearchCV con el modelo, los parámetros, la métrica de puntuación y el número de folds
        self.grid = GridSearchCV(estimator=self.model, param_grid=self.param_grid, scoring=self.score, cv=self.cv)

    def _default_params(self, model):
        """
        Devuelve un conjunto de parámetros predeterminados según el modelo especificado.

        Parámetros:
        - model: El modelo para el cual se desean los parámetros predeterminados.

        Retorna:
        - Un diccionario de parámetros predeterminados para el modelo especificado.
        """
        name = type(model).__name__  # Obtiene el nombre de la clase del modelo

        # Diccionario de parámetros predeterminados para diferentes modelos
        params = {
            'AdaBoostClassifier': {
                'n_estimators': [50, 100, 200, 300],  # Número de estimadores
                'learning_rate': [0.01, 0.05, 0.1, 0.5, 1]  # Tasa de aprendizaje
            },
            'GradientBoostingClassifier': {
                'n_estimators': [50, 100, 200, 300],  # Número de estimadores
                'learning_rate': [0.01, 0.05, 0.1, 0.5],  # Tasa de aprendizaje
                'max_depth': [3, 5, 7, 9],  # Profundidad máxima de los árboles
                'subsample': [0.7, 0.8, 0.9, 1.0]  # Proporción de muestras utilizadas para entrenar cada árbol
            },
            'MLPClassifier': {  # Clasificador de Red Neuronal
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],  # Tamaño de las capas ocultas
                'activation': ['relu', 'tanh'],  # Función de activación
                'alpha': [0.0001, 0.001, 0.01, 0.1],  # Parámetro de regularización L2
                'learning_rate_init': [0.001, 0.01, 0.1],  # Tasa de aprendizaje inicial
                'max_iter': [200, 500, 1000]  # Número máximo de iteraciones
            },
            'SVC': {
                'C': [0.1, 1, 10, 100],  # Parámetro de regularización
                'kernel': ['linear', 'rbf', 'poly'],  # Tipo de kernel
                'gamma': ['scale', 'auto'],  # Coeficiente para el kernel rbf
                'degree': [2, 3, 4]  # Grado del polinomio (solo para kernel polinómico)
            }
        }

        # Si el nombre del modelo está en los parámetros predeterminados, devuelve los parámetros correspondientes
        if name in params:
            return params[name]
        else:
            raise ValueError(f"No hay parámetros predeterminados para el modelo: {name}")

    def tune(self, X, y):
        """
        Ajusta el GridSearchCV en los datos de entrenamiento y retorna el mejor modelo encontrado.

        Parámetros:
        - X: Conjunto de características para el entrenamiento.
        - y: Conjunto de etiquetas correspondientes a las características.

        Retorna:
        - El mejor modelo encontrado por el GridSearchCV.
        """
        self.grid.fit(X, y)  # Ajusta el GridSearchCV en los datos de entrenamiento
        print("Mejores parámetros:", self.grid.best_params_)  # Muestra los mejores parámetros encontrados
        return self.grid.best_estimator_  # Retorna el mejor estimador
