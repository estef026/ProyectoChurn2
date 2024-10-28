from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score


class Tuner:
    def __init__(self, model, param_grid=None, score='accuracy', cv=5, use_random_search=False, n_iter=10):
        """
        Inicializa el tuner para ajustar hiperparámetros de un modelo.

        Parámetros:
        - model: El modelo base que se optimizará (ejemplo: AdaBoostClassifier).
        - param_grid: Diccionario de hiperparámetros a ajustar.
        - score: La métrica a optimizar en el GridSearch (por defecto 'accuracy').
        - cv: Número de folds para la validación cruzada (por defecto 5).
        - use_random_search: Si True, utiliza RandomizedSearchCV en lugar de GridSearchCV.
        - n_iter: Número de combinaciones a probar en RandomizedSearchCV.
        """
        self.model = model
        self.param_grid = param_grid if param_grid else self.default_params(model)
        self.score = score
        self.cv = cv
        self.use_random_search = use_random_search
        self.n_iter = n_iter

        # Inicializa el tipo de búsqueda
        if self.use_random_search:
            self.grid = RandomizedSearchCV(
                estimator=self.model,
                param_distributions=self.param_grid,
                n_iter=self.n_iter,
                scoring=self.score,
                cv=self.cv,
                n_jobs=-1,
                random_state=42,  # Para reproducibilidad
                verbose=1  # Muestra el progreso
            )
        else:
            self.grid = GridSearchCV(
                estimator=self.model,
                param_grid=self.param_grid,
                scoring=self.score,
                cv=self.cv,
                n_jobs=-1,
                verbose=1  # Muestra el progreso
            )

    def default_params(self, model):
        name = type(model).__name__
        print(f"Modelo: {name}")

        params = {
            'AdaBoostClassifier': {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1]
            },
            'GradientBoostingChurn': {
                'n_estimators': [100, 150],  # Ajustado para menos combinaciones
                'learning_rate': [0.01, 0.05],
                'max_depth': [3, 4],  # Ajustado
                'subsample': [0.8, 1.0]  # Ajustado
            },
            'MLPClassifier': {
                'hidden_layer_sizes': [(50,), (100,)],
                'activation': ['relu'],
                'alpha': [0.0001],
                'learning_rate_init': [0.001],
                'max_iter': [200]
            },
            'SVC': {
                'C': [0.1, 1],
                'kernel': ['linear'],
                'gamma': ['scale']
            }
        }

        if name in params:
            return params[name]
        else:
            raise ValueError(f"No hay parámetros predeterminados para el modelo: {name}")

    def tune(self, X, y):
        self.grid.fit(X, y)
        print("Mejores parámetros:", self.grid.best_params_)
        return self.grid.best_params_

# Ejemplo de uso
# gb_tuner = Tuner(model_GB, use_random_search=True, n_iter=10)
# best_params = gb_tuner.tune(X_train, y_train)
