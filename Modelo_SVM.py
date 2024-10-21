import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class ModeloSVM:
    def __init__(self, datos, objetivo):
        self.datos = datos
        self.objetivo = objetivo
        self.X_entrenamiento, self.X_prueba, self.y_entrenamiento, self.y_prueba = train_test_split(
            datos, objetivo, test_size=0.2, random_state=42
        )
    
    def limpieza(self):
        for col in self.X_entrenamiento.select_dtypes(include=[np.number]).columns:
            self.X_entrenamiento[col].fillna(self.X_entrenamiento[col].mean(), inplace=True)
            self.X_prueba[col].fillna(self.X_prueba[col].mean(), inplace=True)
        
        # Rellenar valores faltantes con la moda para columnas categóricas
        for col in self.X_entrenamiento.select_dtypes(include=[object]).columns:
            self.X_entrenamiento[col].fillna(self.X_entrenamiento[col].mode()[0], inplace=True)
            self.X_prueba[col].fillna(self.X_prueba[col].mode()[0], inplace=True)
        
    def transformaciones(self):
        # Transformar características categóricas con OneHotEncoder y estandarizar características numéricas
        caracteristicas_numericas = ['caracteristica_numerica1', 'caracteristica_numerica2']
        caracteristicas_categoricas = ['caracteristica_categorica']
        
        self.preprocesador = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), caracteristicas_numericas),
                ('cat', OneHotEncoder(), caracteristicas_categoricas)
            ]
        )
        
        self.X_entrenamiento = self.preprocesador.fit_transform(self.X_entrenamiento)
        self.X_prueba = self.preprocesador.transform(self.X_prueba)
        
    def preprocesar(self):
        self.limpieza()
        self.transformaciones()
        
    def entrenar(self):
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf', 'poly']
        }
        
        svm = SVC()
        self.busqueda_grid = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy')
        self.busqueda_grid.fit(self.X_entrenamiento, self.y_entrenamiento)
        
    def evaluar(self):
        y_pred = self.busqueda_grid.predict(self.X_prueba)
        exactitud = accuracy_score(self.y_prueba, y_pred)
        return exactitud
        
    def predecir(self, nuevos_datos):
        nuevos_datos_transformados = self.preprocesador.transform(nuevos_datos)
        predicciones = self.busqueda_grid.predict(nuevos_datos_transformados)
        return predicciones