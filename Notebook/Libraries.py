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

        # Si 'Sesgo' es un archivo o módulo local, asegúrate de importarlo
        try:
            from Codigo import Sesgo
            from Codigo.Sesgo import stats, biased_variables
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

