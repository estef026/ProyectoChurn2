# Caso de negocio para la predicción de churn o tasa de abandono de clientes en un banco

**¿Que desarrollamos en el proyecto? 📈**

Se desarrollaron y evaluaron cuatro (4) modelos de *Machine Learning* para la predicción del *churn* o tasa de abandono 
de los clientes de un banco. Los datos utilizados están compuestos por 10.127 registros de clientes, que contienen su perfil
sociodemográfico, capacidad de endeudamiento, cantidad de productos financieros y otras variables
que miden la relación del cliente con el banco.

Los datos de entrada se encuentran en la ubicación `Datos/bank_churn.xlsx`

El proyecto contiene las siguientes secciones, desarrolladas en el Jupyter Notebook:

* EDA (Exploratory Data Analysis)
* Data preparation
* Modeling and evaluation


**Estructura del proyecto**

```bash
ProyectoChurn
├── Codigo
│   ├── Graficas_EDA.py
│   ├── Metodos_Oversampling.py
│   ├── Multivariado.py
│   ├── OutliersDetection.py
│   └── Sesgo.py
├── Datos
│   ├── bank_churn.xlsx
│   ├── Datos_finales_entrada_df_ro.csv
│   ├── DatosADASYN.csv
│   ├── DatosRandomOversampling.csv
│   ├── DatosSMOTE.csv
│   └── VariablesNumericas.csv
├── Modelos
│   ├── Ada_Boost_Churn.py
│   ├── Gradient_Boosting.py
│   ├── Grid_Search.py
│   ├── Grid_Search_NN.py
│   ├── Modelo_RN.py
│   └── SVM_Churn.py
├── Notebook
│   ├── Libraries.py
│   ├── logs.log
│   └── Notebook_Proyecto_Churn.ipynb
└─Readme.md
````
## Comenzando 🚀

**¡Recomendaciones!** ✅

* Uso de Python 3.11 como interprete. Esta versión es necesaria para utilizar la librería `Pycaret`
* Instalar las librerías contenidas en el archivo `Libraries.py`. La línea de código para su instalación
es la primera línea a ejecutar en el archivo `Notebook Proyecto Churn.ipynb`
* Conservar la estructura de las carpetas, especialmente para la ejecución de los modelos construidos.

## Autores:
  Andree Amahar Aaron Quiroz ‍🧞‍♂️

  Eva Estefania Martinez Castillo 🧜🏼‍♀️

  Yud Karem Rozo Avila  🦹‍♀️

  Zorayda Acevedo Fernandez 🧛‍♀️
  