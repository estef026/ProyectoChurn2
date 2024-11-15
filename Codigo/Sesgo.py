from scipy.stats import skew
from statistics import mean
from statistics import stdev
import numpy as np
import pandas as pd

def stats(data):
    """
       Calcula estadísticas descriptivas (coeficiente de asimetría, media y desviación estándar)
       para todas las columnas numéricas de un DataFrame.

       Parámetros:
       data : DataFrame que contiene los datos en los que se calcularán las estadísticas.

       Retorna:
           DataFrame con las siguientes columnas:
           - 'Variable': nombre de cada columna numérica en el DataFrame original.
           - 'Skew': coeficiente de asimetría de cada variable, indicando la simetría de su distribución.
           - 'Mean': media (promedio) de cada variable.
           - 'Standard deviation': desviación estándar de cada variable.
           """
    variables = data.select_dtypes(include=[np.number]).columns

    skew_coef = list()
    average = list()
    std = list()


    for i in variables:
        skew_coef.append(skew(data[i]))
        average.append(mean(data[i]))
        std.append(stdev(data[i]))

    statistics = pd.DataFrame()
    statistics['Variable'] = variables
    statistics['Skew'] = skew_coef
    statistics['Mean'] = average
    statistics['Standard deviation'] = std

    return statistics

def biased_variables(statistics):
    """
        Filtra y retorna las variables con una asimetría (skewness) significativa, es decir, aquellas
        cuyo coeficiente de asimetría es >= 0.5 o <= -0.5.

        Parámetros:

        statistics :DataFrame de estadísticas (resultado de la función `stats`) que contiene una columna
        llamada 'Skew' con el coeficiente de asimetría de cada variable.

        Retorna:
            Serie de nombres de variables con coeficientes de asimetría >= 0.5 o <= -0.5, lo que indica
            que tienen una distribución sesgada.
            """
    biased_variables = statistics.query('Skew >= 0.5' or 'Skew <= -0.5')['Variable']
    return biased_variables