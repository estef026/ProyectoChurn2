import pandas as pd

class OutlierDetector:
    def __init__(self, df):
        """
                Inicializa la clase con el DataFrame proporcionado.

                Parámetros:
                -----------
                df : pandas.DataFrame
                    DataFrame en el que se desea identificar valores atípicos.
                """
        self.df = df

    def count_outliers(self, column):
        """
                Calcula el número de valores atípicos en una columna específica utilizando
                el metodo del IQR.

                Parámetros:

                column: Nombre de la columna numérica en la que se desea contar los outliers.

                Retorna:Cantidad de valores atípicos en la columna especificada.

            """


        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]

        return len(outliers)

    def outliers_summary(self):
        """
                Genera un resumen de los valores atípicos para todas las columnas numéricas del DataFrame.

                Retorna:
                    DataFrame con tres columnas:
                    - 'Column': nombre de la columna numérica.
                    - 'Outlier_Count': cantidad de valores atípicos en la columna.
                    - 'Outlier_Percentage': porcentaje de outliers sobre el total de registros.
                    """
        total_records = len(self.df)

        outlier_data = []


        for column in self.df.select_dtypes(include=['float', 'int']).columns:
            count = self.count_outliers(column)
            percentage = (count / total_records) * 100
            outlier_data.append((column, count, percentage))


        outliers_df = pd.DataFrame(outlier_data, columns=['Column', 'Outlier_Count', 'Outlier_Percentage'])
        return outliers_df