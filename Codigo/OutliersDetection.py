import pandas as pd

class OutlierDetector:
    def __init__(self, df):
        self.df = df

    def count_outliers(self, column):



        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]

        return len(outliers)

    def outliers_summary(self):

        outlier_counts = {}
        for column in self.df.select_dtypes(include=['float', 'int']).columns:
            outlier_counts[column] = self.count_outliers(column)

        # Convert dictionary to DataFrame
        outliers_df = pd.DataFrame(outlier_counts.items(), columns=['Column', 'Outlier_Count'])
        return outliers_df