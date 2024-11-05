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
        # Total number of records in the DataFrame
        total_records = len(self.df)
        # Create a list to store results
        outlier_data = []

        # Loop through each numerical column to calculate outlier counts and percentages
        for column in self.df.select_dtypes(include=['float', 'int']).columns:
            count = self.count_outliers(column)
            percentage = (count / total_records) * 100
            outlier_data.append((column, count, percentage))

        # Convert list to DataFrame
        outliers_df = pd.DataFrame(outlier_data, columns=['Column', 'Outlier_Count', 'Outlier_Percentage'])
        return outliers_df