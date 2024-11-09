# exploracion_datos.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Exploración de datos categóricos
def plot_categorical(data, batch_size=16):
    """
    Genera gráficos de frecuencia para columnas categóricas.

    Parámetros:
    data (DataFrame): Datos a analizar
    batch_size (int): Número de columnas a graficar por lote
    """
    categorical_columns = data.select_dtypes(include=['category', 'object']).columns
    cat_columns = len(categorical_columns)

    for start in range(0, cat_columns, batch_size):
        end = min(start + batch_size, cat_columns)
        current_batch = categorical_columns[start:end]
        cat_plots = len(current_batch)

        num_rows = (cat_plots + 3) // 4
        fig, axes = plt.subplots(nrows=num_rows, ncols=4, figsize=(20, num_rows * 5))

        if num_rows == 1 and cat_plots == 1:
            axes = np.array([[axes]])
        elif num_rows == 1:
            axes = np.expand_dims(axes, axis=0)

        for i, col in enumerate(current_batch):
            row = i // 4
            col_idx = i % 4
            ax = axes[row, col_idx]
            sns.countplot(y=col, data=data, hue=col, palette="Set2", ax=ax)
            ax.set_title(f'Frecuencia de {col}')

        if cat_plots < num_rows * 4:
            for i in range(cat_plots, num_rows * 4):
                fig.delaxes(axes.flatten()[i])

        plt.tight_layout()
        plt.show()

# Exploración de datos numéricos
def plot_numerical(data, batch_size=16):
    """
    Genera histogramas y boxplots para columnas numéricas.

    Parámetros:
    data (DataFrame): Datos a analizar
    batch_size (int): Número de columnas a graficar por lote
    """
    numerical_columns = data.select_dtypes(include=[np.number]).columns
    num_columns = len(numerical_columns)

    # Histogramas
    for start in range(0, num_columns, batch_size):
        end = min(start + batch_size, num_columns)
        current_batch = numerical_columns[start:end]
        num_plots = len(current_batch)

        fig, axes = plt.subplots(nrows=(num_plots + 3) // 4, ncols=4, figsize=(20, (num_plots + 3) // 4 * 5))
        fig.tight_layout(pad=5.0)

        for i, col in enumerate(current_batch):
            row = i // 4
            col_idx = i % 4
            ax = axes[row, col_idx] if num_plots > 1 else axes
            sns.histplot(data[col], kde=True, color='blue', ax=ax)
            ax.set_title(f'Histograma de {col}')

        plt.tight_layout()
        plt.show()

    # Boxplots
    for start in range(0, num_columns, batch_size):
        end = min(start + batch_size, num_columns)
        current_batch = numerical_columns[start:end]
        num_plots = len(current_batch)

        fig, axes = plt.subplots(nrows=(num_plots + 3) // 4, ncols=4, figsize=(20, (num_plots + 3) // 4 * 5))
        fig.tight_layout(pad=5.0)

        for i, col in enumerate(current_batch):
            row = i // 4
            col_idx = i % 4
            ax = axes[row, col_idx] if num_plots > 1 else axes
            sns.boxplot(data[col], color='green', ax=ax)
            ax.set_title(f'Boxplot de {col}')

        plt.tight_layout()
        plt.show()

# Diagrama de dispersión entre variables numéricas principales
def plot_scatter(data, batch_size=10):
    """
    Genera diagramas de dispersión entre pares de columnas numéricas.

    Parámetros:
    data (DataFrame): Datos a analizar
    batch_size (int): Número de columnas a graficar por lote
    """
    numerical_columns = data.select_dtypes(include=[np.number]).columns
    num_columns = len(numerical_columns)

    for start in range(0, num_columns, batch_size):
        end = min(start + batch_size, num_columns)
        current_batch = numerical_columns[start:end]
        num_plots = len(current_batch)

        fig, axes = plt.subplots(nrows=(num_plots * (num_plots - 1)) // 2 // batch_size + 1, ncols=batch_size,
                                 figsize=(batch_size * 5, (num_plots * (num_plots - 1)) // 2 // batch_size * 5 + 10))
        fig.tight_layout(pad=5.0)

        plot_count = 0
        for i in range(num_plots - 1):
            for j in range(i + 1, num_plots):
                row = plot_count // batch_size
                col = plot_count % batch_size
                ax = axes[row, col] if num_plots > 1 else axes
                sns.scatterplot(x=data[current_batch[i]], y=data[current_batch[j]], ax=ax)
                ax.set_title(f'{current_batch[i]} vs {current_batch[j]}')
                plot_count += 1

        plt.tight_layout()
        plt.show()
