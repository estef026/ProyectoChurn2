# graficos_multivariados.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Función para gráficos de dispersión
def grafico_dispersion(df, x, y, title, xlabel, ylabel, hue='attrition_flag'):
    # Cambiar la variable objetivo a numérica para la visualización si es necesario
    if df[hue].dtype == 'object':
        df[hue] = df[hue].map({'Existing Customer': 0, 'Attrited Customer': 1})

    # Definir paleta de colores personalizada
    custom_palette = {'Existing Customer': 'red', 'Attrited Customer': 'gray'}

    plt.figure(figsize=(6, 4))
    sns.scatterplot(
        data=df, x=x, y=y,
        sizes=(50, 600),
        hue=hue,
        palette=custom_palette,
        alpha=0.7,
        legend='full'
    )

    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)

    # Crear leyenda personalizada
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', label='Existing Customer', markerfacecolor='red', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Attrited Customer', markerfacecolor='gray', markersize=10)
    ]
    plt.legend(handles=handles, title='Estado del Cliente', fontsize=10, title_fontsize=12,
               loc='upper left', frameon=True, facecolor='white', edgecolor='black', shadow=True)

    plt.tight_layout()
    plt.show()


# Función para gráficos de cajas y bigotes
def grafico_cajas(df, x, y, title, xlabel, ylabel, hue='attrition_flag'):
    custom_palette = {'Existing Customer': 'red', 'Attrited Customer': 'gray'}

    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x=x, y=y, hue=hue, palette=custom_palette, fliersize=4)

    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.legend(title='Estado del Cliente', fontsize=10, title_fontsize=12,
               loc='upper left', frameon=True, facecolor='white', edgecolor='black', shadow=True)

    sns.despine()
    plt.tight_layout()
    plt.show()
