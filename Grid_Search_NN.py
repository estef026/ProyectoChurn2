import numpy as np
from pandas.core.common import random_state
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import AUC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def create_model(input_shape = 43, optimizer = 'adam'):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(input_shape,)))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizer, loss='binary_crossentropy',
                    metrics=['recall', 'precision', AUC(name='auc')])
    model.summary()
    return model

def grid_search_nn(estimator, scoring, cv, X_train, y_train):

    grid_search = GridSearchCV(estimator= estimator,
                               param_grid={'batch_size': [32, 64, 128],
                                            'epochs': [5, 10, 15],
                                            'optimizer': ['adam', 'rmsprop'],
                                            'optimizer__learning_rate': [0.001, 0.01, 0.1]},
                               scoring= scoring,
                               cv=cv)

    grid_search.fit(X_train, y_train, verbose=0)

    return grid_search.best_params_, grid_search.best_score_
