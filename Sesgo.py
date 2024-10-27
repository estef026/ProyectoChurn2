from scipy.stats import skew
from statistics import mean
from statistics import stdev
import numpy as np
import pandas as pd

def stats(data):
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
    biased_variables = statistics.query('Skew >= 0.5' or 'Skew <= -0.5')['Variable']
    return biased_variables