#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trabajo 3 - Problema de regresión
Alumno: Fco Javier Bolívar Expósito
"""

import warnings
import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Establecemos semilla para obtener resultados reproducibles
np.random.seed(500)


# Separación de un dataset de regresión en conjunto de training y de test
# Mantiene aproximadamente la distribución de valores de Y en los dos conjuntos
# 'partitions': Número de subconjuntos en el que partir la distribución de
# valores de Y.
def balanced_train_test_split(x, y, test_size=0.25, partitions=16, seed=500):
    x_train, x_test, y_train, y_test = [], [], [], []

    crime_range = np.linspace(y.min(), y.max(), num=partitions)

    # Para cada subconjunto de valores de y
    for i in range(len(crime_range) - 1):
        sub_idx = (y >= crime_range[i]) & (y < crime_range[i + 1])

        # Añadimos de forma aleatoria el 75% a training y el 25% a test
        subx_train, subx_test, suby_train, suby_test = train_test_split(
            x[sub_idx], y[sub_idx], test_size=test_size, random_state=seed)

        x_train.append(subx_train)
        x_test.append(subx_test)
        y_train.append(np.ravel(suby_train))
        y_test.append(np.ravel(suby_test))

    x_train = np.vstack(x_train)
    x_test = np.vstack(x_test)
    y_train = np.concatenate(y_train)
    y_test = np.concatenate(y_test)

    # Mezclamos aleatoriamente los train / test sets obtenidos
    x_train, y_train = shuffle(x_train, y_train, random_state=seed)
    x_test, y_test = shuffle(x_test, y_test, random_state=seed)

    return x_train, x_test, y_train, y_test


# Visualización de información sobre características con baja varianza,
# para decidir si estas no aportan suficiente información
def variance_analisis(features, threshold):
    var = np.var(x_train, axis=0)
    mean = np.mean(x_train, axis=0).round(2)

    # Mostramos la varianza y la media de las características con var baja
    print('Features with low variance (var < {})\n'.format(threshold))
    for fe in np.where(var < threshold)[0]:
        print('Feature nº' + str(fe) + ': var = ' + str(var[fe])
              + '  mean = ' + str(mean[fe]))

    # Mostramos nº de valores distintos a 0 de las características con baja var
    table = []

    for fe in np.where((var < threshold) & (mean > 0.00))[0]:
        row = {'Feature nº': str(fe)}

        row['Non zero values'] = np.count_nonzero(data[:, fe])

        table.append(row)

    print('\nValores distintos de 0')
    df = pd.DataFrame(table, columns=['Feature nº', 'Non zero values'])
    df.set_index('Feature nº')
    print(df.to_string())


# Visualización de la correlación fuerte entre características
def correlation_analisis(features, threshold):
    # Cálculo de la corr de pearson
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p_corr = np.nan_to_num(np.triu(np.corrcoef(x_train, rowvar=False), 1),
                               copy=False)
        
    # Mostramos las características con p-corr que supere al 'threshold'
    fe_a, fe_b = np.where((p_corr > threshold) | (p_corr < -threshold))

    print('\nFeatures with Pearson Correlation > {} or < -{}'.format(threshold, threshold))
    for a, b in zip(fe_a, fe_b):
        print('nº{} with nº{}: p-corr = {}'.format(a, b, p_corr[a, b]))

    # Cálculo de la corr de spearman
    sp_corr = np.nan_to_num(np.triu(
        pd.DataFrame(x_train).corr(method='spearman'), 1))
    
    # Mostramos las características con sp-corr que supere al 'threshold'
    fe_a, fe_b = np.where((sp_corr > threshold) | (sp_corr < -threshold))

    print('\nFeatures with Spearman Correlation > {} or < -{}'.format(threshold, threshold))
    for a, b in zip(fe_a, fe_b):
        print('nº{} with nº{}: p-corr = {}'.format(a, b, sp_corr[a, b]))

# Lectura de datos
data = np.loadtxt('datos/communities.data', np.object, None, ',')

# Preprocesamiento: Se eliminan características no predictivas
data = np.delete(data, np.arange(5), axis=1)

# Análisis missing values
print('Missing Values Analisis\n' + '-----------------------')
print('Característica\tValores perdidos')
missing = np.count_nonzero(data == '?', axis=0)
col_miss = np.where(missing > 0)[0]
for i in col_miss:
    print('{}\t\t{}'.format(i, missing[i]))

input("\n--- Pulsar tecla para continuar ---\n")

# Preprocesamiento: Tratamiento de los valores perdidos
# Eliminamos características con demasiados valores perdidos
print('Preprocesamiento Missing Values\n' + '--------------------------')
print('Se eliminan las características con muchos valores perdidos:\n',
      col_miss[1:])

data = np.delete(data, col_miss[1:], axis=1)

# Imputamos el valor perdido que queda con la media de la característica
data[np.where(data[:, 25] == '?')[0], 25] = np.nan
data = data.astype(np.float64, copy=False)
data = SimpleImputer().fit_transform(data)
print('\nSe imputa con el valor medio de la característica el valor perdido' +
      ' de la característica nº', col_miss[1])

input("\n--- Pulsar tecla para continuar ---\n")

# Separación del data set en Train / Test set
x_train, x_test, y_train, y_test = balanced_train_test_split(data[:, :-1],
                                                             data[:, -1],
                                                             partitions=16)

# Análisis de la varianza de cada característica
print('Variance analisis\n' + '-----------------')
variance_analisis(x_train, 0.02)

input("\n--- Pulsar tecla para continuar ---\n")

# Selección de modelo y entrenamiento
# Se eligen los mejores hiperparámetros para los modelos 'SGDRegresor' y
# 'LinearRegression' usando validación cruzada 5-fold partiendo el train set,
# tras esto se entrena cada modelo usando todo el train set.
scoring = 'neg_mean_absolute_error'
parameters = [{'penalty': ['l1', 'l2'], 'alpha': np.logspace(-50, 0, 51), 'eta0': np.logspace(-4, 0, 5)}]
col = ['mean_fit_time', 'mean_test_score', 'std_test_score', 'rank_test_score']

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    sgdReg = GridSearchCV(SGDRegressor(), parameters, scoring)
    sgdReg.fit(x_train, y_train)

linReg = GridSearchCV(LinearRegression(), {'normalize': [False]}, scoring)
linReg.fit(x_train, y_train)

# Se muestran los hiperparámetros escogidos y Eval para ambos modelos
# Observamos que la Regresión Logística proporciona mejores resultados
print('Resultados de selección de hiperparámetros por validación cruzada')
print("SGD Best hyperparameters: ", sgdReg.best_params_)
print("SGD CV-MAE :", -sgdReg.best_score_)

print("\nLin Best hyperparameters: ", linReg.best_params_)
print("Lin CV-MAE :", -linReg.best_score_)

input("\n--- Pulsar tecla para continuar ---\n")

# Predicción con los modelos entrenados del train y test set
print('Métricas de evaluación para los modelos entrenados para train y test')
sgd_pred = sgdReg.predict(x_train)
lin_pred = linReg.predict(x_train)
print('SGD Train-RMSE: ', np.sqrt(mean_squared_error(y_train, sgd_pred)))
print('SGD Train-MAE: ', mean_absolute_error(y_train, sgd_pred))
print('Lin Train-RMSE: ', np.sqrt(mean_squared_error(y_train, lin_pred)))
print('Lin Train-MAE: ', mean_absolute_error(y_train, lin_pred))

sgd_pred = sgdReg.predict(x_test)
lin_pred = linReg.predict(x_test)
print('\nSGD Test-RMSE: ', np.sqrt(mean_squared_error(y_test, sgd_pred)))
print('SGD Test-MAE: ', mean_absolute_error(y_test, sgd_pred))
print('Lin Test-RMSE: ', np.sqrt(mean_squared_error(y_test, lin_pred)))
print('Lin Test-MAE: ', mean_absolute_error(y_test, lin_pred))
