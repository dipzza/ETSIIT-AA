#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trabajo 3 - Problema de clasificación
Alumno: Fco Javier Bolívar Expósito
"""

import warnings
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.model_selection import GridSearchCV

# Establecemos semilla para obtener resultados reproducibles
np.random.seed(500)


# Lectura de un dataset que viene separado en train/test set
def read_split_data(path, dtype=np.int32):
    train_set = np.loadtxt(path + '.tra', dtype, None, ',')
    test_set = np.loadtxt(path + '.tes', dtype, None, ',')

    x_train = train_set[:, :-1]
    y_train = np.ravel(train_set[:, -1:])
    x_test = test_set[:, :-1]
    y_test = np.ravel(test_set[:, -1:])

    return x_train, y_train, x_test, y_test


# Visualización de información sobre características con baja varianza,
# para decidir si estas no aportan suficiente información
def variance_analisis(features, threshold, n_class):
    var = np.var(x_train, axis=0).round(2)
    mean = np.mean(x_train, axis=0).round(2)

    # Mostramos la varianza y la media de las características con var baja
    print('Features with low variance (var < {})\n'.format(threshold))
    for fe in np.where(var < threshold)[0]:
        print('Feature nº' + str(fe) + ': var = ' + str(var[fe])
              + '  mean = ' + str(mean[fe]))

    # Análisis de la relevancia por clase de las características con baja var
    table = []
    row = {'': 'Total samples of each Class'}

    for i in range(n_class):
        row['Class ' + str(i)] = x_train[np.where(y_train == i)[0]].shape[0]

    table.append(row)

    for fe in np.where((var < threshold) & (mean > 0.0))[0]:
        row = {'': 'Caract. ' + str(fe) + ' not zero'}

        for i in range(10):
            row['Class ' + str(i)] = np.where(
                x_train[np.where(y_train == i)[0], fe] != 0)[0].size

        table.append(row)

    print('\nNot zero values of low variance features with mean > 0')
    df = pd.DataFrame(table, columns=[''] + ['Class ' + str(i) for i in range(n_class)])
    df.set_index('', inplace=True)
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


# Lectura del Data Set
x_train, y_train, x_test, y_test = read_split_data('datos/optdigits')

# Análisis del problema
print('Variance analisis\n' + '-----------------')
variance_analisis(x_train, 0.4, 10)

input("\n--- Pulsar tecla para continuar ---\n")

print('Correlation analisis\n' + '--------------------')
correlation_analisis(x_train, 0.9)

input("\n--- Pulsar tecla para continuar ---\n")

# Preprocesamiento de los datos
# Eliminamos en train y test las carac. detectadas como no predictivas
bad_features = [0, 8, 16, 23, 24, 31, 32, 39, 40, 47, 48, 56, 58]

print('Debido a la poca información que aportan o su correlación con otras ',
      'se eliminan las siguientes características: \n', bad_features)
x_train = np.delete(x_train, bad_features, 1)
x_test = np.delete(x_test, bad_features, 1)

input("\n--- Pulsar tecla para continuar ---\n")

# Selección de modelo y entrenamiento
# Se eligen los mejores hiperparámetros para los modelos 'LogisticRegression' y
# 'Perceptron' usando validación cruzada 5-fold partiendo el train set,
# tras esto se entrena cada modelo usando todo el train set.
parameters_log = [{'penalty': ['l1', 'l2'], 'C': np.logspace(-3, 3, 7)}]
columns_log = ['mean_fit_time', 'param_C', 'param_penalty', 'mean_test_score',
               'std_test_score', 'rank_test_score']
columns_per = ['mean_fit_time', 'param_tol', 'mean_test_score',
               'std_test_score', 'rank_test_score']

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    logReg = GridSearchCV(LogisticRegression(solver='saga'), parameters_log)
    logReg.fit(x_train, y_train)
    print('CV para RL\n', pd.DataFrame(logReg.cv_results_, columns=columns_log).to_string())
    perceptron = GridSearchCV(Perceptron(), {'tol': np.logspace(-6, 3, 10)})
    perceptron.fit(x_train, y_train)
    print('CV para Perceptron\n', pd.DataFrame(perceptron.cv_results_, columns=columns_per).to_string())

# Se muestran los hiperparámetros escogidos y Eval para ambos modelos
# Observamos que la Regresión Logística proporciona mejores resultados
print('\nResultados de selección de hiperparámetros por validación cruzada')
print("LR Best hyperparameters: ", logReg.best_params_)
print("LR CV-Accuracy :", logReg.best_score_)

print("\nP Best hyperparameters: ", perceptron.best_params_)
print("P CV-Accuracy :", perceptron.best_score_)

input("\n--- Pulsar tecla para continuar ---\n")

# Predicción con los modelos entrenados del train y test set
print('Métricas de evaluación para los modelos entrenados para train y test')
ein_reg = logReg.score(x_train, y_train)
ein_per = perceptron.score(x_train, y_train)
print('LR Train-Accuracy: ' + str(ein_reg))
print('P Train-Accuracy: ' + str(ein_per))

etest_reg = logReg.score(x_test, y_test)
etest_per = perceptron.score(x_test, y_test)
print('\nLR Test-Accuracy: ' + str(etest_reg))
print('P Test-Accuracy: ' + str(etest_per))
