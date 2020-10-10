#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 16:22:15 2020

@author: javierb
"""
from sklearn import datasets # Para cargar Iris
import numpy as np

# Lectura de la base de datos de Iris
iris = datasets.load_iris()

# Obtener las dos últimas características y la clase
x = iris.data[:,-2:]
y = iris.target


# Separamos en arrays las muestras de cada clase para conservar la proporción
samples_seto = x[np.where(y==0)]
samples_versi = x[np.where(y==1)]
samples_virgi = x[np.where(y==2)]

# Los reordenamos aleatoriamente
np.random.shuffle(samples_seto)
np.random.shuffle(samples_versi)
np.random.shuffle(samples_virgi)

# Escogemos el 80% inicial de los arrays de las muestras para el conjunto de training
training_x = np.empty((120,2), np.float32)
training_x[0:40] = samples_seto[:40]
training_x[40:80] = samples_versi[:40]
training_x[80:120] = samples_virgi[:40]

training_y = np.empty((120,1), np.int32)
training_y[0:40] = 0
training_y[40:80] = 1
training_y[80:120] = 2

# Escogemos el 20% final de los arrays de las muestras para el conjunto de test
test_x = np.empty((30,2), np.float32)
test_x[0:10] = samples_seto[40:]
test_x[10:20] = samples_versi[40:]
test_x[20:30] = samples_virgi[40:]

test_y = np.empty((30,1), np.int32)
test_y[0:10] = 0
test_y[10:20] = 1
test_y[20:30] = 2