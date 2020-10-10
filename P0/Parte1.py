#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn import datasets # Para cargar Iris
import matplotlib.pyplot as plt # Para visualizar los datos
from matplotlib.lines import Line2D

# Lectura de la base de datos de Iris
iris = datasets.load_iris()

# Obtener las dos últimas características y la clase
x = iris.data[:,-2:]
y = iris.target

# Visualización con un Scatter Plot
# El color de cada punto dependerá de la clase a la que pertenece
# Añadimos titulo, leyenda y etiquetas a los ejes.
plt.scatter(x[:, 0], x[:, 1], c=y, cmap="brg")
plt.title('Iris Scatter Plot')
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])

custom_lines = [Line2D([0], [0], color=(0.0, 0.0, 1.0, 1.0), lw=4),
                Line2D([0], [0], color=(1.0, 0.0, 0.0, 1.0), lw=4),
                Line2D([0], [0], color=(0.0, 1.0, 0.0, 1.0), lw=4)]

plt.legend(custom_lines, iris.target_names[:])
plt.show()
