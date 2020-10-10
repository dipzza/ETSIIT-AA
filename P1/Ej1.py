#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 16:23:09 2020

@author: dipzza
"""

import numpy as np
import matplotlib.pyplot as plt
from terminaltables import AsciiTable

# Funciones de evaluación de la función matemática E(u, v) = (ue^v − 2ve^−u )^2
def function_E(u, v):
    return (u * np.exp(v) - 2 * v * np.exp(-u))**2

def deriv_E_u(u, v):
    return 2 * (np.exp(v) * u - 2 * v * np.exp(-u)) * (2 * v * np.exp(-u) + np.exp(v))

def deriv_E_v(u, v):
    return 2 * (np.exp(v) * u - 2 * np.exp(-u)) * (u * np.exp(v) - 2 * np.exp(-u) * v)

def deriv_E(u, v):
    return np.array([deriv_E_u(u, v), deriv_E_v(u, v)])

# Funciones de evaluación de la función matemática f(x, y) = (x − 2)^2 + 2(y + 2)^2 + 2 sin(2πx) sin(2πy)
def function_f(x, y):
    return (x - 2)**2 + 2 * (y + 2)**2 + 2 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)
    
def deriv_f_x(x, y):
    return 4*np.pi*np.sin(2*np.pi*y)*np.cos(2*np.pi*x) + 2*(x - 2)

def deriv_f_y(x, y):
    return 4*(np.pi*np.sin(2*np.pi*x)*np.cos(2*np.pi*y) + y + 2)

def deriv_f(x, y):
    return np.array([deriv_f_x(x, y), deriv_f_y(x, y)])


# @brief Algoritmo de Gradiente Descendiente para minimizar una función
# @param f Función (entendido desde el ámbito de la programación) que devuelve 
#          la evaluación de la función matemática a minimizar
# @param deriv_f Función (entendido desde el ámbito de la programación) que devuelve 
#          la evaluación de la derivada de la función matemática a minimizar
# @param w Incógnitas de la función a minimizar
# @param alpha Tasa de aprendizaje
# @param epsilon Umbral de parada del algoritmo
# @param itermax Cantidad máxima de iteraciones
# @return iterations Cantidad de iteraciones realizadas
# @return w Solución alcanzada
# @return evals Lista del valor de la función en cada iteración
def batchGradientDescent(f, deriv_f, w, alpha, epsilon=None, itermax=500):
    iterations = 0
    
    evals = []
    
    evals.append(f(*w))
    
    # Mientras no se llegue al umbral de parada o al limite de iteraciones
    # actualizamos los pesos
    while (epsilon is None or evals[-1] > epsilon) and iterations < itermax:
        w = w - alpha * deriv_f(*w)
        iterations += 1
        evals.append(f(*w))
    
    return iterations, w, evals

# Salida 2. a)
print('Resultados ejercicio 2:')
_, w, evals = batchGradientDescent(function_E, deriv_E, [1, 1], 0.1)
print('En la función E(u,v)')
print('Mínimo encontrado en ' + str(w) + ' tras ' + str(evals.index(evals[-1])) + ' iteraciones: ' + str(evals[-1]))

input("--- Pulsar tecla para continuar ---\n")

# Salida 2. b)
iterations, w , _ = batchGradientDescent(function_E, deriv_E, [1, 1], 0.1, 10**-14)
print(str(iterations) + ' iteraciones hasta obtener un valor inferior a 10^-14')
print('Se alcanza por 1a vez un valor inferior a 10^-14 en las cordenadas (u,v): ' + str(w))

input("--- Pulsar tecla para continuar ---\n")

# Salida 3. a)
print('Resultados ejercicio 3:')
_, _, evals = batchGradientDescent(function_f, deriv_f, [1, -1], 0.01, itermax=50)

plt.plot(evals, label='alpha = 0.01')
plt.title('Evolución de f(x,y) con iteraciones de GD')
plt.xlabel('Iteraciones')
plt.ylabel('f(x,y)')

_, _, evals = batchGradientDescent(function_f, deriv_f, [1, -1], 0.1, itermax=50)

plt.plot(evals, label='alpha = 0.1')
plt.title('Evolución de f(x,y) con iteraciones de GD')
plt.xlabel('Iteraciones')
plt.ylabel('f(x,y)')
plt.legend()
plt.show()

input("--- Pulsar tecla para continuar ---\n")

# Salida 3. b)

table_data = [[[2.1, -2.1]], [[3, -3]], [[1.5, 1.5]], [[1, -1]]]

for row in table_data:
    _, w, evals = batchGradientDescent(function_f, deriv_f, row[0], 0.01, itermax=50)
    row.append(w)
    row.append(evals[-1])

table_data.insert(0, ['Punto de Inicio', '(x, y)', 'Valor mínimo'])
        
table = AsciiTable(table_data)
print(table.table)