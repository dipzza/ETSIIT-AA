#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 12:24:42 2020

@author: dipzza
"""

import numpy as np
import matplotlib.pyplot as plt
from terminaltables import AsciiTable

# Funciones de evaluación de la función matemática f(x, y) = (x − 2)^2 + 2(y + 2)^2 + 2 sin(2πx) sin(2πy)
def function_f(x, y):
    return (x - 2)**2 + 2 * (y + 2)**2 + 2 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)
    
def deriv_f_x(x, y):
    return 4*np.pi*np.sin(2*np.pi*y)*np.cos(2*np.pi*x) + 2*(x - 2)

def second_deriv_f_x(x, y):
    return 2 - 8*np.pi**2*np.sin(2*np.pi*y)*np.sin(2*np.pi*x)

def deriv_f_y(x, y):
    return 4*(np.pi*np.sin(2*np.pi*x)*np.cos(2*np.pi*y) + y + 2)

def second_deriv_f_y(x, y):
    return 4 - 8*np.pi**2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)

def deriv_f(x, y):
    return np.array([deriv_f_x(x, y), deriv_f_y(x, y)])

def second_deriv_f(x, y):
    return np.array([second_deriv_f_x(x,y), second_deriv_f_y(x,y)])

# @brief Algoritmo del método de Newton para encontrar mínimo/máximo de una función
# @param f Función (entendido desde el ámbito de la programación) que devuelve 
#          la evaluación de la función matemática que queremos maximizar/minimizar
# @param deriv_f Función (entendido desde el ámbito de la programación) que devuelve 
#          la evaluación de la derivada de primer orden de la función matemática 
#          que queremos maximizar/minimizar
# @param deriv2_f Función (entendido desde el ámbito de la programación) que devuelve 
#          la evaluación de la derivada de segundo orden de la función matemática 
#          que queremos maximizar/minimizar
# @param w Incógnitas de la función
# @param iterations Cantidad de iteraciones
# @return w Solución alcanzada
# @return evals Lista del valor de la función en cada iteración
def newtonMethod(f, deriv_f, deriv2_f, w, iterations=100):
    evals = []
    
    evals.append(f(*w))
    
    for i in range(iterations):
        w = w - deriv_f(*w) / deriv2_f(*w)
        evals.append(f(*w))
    
    return w, evals


# Descenso con iteraciones
_, evals = newtonMethod(function_f, deriv_f, second_deriv_f, [1, -1], 50)

plt.plot(evals)
plt.title('Cambio del valor de f(x,y) con NM')
plt.xlabel('Iteraciones')
plt.ylabel('f(x,y)')
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# Tabla de resultados con puntos de inicio

table_data = [[[2.1, -2.1]], [[3, -3]], [[1.5, 1.5]], [[1, -1]]]

for row in table_data:
    w, evals = newtonMethod(function_f, deriv_f, second_deriv_f, row[0], 50)
    row.append(w)
    row.append(evals[-1])

table_data.insert(0, ['Punto de Inicio', '(x, y)', 'Valor mínimo'])
        
table = AsciiTable(table_data)
print(table.table)