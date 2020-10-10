#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 15:55:30 2020

@author: javierb
"""

import numpy as np
import matplotlib.pyplot as plt # Para visualizar los datos

# Generar 100 valores equiespaciados entre 0 y 2pi
valores = np.linspace(0, 2 * np.pi, 100)

# Calculamos para los valores calculados el seno, el coseno y el seno+coseno
sin = np.sin(valores)
cos = np.cos(valores)
sinpluscos = sin + cos

# Visualizamos con líneas
plt.plot(valores, sin, 'k--', label='sin(x)')
plt.plot(valores, cos, 'b--', label='cos(x)')
plt.plot(valores, sinpluscos, 'r--', label='sin(x) + cos(x)')
plt.title('3 funciones f(x) = y')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()