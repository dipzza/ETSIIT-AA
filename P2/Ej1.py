#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trabajo 2 - Ejercicio 1
Nombre Estudiante: Fco Javier Bolívar Expósito
"""
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Colores y leyendas para las gráficas
# Los colores estarán definidos por un colormap aleatorio de la siguiente lista
cmaps = ['Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
         'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
         'GnBu', 'PuBu', 'RdBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
palette_cuad = plt.cm.get_cmap(random.choice(cmaps))

cmaps = ['Pastel2', 'Paired', 'Accent','Dark2', 'Set1',
         'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c']
palette = plt.cm.get_cmap(random.choice(cmaps))

pos_label = 1.0
neg_label = -1.0
frontier = 0.6

pos_label_patch = mpatches.Patch(color=palette(pos_label), label='Label +1')
neg_label_patch = mpatches.Patch(color=palette(neg_label), label='Label -1')
frontier_patch = mpatches.Patch(color=palette(frontier), label='Plot classifier')


def simula_unif(N, dim, rango):
    return np.random.uniform(rango[0], rango[1], (N, dim))


def simula_gaus(N, dim, sigma):
    media = 0
    out = np.zeros((N, dim), np.float64)
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para
        # la primera columna (eje X) se usará una N(0,sqrt(sigma[0]))
        # y para la segunda (eje Y) N(0,sqrt(sigma[1]))
        out[i, :] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)

    return out


def simula_recta(intervalo):
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    x1 = points[0, 0]
    x2 = points[1, 0]
    y1 = points[0, 1]
    y2 = points[1, 1]
    # y = a*x + b
    a = (y2-y1)/(x2-x1)  # Calculo de la pendiente.
    b = y1 - a*x1        # Calculo del termino independiente.

    return a, b


# EJERCICIO 1.1: Dibujar una gráfica con la nube de puntos de salida correspondiente

# Generamos los datos con las funciones proporcionadas y los pintamos
x = simula_unif(50, 2, [-50, 50])
plt.title('Nube de puntos aleatoria Uniforme')
plt.scatter(x[:, 0], x[:, 1], color=palette(pos_label))
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

x = simula_gaus(50, 2, np.array([5, 7]))
plt.title('Nube de puntos aleatoria Gaussiana')
plt.scatter(x[:, 0], x[:, 1], color=palette(pos_label))
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 1.2: Dibujar una gráfica con la nube de puntos de salida correspondiente


def recta(x, y, m, n):
    return (y - m*x - n)

# @brief Calcula las etiquetas asociadas a una función matemática y 
#        sus parámetros para una serie de puntos
# @param f Función matemática
# @param incognitas Lista de parámetros que necesita f
def calculate_labels(f, incognitas):
    # Se ha decidido hacer con operaciones de arrays por comodidad
    # Los 0 que pueden dar problemas se cambian a 1s con np.where
    labels = np.sign(f(*incognitas))
    return np.where(labels == 0, 1, labels)


# Simulamos una muestra asignando las etiquetas según una recta simulada
x = simula_unif(100, 2, [-50, 50])
m, n = simula_recta([-50, 50])
y = calculate_labels(recta, [x[:, 0], x[:, 1], m, n])

plt.title('Distribución Uniforme, puntos clasificados según recta')
plt.scatter(x[:, 0], x[:, 1], c=y, cmap=palette)
plt.plot([-50, 50], [m*-50 + n, m*50 + n], color=palette(frontier))
plt.axis([-50, 50, -50, 50])
plt.legend(handles=[pos_label_patch, neg_label_patch, frontier_patch])
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# 1.2.b. Dibujar una gráfica donde los puntos muestren el resultado de su etiqueta, junto con la recta usada para ello


# @brief Introduce ruido con la misma proporción para cada etiqueta
# @param labels Etiquetas de una muestra
# @param proportion Proporción en la que se introduce ruido, por defecto 10% de cada etiqueta
def introduce_noise(labels, proportion=0.1):
    pos_index = np.where(labels == 1)
    neg_index = np.where(labels == -1)

    # Obtenemos el porcentaje indicado por proportion de indices aleatorios
    # para cada etiqueta
    index = np.concatenate((
            np.random.choice(pos_index[0], int(len(pos_index[0]) * proportion), replace=False),
            np.random.choice(neg_index[0], int(len(neg_index[0]) * proportion), replace=False)))

    # Introducimos el ruido en los índices aleatorios generados
    labels[index] = -labels[index]


introduce_noise(y)

plt.title('Distribución Uniforme con ruido, puntos clasificados según recta')
plt.scatter(x[:, 0], x[:, 1], c=y, cmap=palette)
plt.plot([-50, 50], [m*-50 + n, m*50 + n], color=palette(frontier))
plt.axis([-50, 50, -50, 50])
plt.legend(handles=[pos_label_patch, neg_label_patch, frontier_patch])
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 1.3: Supongamos ahora que las siguientes funciones definen la frontera de clasificación de los puntos de la muestra en lugar de una recta


def plot_datos_cuad(X, y, fz, palette, title='Point cloud plot', xaxis='x axis', yaxis='y axis'):
    # Preparar datos
    min_xy = X.min(axis=0)
    max_xy = X.max(axis=0)
    border_xy = (max_xy-min_xy)*0.01

    # Generar grid de predicciones
    xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0], 
                      min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
    grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
    pred_y = fz(grid)
    # pred_y[(pred_y>-1) & (pred_y<1)]
    pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)
    
    # Plot
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, pred_y, 50, cmap=palette,vmin=-1, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label('$f(x, y)$')
    ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=2, 
                cmap=palette, edgecolor=palette(0.5))
    
    XX, YY = np.meshgrid(np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]),np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]))
    positions = np.vstack([XX.ravel(), YY.ravel()])
    ax.contour(XX,YY,fz(positions.T).reshape(X.shape[0],X.shape[0]),[0], colors='black')
    
    ax.set(
       xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]), 
       ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]),
       xlabel=xaxis, ylabel=yaxis)
    plt.title(title)
    plt.show()


def f1(x):
    return (x[:, 0] - 10) ** 2 + (x[:, 1] - 20) ** 2 - 400

def f2(x):
    return 0.5 * (x[:, 0] + 10) ** 2 + (x[:, 1] - 20) ** 2 - 400

def f3(x):
    return 0.5 * (x[:, 0] - 10) ** 2 - (x[:, 1] + 20) ** 2 - 400

def f4(x):
    return x[:, 1] - 20 * x[:, 0] ** 2 - 5 * x[:, 0] + 3


functions = [f1, f2, f3, f4]

# Pintamos las fronteras de clasificación
for function in functions:
    y = calculate_labels(function, [x])
    introduce_noise(y)
    plot_datos_cuad(x, y, function, palette_cuad)
    input("\n--- Pulsar tecla para continuar ---\n")
