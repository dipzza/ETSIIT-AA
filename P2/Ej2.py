#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trabajo 2 - Ejercicio 2
Nombre Estudiante: Fco Javier Bolívar Expósito
"""
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Colores y leyendas para las gráficas
# Los colores estarán definidos por un colormap aleatorio de la siguiente lista
cmaps = ['Pastel2', 'Paired', 'Accent','Dark2', 'Set1',
         'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c']
palette = plt.cm.get_cmap(random.choice(cmaps))

pos_label = 1.0
neg_label = -1.0
frontier = 0.6

pos_label_patch = mpatches.Patch(color=palette(pos_label), label='Label +1')
neg_label_patch = mpatches.Patch(color=palette(neg_label), label='Label -1')
frontier_patch = mpatches.Patch(color=palette(frontier), label='Plot classifier')


# Funciones para simular los datos del Ej1
def simula_unif(N, dim, rango):
    return np.random.uniform(rango[0], rango[1], (N, dim))

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


def recta(x, y, m, n):
    return (y - m*x - n)


def linear(w, x1):
    return (-w[0] - w[1] * x1)/w[2]


def calculate_labels(f, incognitas):
    labels = np.sign(f(*incognitas))
    return np.where(labels == 0, 1, labels)


def introduce_noise(y, proportion=0.1):
    pos_index = np.where(y == 1)
    neg_index = np.where(y == -1)

    index = np.concatenate((
            np.random.choice(pos_index[0], int(len(pos_index[0]) * proportion), replace=False),
            np.random.choice(neg_index[0], int(len(neg_index[0]) * proportion), replace=False)))

    y[index] = -y[index]


# EJERCICIO 2.a): ALGORITMO PERCEPTRON
def signo(x):
    if x >= 0:
        return 1
    return -1


# @brief Perceptron Learning Algorithm. Usado para clasificación
# @param data Características de la muestra de entrenamiento
# @param labels Etiquetas de la muestra de entrenamiento
# @param max_iter Iteraciones máximas
# @param w_ini Punto inicial
def ajusta_PLA(data, labels, max_iter, w_ini):

    iterations = 0
    altered = True

    # Hasta que pase una época sin ningún cambio
    while altered:
        iterations += 1
        altered = False
        
        # Permutamos los datos
        index = np.random.permutation(len(labels))
        data_rng = data[index]
        labels_rng = labels[index]

        # Recorremos todos los datos actualizando los pesos w por cada uno que
        # no esté correctamente clasificado
        for x, y in zip(data_rng, labels_rng):
            if signo(np.dot(w_ini, x)) != y:
                w_ini += y * x
                altered = True

        if iterations == max_iter:
            break

    return w_ini, iterations


# Generamos los datos del ejercicio 1.2.a
x = simula_unif(100, 2, [-50, 50])
m, n = simula_recta([-50, 50])
y = calculate_labels(recta, [x[:, 0], x[:, 1], m, n])

x = np.c_[np.ones(x.shape[0]), x]
print('PLA sin ruido en los datos')

# Ajustamos el modelo con PLA y vemos las iteraciones hasta converger
# 2.a.1.a) Punto inicial vector 0
w, iterations = ajusta_PLA(x, y, 500, np.zeros((x.shape[1])))

print('Con w_inicial = [0, 0, 0]')
print('Iteraciones necesarias para converger: {}'.format(iterations))

plt.title('PLA')
plt.scatter(x[:, 1], x[:, 2], c=y, cmap=palette)
plt.plot([-55, 55], [linear(w, -55), linear(w, 55)], color=palette(frontier))
plt.axis([-55, 55, -55, 55])
plt.legend(handles=[pos_label_patch, neg_label_patch, frontier_patch])
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

# 2.a.1.b) Puntos iniciales valores aleatorios entre 0 y 1
iterations_list = []

for i in range(0, 10):
    __, iterations = ajusta_PLA(x, y, 500, np.random.random_sample(x.shape[1]))
    iterations_list.append(iterations)

print('\nCon w_inicial con valores aleatorios entre 0 y 1')
print('Valor medio de iteraciones necesario para converger: {}'
      .format(np.mean(np.asarray(iterations_list))))

input("\n--- Pulsar tecla para continuar ---\n")

# Repetimos el experimento pero con ruido, con los datos del ejercicio 1.2.b
introduce_noise(y)
print('PLA con ruido en los datos')

# 2.a.2.a) Punto inicial vector 0
w, iterations = ajusta_PLA(x, y, 500, np.zeros((x.shape[1])))
print('Con w_inicial = [0, 0, 0]')
print('Iteraciones: {}'.format(iterations))

plt.title('PLA con ruido')
plt.scatter(x[:, 1], x[:, 2], c=y, cmap=palette)
plt.plot([-55, 55], [linear(w, -55), linear(w, 55)], color=palette(frontier))
plt.axis([-55, 55, -55, 55])
plt.legend(handles=[pos_label_patch, neg_label_patch, frontier_patch])
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

# 2.a.2.b) Puntos iniciales valores aleatorios entre 0 y 1
iterations_list = []
for i in range(0, 10):
    __, iterations = ajusta_PLA(x, y, 500, np.random.random_sample(x.shape[1]))
    iterations_list.append(iterations)

print('\nCon w_inicial con valores aleatorios entre 0 y 1')
print('Valor medio de iteraciones: {}'
      .format(np.mean(np.asarray(iterations))))

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################


# EJERCICIO 2.b): REGRESIÓN LOGÍSTICA CON STOCHASTIC GRADIENT DESCENT
# 2.b.1 Implementación de la regresión logística con SGD

# Calcula la proporción de errores para un problema de clasificación binario
def missclassification(data, labels, w):
    errors = 0.0

    for x, y in zip(data, labels):
        if signo(np.dot(w, x)) != y:
            errors += 1.0

    return errors / len(labels)

# Gradiente para la regresión logística
def logisticGradient(x, y, w):
    gradient = np.zeros((x.shape[1]))

    for sample, label in zip(x, y):
        gradient -= (label * sample) / (1 + np.exp(label * (sample @ w)))

    return (gradient / x.shape[0])

# @brief Regresión logística implementada con SGD
# @param data Característica de la muestra
# @param labels Etiquetas de la muestra
# @param lr Learning rate
# @param m Tamaño de los minibatches
# @param threshold Umbral para la condición de parada
# @param maxiter Iteraciones máximas
def sgdRL(data, labels, lr=0.01, m=1, threshold=0.01, maxiter=1500):
    # Inicializamos w a un vector de 0 y calculamos cuantos minibatches
    w = np.zeros((data.shape[1]))
    n_minibatch = int(len(labels)/m)

    # Barajamos las muestras
    index = np.random.permutation(len(labels))
    rng_data = data[index]
    rng_labels = labels[index]

    # Dividimos las muestras en minibatchs
    minibatch_data = np.array_split(rng_data, n_minibatch)
    minibatch_labels = np.array_split(rng_labels, n_minibatch)

    # Iteramos hasta llegar al límite de iteraciones o pasar del umbral
    # En cada iteración se hace un pase completo a los datos
    for i in range(maxiter):
        index = np.random.permutation(len(minibatch_data))

        # Guardamos el resultado de la época anterior
        last_w = np.copy(w)

        # Actualizamos w para cada minibatch
        for i in index:
            w = w - lr * logisticGradient(minibatch_data[i], minibatch_labels[i], w)

        # Si la norma de la diferencia con la anterior época es menor al umbral
        # paramos el algoritmo
        if (np.linalg.norm(last_w - w) < threshold):
            break

    return w


# Usar la muestra de datos etiquetada para encontrar nuestra solución g y estimar Eout
# usando para ello un número suficientemente grande de nuevas muestras (>999).
# Generamos la muestra de entrenamiento y usamos sgdRL para entrenar el modelo
x = simula_unif(100, 2, [0, 2])
m, n = simula_recta([0, 2])
y = calculate_labels(recta, [x[:, 0], x[:, 1], m, n])

x = np.c_[np.ones(x.shape[0]), x]


w = sgdRL(x, y)

plt.title('Regresión Logística con SGD - Training')
plt.scatter(x[:, 1], x[:, 2], c=y, cmap=palette)
plt.plot([0.0, 2.0], [-w[0] / w[2], (-w[0] - 2.0 * w[1]) / w[2]], color=palette(frontier))
plt.legend(handles=[pos_label_patch, neg_label_patch, frontier_patch])
plt.axis([-0.2, 2.2, -0.2, 2.2])
plt.show()
print('Regresión logística con SGD')
print('Ein: {}'.format(missclassification(x, y, w)))

input("\n--- Pulsar tecla para continuar ---\n")

# Generamos datos para el test y comprobamos el error con el modelo entrenado
x = simula_unif(1000, 2, [0, 2])
y = calculate_labels(recta, [x[:, 0], x[:, 1], m, n])
x = np.c_[np.ones(x.shape[0]), x]

plt.title('Regresión Logística con SGD - Test')
plt.scatter(x[:, 1], x[:, 2], c=y, cmap=palette)
plt.plot([0.0, 2.0], [-w[0] / w[2], (-w[0] - 2.0 * w[1]) / w[2]], color=palette(frontier))
plt.axis([-0.2, 2.2, -0.2, 2.2])
plt.legend(handles=[pos_label_patch, neg_label_patch, frontier_patch])
plt.show()
print('Eout: {}'.format(missclassification(x, y, w)))
