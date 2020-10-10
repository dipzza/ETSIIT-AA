#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trabajo 2 - Bonus
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
pseudo_frontier = 0.6
pla_frontier = 0.4

pos_label_patch = mpatches.Patch(color=palette(pos_label), label='Label 8')
neg_label_patch = mpatches.Patch(color=palette(neg_label), label='Label 4')
pseudo_patch = mpatches.Patch(color=palette(pseudo_frontier), label='Plot classifier with LR')
pla_patch = mpatches.Patch(color=palette(pla_frontier), label='Plot classifier with PLA')

# Funcion para leer los datos
def readData(file_x, file_y, digits, labels):
    # Leemos los ficheros	
    datax = np.load(file_x)
    datay = np.load(file_y)
    y = []
    x = []	
    # Solo guardamos los datos cuya clase sea la digits[0] o la digits[1]
    for i in range(0,datay.size):
        if datay[i] == digits[0] or datay[i] == digits[1]:
            if datay[i] == digits[0]:
                y.append(labels[0])
            else:
                y.append(labels[1])
            x.append(np.array([1, datax[i][0], datax[i][1]]))
    			
    x = np.array(x, np.float64)
    y = np.array(y, np.float64)
	
    return x, y

def signo(x):
    if x >= 0:
        return 1
    return -1

# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy', [4,8], [-1,1])
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy', [4,8], [-1,1])


#mostramos los datos
fig, ax = plt.subplots()
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color=palette(neg_label), label='4')
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color=palette(pos_label), label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TRAINING)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), 'o', color=palette(neg_label), label='4')
ax.plot(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), 'o', color=palette(pos_label), label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TEST)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

#LINEAR REGRESSION FOR CLASSIFICATION 

# Calcula la proporción de errores para un problema de clasificación binario
def missclassification(data, labels, w):
    errors = 0.0

    for x, y in zip(data, labels):
        if signo(np.dot(w, x)) != y:
            errors += 1.0

    return errors / len(labels)

# Algoritmo de la Pseudoinversa implementado en la P1
def pseudoinverse(x, y):
    return np.dot(np.linalg.pinv(x),y.reshape(-1, 1)).reshape(3, )

def linear(w, x1):
    return (-w[0] - w[1] * x1)/w[2]

# Clasificamos con la pseudinversa
w_pseudoinverse = pseudoinverse(x, y)
print('Ein con PseudoInversa: {}'.format(missclassification(x, y, w_pseudoinverse)))
print('Eout con PseudoInversa: {}'.format(missclassification(x_test, y_test, w_pseudoinverse)))


# @brief Pocket Perceptron Learning Algorithm. Usado para clasificación
# @param data Características de la muestra de entrenamiento
# @param labels Etiquetas de la muestra de entrenamiento
# @param max_iter Iteraciones máximas
# @param w_ini Punto inicial
def pocketLearningAlgorithm(data, labels, max_iter, w):

    iterations = 0
    altered = True
    best_w = w.copy()
    min_error = missclassification(data, labels, w)

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
            if signo(np.dot(w, x)) != y:
                w += y * x
                altered = True
        
        # Calculamos el error del w calculado en la época actual
        w_error = missclassification(data, labels, w)
        
        # Si el error es menor que el del mejor w hasta el momento lo guardamos
        # como mejor w hasta el momento
        if (w_error < min_error):
            best_w = w.copy()
            min_error = w_error

        if iterations == max_iter:
            break

    return best_w, iterations


w, __ = pocketLearningAlgorithm(x, y, 500, w_pseudoinverse)
e_in = missclassification(x, y, w)
e_out = missclassification(x_test, y_test, w)
print('Ein con Pseudoinversa + Pocket: {}'.format(e_in))
print('Eout con Pseudoinversa + Pocket: {}'.format(e_out))


# Gráfico con la muestra de entrenamiento usada comparando Pseudoinversa y la mejora con PLA-Pocket
plt.title('Digitos Manuscritos (TRAINING) - Pseudoinversa y PLA-Pocket')
plt.xlabel('Intensidad promedio')
plt.ylabel('Simetría')
# Pintamos los datos cambiando el color según su etiqueta
plt.scatter(x[:, 1], x[:, 2], c=y, cmap=palette)
plt.legend(handles=[pos_label_patch, neg_label_patch, pseudo_patch, pla_patch])
# Pintamos la recta y = w0 + w1 * x1 + w2 * x2
plt.plot([0, 0.6], [linear(w, 0), linear(w, 0.6)], color=palette(pla_frontier))
plt.plot([0, 0.6], [linear(w_pseudoinverse, 0), linear(w_pseudoinverse, 0.6)], color=palette(pseudo_frontier))
plt.xlim((0, 1))
plt.ylim((-7.25, -0.5))
plt.show()

# Gráfico con la muestra de test para la solución obtenida con Pseudoinversa + PLA-Pocket
plt.title('Digitos Manuscritos (TEST) - Pseudoinversa y PLA-Pocket')
plt.xlabel('Intensidad promedio')
plt.ylabel('Simetría')
# Pintamos los datos cambiando el color según su etiqueta
plt.scatter(x_test[:, 1], x_test[:, 2], c=y_test, cmap=palette)
plt.legend(handles=[pos_label_patch, neg_label_patch, pseudo_patch, pla_patch])
# Pintamos la recta y = w0 + w1 * x1 + w2 * x2
plt.plot([0, 0.6], [linear(w, 0), linear(w, 0.6)], color=palette(pla_frontier))
plt.plot([0, 0.6], [linear(w_pseudoinverse, 0), linear(w_pseudoinverse, 0.6)], color=palette(pseudo_frontier))
plt.xlim((0, 1))
plt.ylim((-7.25, -0.5))
plt.show()

# input("\n--- Pulsar tecla para continuar ---\n")


#COTA SOBRE EL ERROR

print('Cota sobre Ein')
print('Eout <= ' + str(e_in + np.sqrt(1 / (2 * len(y)) * np.log(2/0.05))))
print('Cota sobre Eout')
print('Eout <= ' + str(e_out + np.sqrt((1 / (2 * len(y))) * np.log(2/0.05))))