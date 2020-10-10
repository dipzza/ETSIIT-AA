#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:56:01 2020

@author: dipzza
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

print('EJERCICIO SOBRE REGRESION LINEAL\n')

label5 = 1
label1 = -1
line = 0.6

# Leyendas customizadas para las gráficas
# Los colores estarán definidos por un colormap aleatorio de la siguiente lista
cmaps = ['Pastel2', 'Paired', 'Accent','Dark2', 'Set1', 
         'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c']
colormap = plt.cm.get_cmap(random.choice(cmaps))
label1_patch = mpatches.Patch(color=colormap(float(label1)), label='Label -1')
label5_patch = mpatches.Patch(color=colormap(float(label5)), label='Label 1')
line_patch = mpatches.Patch(color=colormap(line), label='Lineal reg. model')

# Funcion para leer los datos
def readData(file_x, file_y):
	# Leemos los ficheros
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []
	# Solo guardamos los datos cuya clase sea la 1 o la 5
	for i in range(0, datay.size):
		if datay[i] == 5 or datay[i] == 1:
			if datay[i] == 5:
				y.append(label5)
			else:
				y.append(label1)
			x.append(np.array([1, datax[i][0], datax[i][1]]))
            

	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y

# Funcion para calcular el error
def Err(x,y,w):
    return np.square((np.dot(x, w) - y.reshape(-1, 1))).mean()

# Derivada de la función de error
def deriv_Err(x, y, w):
    return 2 * np.dot(x.T, (np.dot(x, w) - y.reshape(-1, 1))) / len(x)

# Gradiente Descendente Estocastico
def sgd(x, y, alpha, m=64, iterations=300):
    # Inicializamos w a un vector de 0 y calculamos cuantos minibatches
    w = np.zeros((x.shape[1],1))
    n_minibatch = int(len(y)/m)
    
    # Barajamos las muestras
    index = np.random.permutation(len(y))
    rng_x = x[index]
    rng_y = y[index]

    # Dividimos las muestras en minibatch
    minibatch_x = np.array_split(rng_x, n_minibatch)
    minibatch_y = np.array_split(rng_y, n_minibatch)
    
    for i in range(iterations):
        index = np.random.permutation(len(minibatch_x))
        
        # Actualizamos w para cada minibatch-
        for i in index:
            w = w - alpha * deriv_Err(minibatch_x[i], minibatch_y[i], w)

    return w

# Pseudoinversa
def pseudoinverse(x, y):
    return np.dot(np.linalg.pinv(x),y.reshape(-1, 1))

def linear(w, x1):
    return (-w[0] - w[1] * x1)/w[2]


# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')

print('Ejercicio 1\n')

w = sgd(x, y, 0.035)
# Generamos un gráfico para la solución obtenida con SGD
plt.title('Regresión Lineal con SGD')
plt.xlabel('x1: Intensidad promedio')
plt.ylabel('x2: Simetría')
# Pintamos los datos cambiando el color según su etiqueta
plt.scatter(x[:, 1], x[:, 2], c=y, cmap=colormap)
plt.legend(handles=[label1_patch, label5_patch, line_patch])
# Pintamos la recta y = w0 + w1 * x1 + w2 * x2
plt.plot([0, 0.6], [linear(w, 0), linear(w, 0.6)], color=colormap(line))
plt.show()

print ('Bondad del resultado para grad. descendente estocástico:\n')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))

input("--- Pulsar tecla para continuar ---\n")

w = pseudoinverse(x, y)
# Generamos un gráfico para la solución obtenida con PseudoInversa
plt.title('Regresión Lineal con alg. Pseudoinversa')
plt.xlabel('x1: Intensidad promedio')
plt.ylabel('x2: Simetría')
# Pintamos los datos cambiando el color según su etiqueta
plt.scatter(x[:, 1], x[:, 2], c=y, cmap=colormap)
plt.legend(handles=[label1_patch, label5_patch, line_patch])
# Pintamos la recta y = w0 + w1 * x1 + w2 * x2
plt.plot([0, 0.6], [linear(w, 0), linear(w, 0.6)], color=colormap(line))
plt.show()

print ('Bondad del resultado para alg. de la PseudoInversa:\n')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))

input("--- Pulsar tecla para continuar ---\n")


print('Ejercicio 2\n')
# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
	return np.random.uniform(-size,size,(N,d))

# Función para asignar etiquetas
def assign_labels(x1, x2):
    return np.sign((x1 - 0.2) ** 2 + x2 ** 2 - 0.6)

# @brief Para una muestra aleatoria en [-1, 1] x [-1, 1] asigna etiquetas segun 
#        "f(x1, x2)" introduciendo un 10% de ruido, y después ajusta un modelo
#        de regresión lineal con características no lineales y estima el error de ajuste Ein
# @return Ein, Error de ajuste
# @return Eout, Error fuera del conjunto de training, calculada con un conjunto de test
def experimento_lineal():
    # Generamos la muestra de entrenamiento y de test
    x, x_test = np.ones((1000, 3)), np.ones((1000, 3))
    x[:,[1, 2]] = simula_unif(1000, 2, 1)
    x_test[:,[1, 2]] = simula_unif(1000, 2, 1)
    # Asignamos las etiquetas con la función f y generamos un 10% de ruido
    y = assign_labels(x[:, 1], x[:, 2])
    y_test = assign_labels(x_test[:, 1], x_test[:, 2])
    index = np.random.choice(np.arange(len(y)), int(len(y)*0.1), replace=False)
    y[index] = -y[index]
    # Estimamos los pesos w con SGD
    w = sgd(x, y, 0.035, 64, 100)
    
    
    return Err(x, y, w), Err(x_test, y_test, w)

# @brief Para una muestra aleatoria en [-1, 1] x [-1, 1] asigna etiquetas segun 
#        "f(x1, x2)" introduciendo un 10% de ruido, y después ajusta un modelo
#        de regresión lineal con características no lineales y estima el error de ajuste Ein
# @return Ein, Error de ajuste
# @return Eout, Error fuera del conjunto de training, calculada con un conjunto de test
def experimento_no_lineal():
    # Generamos la muestra de entrenamiento y de test
    x, x_test = np.ones((1000, 6)), np.ones((1000, 6))
    x[:,[1, 2]] = simula_unif(1000, 2, 1)
    x_test[:,[1, 2]] = simula_unif(1000, 2, 1)
    # Calculamos las características 3-5
    x[:, 3] = x[:, 1] * x[:, 2]
    x[:, 4] = x[:, 1] * x[:, 1]
    x[:, 5] = x[:, 2] * x[:, 2]
    x_test[:, 3] = x_test[:, 1] * x_test[:, 2]
    x_test[:, 4] = x_test[:, 1] * x_test[:, 1]
    x_test[:, 5] = x_test[:, 2] * x_test[:, 2]
    # Asignamos las etiquetas con la función f y generamos un 10% de ruido
    y = assign_labels(x[:, 1], x[:, 2])
    y_test = assign_labels(x_test[:, 1], x_test[:, 2])
    index = np.random.choice(np.arange(len(y)), int(len(y)*0.1), replace=False)
    y[index] = -y[index]
    # Estimamos los pesos w con SGD
    w = sgd(x, y, 0.035, 64, 100)
    
    return Err(x, y, w), Err(x_test, y_test, w)

print('Experimento con características Lineales')
# Ejecución concreta del experimento lineal en la que pintamos cada paso
# a) Generamos la muestra de entrenamiento y la pintamos
x = np.ones((1000, 3))
x[:,[1, 2]] = simula_unif(1000, 2, 1)
plt.title('Puntos aleatorios en [-1, 1] x [-1, 1]')
plt.scatter(x[:, 1], x[:, 2], color=colormap(float(label5)))
plt.show()

input("--- Pulsar tecla para continuar ---\n")

# b) Usamos la función f para asignar etiquetas a la muestra x
print('Asignamos etiquetas con la función sign((x1-0.2)^2 + x2^2 - 0.6)')
y = assign_labels(x[:, 1], x[:, 2])
plt.title('Puntos aleatorios en [-1, 1] x [-1, 1]')
plt.xlabel('x1')
plt.ylabel('x2')
plt.scatter(x[:, 1], x[:, 2], c=y, cmap=colormap)
plt.legend(handles=[label1_patch, label5_patch])
plt.show()

input("--- Pulsar tecla para continuar ---\n")

# Generamos un 10% de los índices de las etiquetas aleatoriamente y 
# los usamos para cambiar el signo e introducir ruido
print('Introducimos un 10% de ruido y estimamos un modelo de reg. lineal')
index = np.random.choice(np.arange(len(y)), int(len(y)*0.1), replace=False)
y[index] = -y[index]

plt.title('Puntos aleatorios en [-1, 1] x [-1, 1] con 10% de ruido')
plt.xlabel('x1')
plt.ylabel('x2')
plt.scatter(x[:, 1], x[:, 2], c=y, cmap=colormap)
plt.legend(handles=[label1_patch, label5_patch, line_patch])
plt.show()

# c) Modelo de regresión
w = sgd(x, y, 0.035, 64, 100)

plt.title('Puntos aleatorios en [-1, 1] x [-1, 1] con 10% de ruido')
plt.xlabel('x1')
plt.ylabel('x2')
plt.scatter(x[:, 1], x[:, 2], c=y, cmap=colormap)
plt.legend(handles=[label1_patch, label5_patch, line_patch])
plt.plot([-1, 1], [linear(w, -1), linear(w, 1)], c=colormap(line))
plt.axis([-1, 1, -1, 1])
plt.show()
print ('Bondad del resultado para SGD con un modelo de características lineales:\n')
print ("Ein: ", Err(x,y,w))

input("--- Pulsar tecla para continuar ---\n")

# d) Repetir a-c 1000 veces y obtener la media de Ein y Eout
print('Repetición del experimento 1000 veces')
e_in, e_out = np.zeros(1000), np.zeros(1000)

for i in range(1000):
    e_in[i], e_out[i] = experimento_lineal()

print('Bondad media de los resultados')
print ("Ein: ", e_in.mean())
print ("Eout: ", e_out.mean())

input("--- Pulsar tecla para continuar ---\n")

# Ejecución concreta del experimento no lineal para pintar el modelo obtenido
print('Experimento con características no Lineales')
# Generación de las muestras
x = np.ones((1000, 6))
x[:,[1, 2]] = simula_unif(1000, 2, 1)
# Calculamos las características 3-5
x[:, 3] = x[:, 1] * x[:, 2]
x[:, 4] = x[:, 1] * x[:, 1]
x[:, 5] = x[:, 2] * x[:, 2]
# Asignación de etiquetas con ruido
y = assign_labels(x[:, 1], x[:, 2])
index = np.random.choice(np.arange(len(y)), int(len(y)*0.1), replace=False)
y[index] = -y[index]
# Estimación de los pesos
w = sgd(x, y, 0.02, 64, 200)

# Pintamos los datos y el modelo obtenido
x1, x2 = np.meshgrid(np.arange(-1, 1, 0.025), np.arange(-1, 1, 0.025))
ellipse = w[0] + x1*w[1] + x2*w[2] + x1 * x2 * w[3] + (x1**2) * w[4] + (x2**2) * w[5]
plt.title('Puntos aleatorios en [-1, 1] x [-1, 1] con 10% de ruido')
plt.xlabel('x1')
plt.ylabel('x2')
plt.scatter(x[:, 1], x[:, 2], c=y, cmap=colormap)
plt.legend(handles=[label1_patch, label5_patch, line_patch])
plt.contour(x1, x2, ellipse, [0], colors=[colormap(line)])
plt.axis([-1, 1, -1, 1])
plt.show()

print('Bondad media de los resultados')
print("Ein: ", Err(x, y, w))

input("--- Pulsar tecla para continuar ---\n")

print('Repetición del experimento 1000 veces')
for i in range(1000):
    e_in[i], e_out[i] = experimento_no_lineal()

print('Bondad del resultado para SGD con un modelo de características no lineales')
print ("Ein: ", e_in.mean())
print ("Eout: ", e_out.mean())