import numpy as np
#Código que muestra dos funciones: create_wave() y dataset(). La primera
#crea una onda sinusoidal de amplitud 1 y dos ciclos y la segunda genera una secuencia S
#prediciendo el siguiente valor de la onda en función a un conjunto de n valores anteriores.

#Básicamente, busca representar la predicción de valores de la onda sinusoidal
#que es una tarea típica de regresión. Las usaremos en los siguientes códigos


#Función que crea un seno de amplitud 1, dos ciclos y longitud marcada por N
def create_wave(N=100, noise=0.05):
    _x_range = np.linspace(0, 2 * 2 * np.pi, N)
    return np.sin(_x_range) + noise * np.random.randn(len(_x_range))

#Función que genera un dataset válido para entrenar a un modelo de regresión
#Generando para ello pares (x0,y0) donde x0 es una secuencia de n puntos y y0 es el siguiente punto
#que queda fuera de la secuencia.

def dataset(S, n=25, return_sequences=False):
    n_sample = len(S) - n
    
    x = np.zeros((n_sample, n, 1, 1))
    y = np.zeros((n_sample, 1, 1))
    
    for i in range(n_sample):
        x[i, :, 0, 0] = S[i : i + n]
        y[i, 0] = S[i + n]
        
    return x, y