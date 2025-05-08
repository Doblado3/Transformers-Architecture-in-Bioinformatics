# Transformers in Bioinformatics: A Study of Single-Cell Data Analysis

Este repositorio contiene el código y los recursos de mi Trabajo de Fin de Grado, centrado en aplicar modelos Transformers al estudio de datos single-cell.

## Estructura del Repositorio

/AttentionNLP -> Carpeta que contiene el código asociado al capítulo 3: NLP and Attentions; del tutorial seguido. Dentro de la misma, podemos encontrar, entre otros extractos, la implementación de un mecanismo de atenticón en Python, una red GRU para análisis de sentimientos o distintos modelos de prediccion NLP como Trigram y Bigram.

/Experimentacion -> Sección principal del código desarrollado a lo largo del TFG. Contiene los notebooks, destinados a ser ejecutados en Google Colab, con el código necesario para realizar las pruebas con cada modelos, así como un notebook con el código para la visualización de los conjuntos de entrenamiento. Además, dentro hay otra carpeta que contiene una serie de notebooks desarrollados mientras trataba de implementar los modelos, probando distintas estrategias.

/Neural Networks -> Carpeta que contiene el código asociado al capítulo 1: Neural Networks; del tutorial seguido. Dentro de la misma, entre otras, se implementan una Red Neuronal con conexiones residuales para aprender el funcionamiento de una puerta XOR, un modelo SVM o una Red Neuronal con una función softmax para aprender a pasar de binario a decimal.

/RNNs -> Carpeta que contiene el código asociado al capítulo 2: RNNs; del tutorial seguido. Dentro de la misma, entre otras, se implementa una arquitectura LSTM desde cero, una Red Neuronal Recurrente simple desde cero y, además, se prueban las implementaciones con las distintas librerias keras, tensorflow y pytorch.

/Transformers -> Carpeta que contiene el código asociado al capítulo 4: Transformers; del tutorial seguido. Dentro de la misma, se implementa un modelo Transformer en Python para traducir de español a inglés.

## Reproducibilidad

Los notebooks desarrollados para la implementación y prueba de los modelos scGPT, Geneformer y CellPLM han sido creados para ejecutarse en Google Colab, por lo que lo más sencillo para reproducir los resultados es probarlos dentro del mismo. Sin embargo, en caso de poseer una GPU con la suficiente capacidad, con ligeros cambios estos serían perfectamente ejecutables en un entorno local.

## Dependencias

Se incluyen los archivos environment.yml y requeriments.txt para la reproducción de los códigos aportados, además de para consultar versiones en caso de que surjan problemas de compatibilídad. Es posible que las versiones necesarias hayan cambiado en el momento en que estés leyendo esto!!

## Contribuciones

Este trabajo ha sido desarrollado por Pablo Doblado Mendoza bajo la supervisión de el Dr. Juan Antonio Nepomuceno Chamorro. Este, es el Trabajo de Fin de Grado (TFG) desarrollado por el alumno 
para el Grado de Ingeniería de la Salud, mención en Informática Clínica, por la Universidad de Sevilla.

## Contacto

* Pablo Doblado Mendoza
* Pablodoblado3@gmail.com

## Agradecimientos

* Dr. Juan Antonio Nepomuceno Chamorro (Supervisor) 
* The Engineers Guide to Deep Learning”, realizado por Hironobu Suzuki
* Fabian Theis Laboratory
