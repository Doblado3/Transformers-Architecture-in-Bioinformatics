#
#Implementation of Attention Mechanism with Python
#Esto no son arquitecturas Encoder/Decoder
#

import tensorflow as tf

"""
Bahdanau Attention, a.k.a. Additive attention, Multi-Layer perceptron
"""

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
        
    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(query_with_time_axis)))
        
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1) #Suma los distintos valores de cada vector
        
        return context_vector, attention_weights
    
"""
Luong attention, a.k.a. Bilinear Attention, General Attention.
"""

class LuongAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(LuongAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        
    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)
        #Multiplicamos la matriz query por la matriz de pesos global
        score = tf.transpose(tf.matmul(query_with_time_axis, self.W(values), transpose_b=True), perm=[0, 2, 1])

        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, attention_weights