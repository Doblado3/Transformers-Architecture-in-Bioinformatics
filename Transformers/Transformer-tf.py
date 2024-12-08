#
# Traductor de Español a Inglés usando un Transformer
#

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import logging
import time
import sys

from Common import LanguageTranslationHelper as lth
from sklearn.model_selection import train_test_split



#
# Positional Encoding
#

def positional_encoding(position, d_model):
    #"Lo que va dentro de los senos y cosenos"
    def _get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates
    
    angle_rads = _get_angles(np.arange(position)[:, np.newasis], np.arange(d_model)[np.newaxis, :], d_model)
    
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.sin(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


#
# Producto Vectorial Escalado Attention
#

def scaled_dot_product_attention(q, k, v, mask):
    #Q x K(T) obtiene las similitudes
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    
    #Normaliza los resultados
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    
    #Si hay algún tipo de máscara, como convertir a infinito
    #posiciones adelantadas, la aplica
    if mask is not None:
        scaled_attention_logits += mask * -1e9
    
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    
    return output, attention_weights

#
#Multi-Head attention
#
   

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        #Calcula dk
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        
        #Obtiene las matrices previamente al producto escalar
        self.Wq = tf.keras.layers.Dense(d_model)
        self.Wk = tf.keras.layers.Dense(d_model)
        self.Wv = tf.keras.layers.Dense(d_model)  
        
        self.dense = tf.keras.layers.Dense(d_model)
    
    def call(self, v, k, q, mask):
        
        def _split_heads(x, batch_size):
            #Separamos la dimensión del modelo en diferentes cabezas
            x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
            return tf.transpose(x, perm=[0, 2, 1, 3])
        
        batch_size = tf.shape(q)[0]
        
        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)
        
        q = _split_heads(q, batch_size)
        k = _split_heads(k, batch_size)
        v = _split_heads(v, batch_size)
        
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concact_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concact_attention)
        
        return output, attention_weights 