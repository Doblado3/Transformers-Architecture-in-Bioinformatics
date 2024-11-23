import tensorflow as tf
from tensorflow.keras import layers

target_sentences = [14, 12, 2, 15, 0]

embedding_dim = 3

embedding_layer = layers.Embedding(max(target_sentences) + 1,
                                   embedding_dim)

for i in range(len(target_sentences)):
    result = embedding_layer(tf.constant(target_sentences[i])).numpy()
    print("{:>10} => {}".format(target_sentences[i], result))