#
#Sentiment Analysis using GRU with Attention, esta es una implementació bastante básica
#ya que no es una tarea excesivamente compleja. Clasifica bien 16/20 frases
#

from sklearn.model_selection import train_test_split
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import Tokenizer as tokenizer
import Attention
from tensorflow.keras import optimizers, losses, metrics
import sys

#
#Create the Dataset
#


split = True
file_path = "DataSets/sentiment_labelled_sentences/yelp_labelled.txt"

Lang = tokenizer.Tokenizer(file_path, category=True) #Convertimos las secuencias en números
Lang.read_dataset(padding=True)

vocab_size = Lang.vocab_size()
max_len = Lang.max_len()

x = []
y = []

#Igualamos el tamaño de todas las secuencias mediante padding
_pad = np.full(max_len, "<pad>") 
_pad = np.array(_pad)

#Agregamos una lísta de índices pad para rellenar las secuencias
def padding(seq):
    w_len = len(seq)
    idxs = map(lambda w: Lang.word2idx[w], _pad[w_len:]) #Asignamos índices a las palabras
    for z in idxs:
        seq.append(z)
    return seq

for _, (sentence, label) in enumerate(zip(Lang.sentences, Lang.labels)):
    x.append(padding(sentence))
    y.append([label])
    
if split == True:
    X = np.array(x)
    Y = np.array(y)
    (X_train, X_validate, Y_train, Y_validate) = train_test_split(X, Y, test_size=0.2)
else:
    X_train = np.array(x)
    Y_train = np.array(y)
    
BUFFER_SIZE = int(X_train.shape[0])
BATCH_SIZE = 32
N_BATCH = BUFFER_SIZE // BATCH_SIZE #Tamaño de los grupos a entrenar

dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(BUFFER_SIZE)
#Drop_remainder elimina el útlimo grupo si tiene menos elementos que el resto
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True) 

""" El dataset está balanceado
# Etiquetas están en Y_train (por ejemplo, después de la tokenización y split)
unique, counts = np.unique(Y_train, return_counts=True)

# Mostrar conteo y proporción
for label, count in zip(unique, counts):
    print(f"Etiqueta {label}: {count} instancias ({(count / len(Y_train)) * 100:.2f}%)")

"""

#Create the model
#

input_nodes = 1
hidden_nodes = 128
output_nodes = 1

embedding_dim = 64

#Como queremos hacer las mismas pruebas para dos modelos, nos creamos una clase
class SentimentAnalysis(tf.keras.Model):
    def __init__(self, hidden_units, output_units, vocab_size, embedding_dim):
        super().__init__()
        
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        
        self.gru = tf.keras.layers.GRU(
            hidden_units,
            activation = "tanh",
            recurrent_activation = "sigmoid",
            kernel_initializer = "glorot_normal",
            recurrent_initializer = "orthogonal",
            return_sequences=True,
            return_state=True, #Many-to-Many
        )
        
        self.attention = Attention.BahdanauAttention(hidden_nodes)
        self.dense = tf.keras.layers.Dense(output_units, activation="sigmoid")
        
    def call(self, x):
        x = self.embedding(x)
        output, state = self.gru(x)
        context_vector, _ = self.attention(state, output)
        x = self.dense(context_vector)
        return x
    
model = SentimentAnalysis(hidden_nodes, output_nodes, vocab_size, embedding_dim)

model.build(input_shape=(None, max_len))
model.summary()

#
#Fase de entrenamiento
#

lr = 0.001
beta1 = 0.99
beta2 = 0.9999

loss_function = losses.MeanSquaredError()
optimizer = optimizers.Adam(learning_rate=lr, beta_1=beta1, beta_2=beta2)
train_loss = metrics.Mean()

@tf.function
def train(x, y):
    with tf.GradientTape() as tape:
        pred = model(x)
        loss = loss_function(y, pred)
        grad = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grad, model.trainable_variables))
    return loss


def comp(y, Y):
    if np.abs(y - Y) > 1 / 2:
        return False
    return True

def cal_accuracy():
    if split == True:
        num_correct = 0
        for _, (X, Y) in enumerate(zip(X_validate, Y_validate)):
            x = X[np.newaxis, ...]
            
            y, = model(x)
            y = y[0].numpy()
            if comp(y, Y) == True:
                num_correct += 1
                
        return num_correct / len(X_validate)
    
    else:
        return None
    
if len(sys.argv) == 2:
    n_epochs = int(sys.argv[1])
else:
    n_epochs = 40
    
history_loss = []

for epoch in range(1, n_epochs + 1):
    _loss = 0.0
    for (batch, (X_train, Y_train)) in enumerate(dataset):
        loss = train(X_train, Y_train)
        _loss += loss.numpy()
        
    train_loss(_loss)
    history_loss.append(train_loss.result())
    
    if epoch % 10 == 0 or epoch == 1:
        accuracy = cal_accuracy()
        if accuracy == None:
            print(
                "epoch: {}/{}, loss: {:.3}".format(epoch, n_epochs, train_loss.result())
            )
            
        else:
            
            print(
                "epoch: {}/{}, loss: {:.3} accuracy: {:>5f}".format(
                    epoch, n_epochs, train_loss.result(), accuracy
                )
            )
            
if n_epochs > 0:
    plt.plot(history_loss, label="loss")
    plt.legend(loc="best")
    plt.title("Training Loss History")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.grid(True)
    plt.show()
    
#
#Test
#

num_candidate = 20

def i2w(i):
    return "Positive" if i == 1 else "Negative"

if split == True:
    keys = np.arange(len(X_validate))
else:
    keys = np.arange(Lang.num_texts())
    X_validate = np.array(x)
    Y_validate = np.array(y)
    
n = np.minimum(num_candidate, len(keys))
keys = np.random.permutation(keys)[:n]

_correct = 0

for key in keys:
    x = X_validate[key]
    
    sentence = []
    print("Text:", end="")
    for w in x:
        if Lang.idx2word[w] == "<sos>" or Lang.idx2word[w] == "<eos>":
            continue
        if Lang.idx2word[w] == "<pad>":
            break
        print(Lang.idx2word[w], end=" ")
        sentence.append(Lang.idx2word[w])
    print(".")

    x = x[np.newaxis, ...]

    # Forward Propagation
    y = model(x, training=False)
    y = y[0].numpy()

    print("Correct value   => ", i2w(Y_validate[key]))
    print("Estimated value => ", i2w(np.round(y)))
    if comp(y, Y_validate[key]) == False:
        print("*** Wrong ***")
    else:
        _correct += 1

    print("")

print("\n{} out of {} sentences are correct.".format(_correct, n))

if split == True:
    accuracy = cal_accuracy()
    print("Accuracy: {:>5f}".format(accuracy))
    
