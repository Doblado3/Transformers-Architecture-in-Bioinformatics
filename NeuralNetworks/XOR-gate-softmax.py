import keras
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("..")
from Common import Layers
from Common.Optimizer import update_weights
from Common.ActivationFunctions import sigmoid, deriv_sigmoid, Softmax

"""
import random as rn
import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(0)
rn.seed(0)
"""

#Create Datasets

#Inputs
X = np.array([[0,0],[0,1],[1,0],[1,1]])

#Ground-truth labels
Y = np.array([[0],[1],[1],[0]])

#Convert into One-Hot vector, we need the input data to be categorical
Y = keras.utils.to_categorical(Y,2) #Y:Arrays to be converted to a matrix, 2:Number of categories

"""
Y= [[1. 0.]
    [0. 1.]
    [0. 1.]
    [1. 0.]]
"""

#Convert row vectors into column vectors
X = X.reshape(4, 2, 1)
Y = Y.reshape(4, 2, 1)

#Create Model

input_nodes = 2
hidden_nodes = 3
output_nodes = 2

#We create a dense Layer and a softmax Layer
dense = Layers.Dense(input_nodes, hidden_nodes, sigmoid, deriv_sigmoid)
softmax = Layers.Dense(hidden_nodes, output_nodes, activate_class=Softmax())

# Show parameters
#
params = 0
print("_________________________________________________________________")
print(" Layer (type)                Output Shape              Param #")
print("=================================================================")
param = dense.num_params()
params += param
print(
    " dense (Dense)               (None, {:>2})             {:>8}".format(
        hidden_nodes, param
    )
)
param = softmax.num_params()
params += param
print(
    " softmax (Softmax)           (None, {:>2})             {:>8}".format(
        output_nodes, param
    )
)
print("=================================================================")
print("Total params: {}\n".format(params))

#Training

def train(x, Y, lr=0.001):
    
    #Fordward Propagation
    y = dense.forward_prop(x) #Los outputs los obtiene la propia red, no se necesitan pasar
    y = softmax.forward_prop(y)
    
    #Back Propagation
    loss = -np.sum(Y * np.log(y + 1e-8))
    
    dL = -Y / (y + 1e-8) #Gradiente que le llega a la softmax
    dx = softmax.back_prop(dL) #Gradiente que le llega a la Dense Layer
    _ = dense.back_prop(dx)
    
    #Weights and Bias Update
    update_weights([dense, softmax], lr=lr)
    
    return loss

n_epochs = 15000
lr = 0.1

history_loss = []

#
# Training loop
#
for epoch in range(1, n_epochs + 1):

    loss = 0.0

    for i in range(0, len(Y)):
        loss += train(X[i], Y[i], lr)

    history_loss.append(loss / len(Y))
    if epoch % 1000 == 0 or epoch == 1:
        print("epoch: {} / {}  Loss = {:.6f}".format(epoch, n_epochs, loss))

    if loss <= 1e-7:
        break

#
# Show loss history
#
plt.plot(history_loss, color="b", label="Gradient descent")
plt.title("Training Loss History")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# Test
# ========================================
def predict(x):
    # Forward Propagation
    y = dense.forward_prop(x)
    return softmax.forward_prop(y)


print("-------------------------------------")
print("x0 XOR x1 => prob(0)   prob(1)")
print("=====================================")

for i in range(0, len(Y)):
    x = X[i]
    y = predict(x)
    print(" {} XOR  {} => {:.4f}    {:.4f}".format(x[0][0], x[1][0], y[0][0], y[1][0]))

print("=====================================")


x = np.arange(2)
plt.subplots_adjust(wspace=0.4, hspace=0.8)

for i in range(4):
    plt.subplot(4, 1, i + 1)
    title = str(X[i][0][0]) + "  XOR  " + str(X[i][1][0])
    plt.title(title)
    plt.bar(x, predict(X[i]).reshape(2), tick_label=x, align="center")
    plt.ylim(0, 1)
    plt.ylabel("prob.")

plt.show()
    