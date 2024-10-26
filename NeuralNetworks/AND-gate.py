import numpy as np
import matplotlib.pyplot as plt

#Definimos la función de activación
def activate_func(x):
    return 1 if x > 0 else 0

#Preparamos los inputs y los outputs
X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([[0],[0],[0],[1]])
OPERATION = "AND"

#Inicializamos los pesos y el sesgo del perceptrón
W = np.random.uniform(size=(2))#1 peso por input
b = np.random.uniform(size=(1))

#Training until the perceptron learns to match the output with the label data
#We select the hyperparameter before the training starts
n_epochs = 150 #rondas
lr = 0.01 #learning rate

history_loss = []

for epoch in range(1, n_epochs +1):
    loss = 0.0
    
    for i in range(0, len(Y)):
        #Fordward Propagation
        x0 = X[i][0]
        x1 = X[i][1]
        y_h = W[0] * x0 + W[1] * x1 + b[0]
        y = activate_func(y_h)
        
        #Updating Weights and Biases
        loss += (y - Y[i]) ** 2 / 2 #Loss Function
        
        W[0] -= lr * (y-Y[i].item()) * x0
        W[1] -= lr * (y-Y[i].item()) * x1
        b[0] -= lr * (y-Y[i].item())
    
    #This will show you how the system is working, updating each 10 rounds  
    history_loss.append(loss / len(Y))  
    if epoch % 10 == 0 or epoch == 1:
        print("epoch: {} / {}  Loss = {:.6f}".format(epoch, n_epochs, loss[0]))
        
 #Show loss history       
plt.plot(history_loss, color="b", label="loss")
plt.title("Training Loss History")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.show()

#Test

print("--------------------------")   
print("x0 {} x1 => result".format(OPERATION))
print("===========================") 

for x in X:
    y = activate_func(W[0] * x[0] + W[1] * x[1] + b[0])
    print(" {} {}  {} =>    {}".format(x[0], OPERATION, x[1], int(y)))
    
print("===========================") 

#Show Decision Line

slope = -W[0] / W[1]
bias = -b[0] / W[1]

fig, ax = plt.subplots(1, 1)
title = "Decision Line (" + OPERATION + "-gate)"
plt.title(title)
plt.xlabel("X[0]")
plt.ylabel("X[1]")
plt.grid(False)
plt.ylim([-0.1, 1.4])
plt.xlim([-0.1, 1.4])

for i in range(len(X)):
    x0, x1 = X[i]
    y = Y[i]
    color = "b" if y[0] == 0 else "r"
    plt.plot(x0, x1, "*", markersize=10, color=color)

ax.axline((0, bias), slope=slope)

plt.show()