import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#Let's see how a gradient function works

#Define functions
bias = 4

def L(x):
    return(x-bias) ** 2 / 2

def dL(x):
    return(x - bias)

#Gradient descent algorithm

lr = 0.1 #learning rate

#Initialize x
x = 14

#Training loop
val_list = []
grad_list = []

for epoch in range(300):
    
    #update x
    x = x - lr * dL(x)
    
    val_list.append((np.round(x, 3), np.round(L(x), 5)))
    grad_list.append(np.round(dL(x), 5))
    
print("x_min = {:.3f}\t=> L({:.3f}) = {:.3f})".format(x, x, L(x)))

#Display Animation

fig, ax = plt.subplots()

x = np.linspace(-5,15)

L = L(x)
plt.plot(x,L)
plt.ylabel("L(x)")
plt.xlabel("x")

def draw_step(i, val_list):
    ax.scatter(val_list[i][0], val_list[i][1], s=30, c="red")


ani = FuncAnimation(fig, draw_step, frames=len(val_list), fargs=(val_list,))

plt.show()