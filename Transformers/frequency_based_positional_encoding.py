#
# Pequeña prueba de código para visualizar el mecanismo de positional encoding "vanilla"
#
import numpy as np
import matplotlib.pyplot as plt


def posenc(pos, d_model):
    div_term = np.exp(
        np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    rep = np.zeros(d_model)
    rep[0::2] = np.sin(pos * div_term)
    rep[1::2] = np.cos(pos * div_term)
    return rep

d_model = 2
positions = np.arange(101) #0-49

positional_encodings = np.array([posenc(pos, d_model) for pos in positions])

plt.figure(figsize=(12, 6))
for i in range(d_model):
    plt.plot(positions, positional_encodings[:, i], label=f'Dimension {i+1}')

plt.title('Positional Encoding')
plt.xlabel('Position')
plt.ylabel('Encoding Value')
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
plt.grid(True)
plt.show()