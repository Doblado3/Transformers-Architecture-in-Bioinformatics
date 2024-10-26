from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#XOR gate
#Create Datasets
X = [[0,0],[1,0],[0,1],[1,1]]
Y = [0,1,1,0]

#Create the model
model = KNeighborsClassifier(n_neighbors = 1)#Solo se fija en el valor del adyacente

#Train
model.fit(X,Y)

#Test
result = model.predict(X)

print("-----------------------")
print("x0 XOR x1 => result")
print("========================")

for i in range(len(X)):
    
    _x = X[1]
    print(" {} XOR {} =>   {}".format(_x[0], _x[1], int(result[i])))
    
print("=========================")