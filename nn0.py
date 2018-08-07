import numpy as np

LEARNING_RATE = 0.1

def sigmoid(x,deriv=False):
    if (deriv==True):
        return sigmoid(x)*(1-sigmoid(x))
    return (1/(1+np.exp(-x)))

X = np.array([#1 #2
             [0,0],
             [0,1],
             [1,0],
             [1,1]])
b = np.random.random((1))
W = np.random.random((2))
Y = np.array([0,1,1,1])

for i in range(1,2000):

    Z = np.dot(X,W)+b   #Forward Propagation
    A = sigmoid(Z)

    dZ = (A-Y)          #Backpropagation

    dW = np.dot(dZ,X)/4
    db = np.sum(dZ,axis=0,keepdims=True)/4

    print("error:",np.sum(dZ,axis=0,keepdims=True))
    W = W - dW*LEARNING_RATE
    b = b - db*LEARNING_RATE

X_in = np.array([0,0])          # This part is for taking inputs for calculating res

while 1:
    X_in[0]=input("input 1")
    X_in[1]=input("input 2")
    Z = np.dot(X_in,W)+b
    A=sigmoid(Z)
    print("result:",A)
