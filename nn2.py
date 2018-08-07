import numpy as np

LEARNING_RATE = 0.1

def sigmoid(x,deriv=False):
    if (deriv==True):
        return sigmoid(x)*(1-sigmoid(x))
    return (1/(1+np.exp(-x)))

X = np.array([#1 #2 #3
             [0,0,0],
             [0,0,1],
             [0,1,0],
             [0,1,1],
             [1,0,0],
             [1,0,1],
             [1,1,0],
             [1,1,1],]).T

b1 = 0.21
b2 = 0.6
b3 =0.54

W1 = np.random.random((10,3))
W2 = np.random.random((10,10))
W3 = np.random.random((1,10))
Y = np.array([[0],[0],[0],[1],[0],[1],[0],[1]]).T


for i in range(1,10):

    Z1 = np.dot(W1,X)+b1  #Forward Propagation
    A1 = sigmoid(Z1)

    Z2 = np.dot(W2,A1)+b2
    A2 = sigmoid(Z2)

    Z3 = np.dot(W3,A2)+b3
    A3 = sigmoid(Z3)

    dZ3 = (A3-Y)          #Backpropagation
    dW3 = np.dot(dZ3,A2.T)/8
    db3 = np.sum(dZ3,axis=1,keepdims=True)/8

    dZ2 = np.dot(dW3.T,dZ3)*sigmoid(Z2,deriv=True)
    dW2 = np.dot(dZ2,A1.T)
    db2 = np.sum(dZ2,axis=1,keepdims=True)/8

    dZ1 = np.dot(dW2.T,dZ2)*sigmoid(Z1,deriv=True)
    dW1 = np.dot(dZ1,X.T)
    db1 = np.sum(dZ1,axis=1,keepdims=True)/8

    W3 = W3 - dW3*LEARNING_RATE
    W2 = W2 - dW2*LEARNING_RATE
    W1 = W1 - dW1*LEARNING_RATE
    b3 = b3 - db3*LEARNING_RATE
    b2 = b2 - db2*LEARNING_RATE
    b1 = b1 - db1*LEARNING_RATE
    if i%100==0:
        print(A3)
        #print("error:",db3*8)

X_in = np.array([[0,0,0]]).T          # This part is for taking inputs for calculating results


while 1:
    X_in[0]=input("input 1:")
    X_in[1]=input("input 2:")
    X_in[2]=input("input 3:")

    Z1 = np.dot(W1,X_in) +b1#Forward Propagation
    A1 = sigmoid(Z1)

    Z2 = np.dot(W2,A1)+b2
    A2 = sigmoid(Z2)

    Z3 = np.dot(W3,A2)+b3
    A3 = sigmoid(Z3)

    print("result:",A3)
