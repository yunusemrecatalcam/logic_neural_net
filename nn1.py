import numpy as np

LEARNING_RATE = 0.1

def sigmoid(x,deriv=False):
    if (deriv==True):
        return sigmoid(x)*(1-sigmoid(x))
    return (1/(1+np.exp(-x)))

X = np.array([#1 #2
             [0,0,0],
             [0,1,1],
             [1,0,1],
             [1,1,1]]).T

b1 = 0.21
b2 = 0.6

W1 = np.random.random((5,3))
W2 = np.random.random((1,5))
Y = np.array([[0],[1],[1],[1]]).T


for i in range(1,2):

    Z1 = np.dot(W1,X)+b1  #Forward Propagation
    A1 = sigmoid(Z1)

    Z2 = np.dot(W2,A1)+b2
    A2 = sigmoid(Z2)
    print (A2)
    print (Y)
    dZ2 = (A2-Y)          #Backpropagation
    print ("dz2",dZ2)
    print("a1t",A1.T)
    dW2 = np.dot(dZ2,A1.T)/4
    db2 = np.sum(dZ2,axis=0,keepdims=True)/4
    print dW2
    dZ1 = np.dot(dW2.T,dZ2.T)
    dW1 = np.dot(dZ1,X)
    db1 = np.sum(dZ1,axis=1,keepdims=True)/4
    print("dz1",dZ1)
    print("db1111",db1)
    W2 = W2 - dW2.T
    W1 = W1 - dW1.T
    b2 = b2 - db2
    b1 = b1 - db1

    print(db2*4)

X_in = np.array([0,0,0])          # This part is for taking inputs for calculating results
print("w1",W1)
print("w2",W2)
while 1:
    X_in[0]=input("input 1")
    X_in[1]=input("input 2")
    X_in[2]=input("input 3")
    Z1 = np.dot(X_in,W1) +b1#Forward Propagation
    A1 = sigmoid(Z1)
    print("a1",A1)
    Z2 = np.dot(A1,W2)+b2
    A2 = sigmoid(Z2)
    print("result:",A2)
