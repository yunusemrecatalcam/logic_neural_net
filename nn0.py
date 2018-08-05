import numpy as np

LEARNING_RATE = 0.1

def sigmoid(x,deriv=False):
    if (deriv==True):
        return sigmoid(x)*sigmoid(1-x)
    return (1/(1+np.exp(-x)))

X = np.array([#1 #2 #3
             [0,0],
             [0,1],
             [1,0],
             [1,1]])
b = np.random.random((1))
W = np.random.random((2))
Y = np.array([0,1,1,1])
print (W.shape)
print(Y.shape)
for i in range(1,4000):

    Z = np.dot(X,W)+b
    A = sigmoid(Z)

    dZ = (A-Y)
    dW = np.dot(dZ,A.T)/4
    db = np.sum(dZ,axis=0,keepdims=True)/4
    print dW
    print("out:",A)
    print("cor:",Y)
    print("par",(A-Y))
    print("err:",dZ)
    print("dif",dW)
    print("bias",b)
    W = W - dW*LEARNING_RATE
    b = b - db*LEARNING_RATE
    print("\n")
    #print(X)
    #print(W)

print (W)
print b
