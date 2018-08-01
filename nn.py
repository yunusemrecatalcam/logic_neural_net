import numpy as np

LEARNING_RATE = 0.1

def sigmoid(x,deriv=False):
    if (deriv==True):
        return x*(1-x)
    return (1/(1+np.exp(-x)))

X = np.array([#1 #2 #bias
             [0,0,1],
             [0,1,1],
             [1,0,1],
             [1,1,1]])

W = np.array([0.5, 0.5,0.5])
Y = np.array([0,0,0,1])

for i in range(1,200):

    Z = np.dot(X,W)
    l1 = sigmoid(Z)
    #print (l1)
    l1_error = Y - l1
    #print(l1_error)
    l1_delta = l1_error * sigmoid(l1,True)
    print (l1_delta)
    print(".....")
    print(X.T)
    W += np.dot(X.T,l1_delta)
    print(np.dot(X.T,l1_delta))

print W
