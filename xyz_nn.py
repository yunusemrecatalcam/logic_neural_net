import numpy as np

def sigmoid(x,deriv=False):
    if (deriv==True):      #this is the reverse func of derivative
        return (x*(1-x))
    else:
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

Y = np.array([[0],[1],[0],[1],[0],[1],[0],[1]]).T

W1 = 2*np.random.random((4,3))-1
W2 = 2*np.random.random((1,4))-1
LEARNING_RATE = 1

for j in range(6000):
    Z1 = np.dot(W1,X)
    A1 = sigmoid(Z1)

    Z2 = np.dot(W2,A1)#+b2
    A2 = sigmoid(Z2)

    dZ2 = (Y-A2)* sigmoid(A2,deriv=True)
    dW2 = np.dot(dZ2,A1.T)

    dZ1 = np.dot(dW2.T,dZ2)*sigmoid(A2,deriv=True)
    dW1 = np.dot(dZ1,X.T)

    W2 += dW2*LEARNING_RATE
    W1 += dW1*LEARNING_RATE
    print(np.sum((Y-A2),axis=1))
    #print(A2)

# In[4]:


X_in = np.array([[0,0,0]]).T

while 1:
    X_in[0]=input("input 1:")
    X_in[1]=input("input 2:")
    X_in[2]=input("input 3:")
    print("inputs",X_in)

    Z1 = np.dot(W1,X_in)  #Forward Propagation
    A1 = sigmoid(Z1)
    #print ("W1,X",W1.shape,X.shape)
    #print ("A1",A1.shape)

    Z2 = np.dot(W2,A1)
    A2 = sigmoid(Z2)

    print("result:",A2)
