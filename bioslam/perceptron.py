"""

Simple perceptron neural network
Author: i am trask
URL: https://iamtrask.github.io/2015/07/12/basic-python-network/
Video: https://www.youtube.com/watch?v=kft1AJ9WVDk

"""

import numpy as np

# Sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

# Input dataset
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])

# Output dataset           
y = np.array([[0,0,1,1]]).T

# Seed random numbers to make calculation deterministic
np.random.seed(1)

# Initialize synaptic weights randomly with mean 0
syn0 = 2 * np.random.random_sample((3,1)) - 1

for iter in range(10000):

    # Forward propagation
    l0 = X # Inputs
    l1 = nonlin(np.dot(l0,syn0)) # Outputs

    # Calculate error
    l1_error = y - l1 # Training outputs - outputs

    # Calculate adjustments
    l1_delta = l1_error * nonlin(l1,True) # Multiply error by sigmoid derivatives of outputs

    # Update weights
    syn0 += np.dot(l0.T,l1_delta)

print("Output After Training:")
print(l1)