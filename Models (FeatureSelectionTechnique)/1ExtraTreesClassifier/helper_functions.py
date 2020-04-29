import math
import numpy as np
import matplotlib.pyplot as plt

""" This file contains helper_functions for the main deep L-layer model,
Each module can be tested independently by passing values through a seperate file """



def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
	""" Creates a list of random minibatches from (X_train, Y_train)
	Returns: mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y) """
	
	np.random.seed(seed) # To make your "random" minibatches the same at each run
	m = X.shape[1]		 # number of training examples
	mini_batches = []
		
	# Shuffle (X, Y)
	permutation = list(np.random.permutation(m))
	shuffled_X = X[:, permutation]
	shuffled_Y = Y[:, permutation].reshape((1,m))

	# Partition (shuffled_X, shuffled_Y)
	num_complete_minibatches = math.floor(m/mini_batch_size)
	for k in range(0, num_complete_minibatches):
		mini_batch_X = shuffled_X[:, mini_batch_size*k : mini_batch_size*(k+1)]
		mini_batch_Y = shuffled_Y[:, mini_batch_size*k : mini_batch_size*(k+1)]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)
	
	# Handle the end case if last mini-batch < mini_batch_size
	if m % mini_batch_size != 0:
		mini_batch_X = shuffled_X[:, mini_batch_size * math.floor(m / mini_batch_size) : m]
		mini_batch_Y = shuffled_Y[:, mini_batch_size * math.floor(m / mini_batch_size) : m]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)
	
	return mini_batches
	

	
def sigmoid(z):
	""" Arguments: z -- A scalar or numpy array of any size """
	sigm = 1 / (1 + np.exp(-z))
	
	return sigm
	
	
	
def relu(z):
	rel = np.maximum(0,x)
	
	return rel
	
	
	
def initialize_parameters_random(layers):
	""" Returns: parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL" """ 
	np.random.seed(3)
	parameters = {}
	L = len(layers)
	
	for l in range(1, L):
		parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * 0.01
		parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

	return parameters
	
	
	
def initialize_parameters_he(layers_dims):
	""" Initialize parameters based on the 'He' function.
	'He' initialization works well for networks with ReLU activations """
	np.random.seed(3)
	parameters = {}
	L = len(layers) - 1
	 
	for l in range(1, L + 1):
		parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2./layers_dims[l-1])
		parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
		
	return parameters
	
	
	
def linear_forward(A, W, b):
	""" cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently """  
	Z = np.dot(W, A) + b
	assert(Z.shape == (W.shape[0], A.shape[1]))
	
	cache = (A, W, b)
	
	return Z, cache
	
	
	
def linear_activation_forward(A_prev, W, b, activation):
    """ Implement the forward propagation for the LINEAR->ACTIVATION layer
    Arguments: A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples) """
    
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache
	
	
	
def L_model_forward(X, parameters):
    """ Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    Combine linear_forward & linear_activation_forward() functions into a model """

    caches = []
    A = X
    L = len(parameters) // 2
    
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu") 
        caches.append(cache)
    
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "sigmoid")
    caches.append(cache)
    assert(AL.shape == (1,X.shape[1]))
            
    return AL, caches