import math
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

""" This file contains helper_functions for the main deep L-layer model,
Each module can be tested independently by passing values through a seperate file """



def random_mini_batches(X, Y, mini_batch_size, seed):
	""" Creates a list of random minibatches from (X_train, Y_train)
	Returns: mini_batches - list of synchronous (mini_batch_X, mini_batch_Y) """
	
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
	""" Arguments: z - A scalar or numpy array of any size """
	sigm = 1 / (1 + np.exp(-z))
	cache = sigm
	
	return sigm, cache
	
	
	
def relu(z):
	rel = np.maximum(0,z)
	cache = z
	
	return rel, cache
	
	
	
def sigmoid_backward(dA, cache):
	""" Returns: dZ - Gradient of the cost with respect to Z """
	Z = cache
	sigm = 1 / (1 + np.exp(-Z))
	dZ = dA * sigm * (1-sigm)
	
	assert (dZ.shape == Z.shape)
	
	return dZ
	
	
	
def relu_backward(dA, cache):
	""" Implement the backward propagation for a single RELU unit """
	Z = cache
	dZ = np.array(dA, copy=True) # convert dz to a correct object.
	# If z <= 0, set dz to 0 as well. 
	dZ[Z <= 0] = 0
	
	assert (dZ.shape == Z.shape)
	
	return dZ
	
	
	
def initialize_parameters_random(layers):
	""" Returns: parameters - python dictionary containing your parameters "W1", "b1", ..., "WL", "bL" """ 
	np.random.seed(3)
	parameters = {}
	L = len(layers)
	
	for l in range(1, L):
		parameters['W' + str(l)] = np.random.randn(layers[l], layers[l-1]) * 0.001
		parameters['b' + str(l)] = np.zeros((layers[l], 1))
		
	assert(parameters['W' + str(l)].shape == (layers[l], layers[l-1]))
	assert(parameters['b' + str(l)].shape == (layers[l], 1))

	return parameters
	
	
	
def initialize_parameters_he(layers):
	""" Initialize parameters based on the 'He' function.
	'He' initialization works well for networks with ReLU activations """
	np.random.seed(3)
	parameters = {}
	L = len(layers)
	 
	for l in range(1, L):
		parameters['W' + str(l)] = np.random.randn(layers[l], layers[l-1]) * np.sqrt(2./layers[l-1])
		parameters['b' + str(l)] = np.zeros((layers[l], 1))
		
	assert(parameters['W' + str(l)].shape == (layers[l], layers[l-1]))
	assert(parameters['b' + str(l)].shape == (layers[l], 1))
		
	return parameters
	
	
	
def linear_forward(A, W, b):
	""" cache - a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently """	 
	Z = np.dot(W, A) + b
	
	assert(Z.shape == (W.shape[0], A.shape[1]))
	cache = (A, W, b)
	
	return Z, cache
	
	
	
def linear_activation_forward(A_prev, W, b, activation):
	""" Implement the forward propagation for the LINEAR->ACTIVATION layer
	Arguments: A_prev - activations from previous layer (or input data): (size of previous layer, number of examples) """
	
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
	
	
	
def compute_cost(AL, Y):
	""" Compute Cross-entropy loss or cost without regularization, optimize later """
	m = Y.shape[1]
	#AL += 1e-8
	# try:
	cost = - 1/m * (np.dot(Y, np.log(AL).T) + np.dot(1-Y, np.log(1-AL).T))
	#print(cost)
	#print(m)
	# except:
		# return "gibberish"
	# Remove unwanted dimensions
	cost = np.squeeze(cost) # To make sure cost's shape is what we expect
	assert(cost.shape == ())
	# assert(cost!='nan')
	#print(cost)
	return cost
	
	
	
def linear_backward(dZ, cache):
	""" Implement the linear portion of backward propagation for a single layer (layer l) """
	A_prev, W, b = cache
	m = A_prev.shape[1]

	dW = 1/m * (np.dot(dZ, A_prev.T)) 
	db = 1/m * np.sum(dZ, axis = 1, keepdims = True)
	dA_prev = np.dot(W.T, dZ)
	
	assert (dA_prev.shape == A_prev.shape)
	assert (dW.shape == W.shape)
	assert (db.shape == b.shape)
	
	return dA_prev, dW, db
	
	
	
def linear_activation_backward(dA, cache, activation):
	""" Implement the backward propagation for the LINEAR->ACTIVATION layer.
	Arguments: dA - post-activation gradient for current layer l """
	
	linear_cache, activation_cache = cache
	
	if activation == "relu":
		dZ = relu_backward(dA, activation_cache)
		dA_prev, dW, db = linear_backward(dZ, linear_cache)
		
	elif activation == "sigmoid":
		dZ = sigmoid_backward(dA, activation_cache)
		dA_prev, dW, db = linear_backward(dZ, linear_cache)
	
	return dA_prev, dW, db
	
	
	
def L_model_backward(AL, Y, caches):
	""" Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
	
	Arguments: AL -- probability vector, output of the forward propagation (L_model_forward())
		Y - true "label" vector (containing 0 if non-cat, 1 if cat)
	
	Returns: grads -- A dictionary with the gradients """
	
	grads = {}
	L = len(caches) # no. of layers
	m = AL.shape[1]
	Y = Y.reshape(AL.shape) # Y is the same shape as AL
	
	# Initializing the backpropagation
	dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
	
	# Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
	current_cache = caches[L-1]
	grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")
	
	# Loop from l=L-2 to l=0
	for l in reversed(range(L-1)):
		# lth layer: (RELU -> LINEAR) gradients.
		# Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
		current_cache = caches[l]
		dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)], current_cache, activation = "relu")
		grads["dA" + str(l)] = dA_prev_temp
		grads["dW" + str(l + 1)] = dW_temp
		grads["db" + str(l + 1)] = db_temp

	return grads
	
	
	
def update_parameters(parameters, grads, learning_rate):
	""" Update parameters using gradient descent
	
	parameters - python dictionary containing your updated parameters 
				  parameters["W" + str(l)] = ... 
				  parameters["b" + str(l)] = ... """
	
	L = len(parameters) // 2

	# Update rule for each parameter
	for l in range(L):
		parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW"+ str(l+1)]
		parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db"+ str(l+1)]
		
		#print(parameters["W" + str(l+1)])
		
	return parameters



def predict(X, y, parameters):
	""" Use to predict the results of a	 L-layer neural network """
	PERF_FORMAT_STRING = "\
	\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
	Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
	RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\
	\tFalse negatives: {:4d}\tTrue negatives: {:4d}"
	
	m = X.shape[1]
	n = len(parameters) // 2
	pred = np.zeros((1,m))
	
	# Forward propagation
	probab, caches = L_model_forward(X, parameters)

	# convert probability to 'Benign/Malignant' predictions
	for i in range(0, probab.shape[1]):
		if probab[0,i] > 0.5:
			pred[0,i] = 1
		else:
			pred[0,i] = 0
	
	#print ("predictions: " + str(p))
	#print ("true labels: " + str(y))
	true_negatives = 0
	false_negatives = 0
	true_positives = 0
	false_positives = 0
	
	for prediction, truth in zip(np.squeeze(pred), np.squeeze(y)):
		if prediction == 1 and truth == 1:
			true_negatives += 1
		elif prediction == 1 and truth == 0:
			false_negatives += 1
		elif prediction == 0 and truth == 1:
			false_positives += 1
		elif prediction == 0 and truth == 0:
			true_positives += 1

	try:
		total_predictions = true_negatives + false_negatives + false_positives + true_positives
		accuracy = 1.0*(true_positives + true_negatives)/total_predictions
		precision = 1.0*true_positives/(true_positives+false_positives)
		recall = 1.0*true_positives/(true_positives+false_negatives)
		f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
		f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
		print(PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5))
		print(RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives))
		print("")
	except:
		print("Got a divide by zero when trying out:", clf)
		print("Precision or recall may be undefined due to a lack of true positive predicitons.")

	print("False positive rate: " + str((false_positives/(false_positives+true_negatives))*100))
	print("False negative rate: " + str((false_negatives/(false_negatives+true_positives))*100))
	
	## other methods to calculate accuracy
	# print("accuracy: " + str(np.sum((p == y)/m)))
	# print("accuracy: {} %".format(100 - np.mean(np.abs(pred - y)) * 100))
		
	return	
	
	
	
def predict_accuracy(X, y, parameters):
	m = X.shape[1]
	n = len(parameters) // 2
	pred = np.zeros((1,m))
	
	probab, caches = L_model_forward(X, parameters)

	for i in range(0, probab.shape[1]):
		if probab[0,i] > 0.5:
			pred[0,i] = 1
		else:
			pred[0,i] = 0
	
	print("accuracy: {} %".format(100 - np.mean(np.abs(pred - y)) * 100))
	
	
	
def print_mislabeled_images(classes, X, y, p):
	pass



def load_data():
	os.chdir(r"E:\00Malicious-PEfile-Detection\Models (FeatureSelectionTechnique)\1ExtraTreesClassifier")
	dataset = pickle.loads(open('split_data.pkl','rb').read())
	
	X_train, X_dev, X_test, y_train, y_dev, y_test = dataset
	
	return X_train, X_dev, X_test, y_train, y_dev, y_test