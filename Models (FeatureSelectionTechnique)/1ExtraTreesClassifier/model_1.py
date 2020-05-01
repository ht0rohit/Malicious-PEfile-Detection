import matplotlib.pyplot as plt
from helper_functions import *

""" Implement 1 layer & Deep NN without optimization or regularization"""



def model_0(X, Y, layers, learning_rate, mini_batch_size, epsilon, num_epochs, print_cost):
	
	""" single hidden layer neural network model: 

	Arguments: X - input data, of shape (no_of_features, number of examples)
		Y - "legitimate" vector (Benign / Malignant), of shape (1, number of examples)
		layers - python list, containing the size of each layer
		learning_rate - the learning rate, scalar
		mini_batch_size - the size of a mini batch 
		epsilon - use for gradient checking
		num_epochs - number of epochs
		print_cost - True to print the cost every 1000 epochs

	Returns: d -- python dictionary with updated parameters & costs """

	L = len(layers)			# no. of layers in the neural network
	costs = []				# Keep track of the cost
	seed = 10				
	m = X.shape[1]			# number of training examples
	
	
	# Initialize parameters differently & compare
	parameters = initialize_parameters_he(layers)
	#parameters = initialize_parameters_random(layers)
	
	# Optimize cost every iteration
	for i in range(num_epochs):
		# Define the random minibatches. Increment the seed to reshuffle the dataset differently after each epoch
		seed = seed + 1
		
		minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
		cost_total = 0
		
		for minibatch in minibatches:

			# Select a minibatch
			(minibatch_X, minibatch_Y) = minibatch

			# Forward propagation
			AL, caches = L_model_forward(minibatch_X, parameters)

			# Compute cost and add to the cost total
			c = compute_cost(AL, minibatch_Y)
			cost_total += c
			
			# if(c == 'nan'):
				# print("i " + str(i))
				# return

			# Backward propagation
			grads = L_model_backward(AL, minibatch_Y, caches)

			# Update parameters
			parameters = update_parameters(parameters, grads, learning_rate)
		
		cost_avg = cost_total / m
		
		# # Print the cost every 100 epoch
		if print_cost and i % 100 == 0:
			print ("Cost after epoch %i: %f" %(i, cost_avg))
		if print_cost and i % 100 == 0:
			costs.append(cost_avg)
				
	# plot the cost
	# plt.plot(costs)
	# plt.ylabel('cost')
	# plt.xlabel('epochs (per 100)')
	# plt.title("Learning rate = " + str(learning_rate))
	# plt.show()
	
	d = {"costs": costs,
		 "parameters" : parameters,
		 "learning_rate" : learning_rate
		}
	
	return d
	
	
	
if __name__ == '__main__':
	X_train, X_dev, X_test, y_train, y_dev, y_test = load_data()
	# Reshape Y to avoid rank matrix
	y_train = y_train.reshape(1, y_train.shape[0])
	y_dev = y_dev.reshape(1, y_dev.shape[0])
	y_test = y_test.reshape(1, y_test.shape[0])
	
	
	# Analyze model for different learning_rates
	learning_rates = [0.01, 0.001, 0.0001]
	models = {}
	for i in learning_rates:
		print ("learning rate is: " + str(i))
		models[i] = model_0(X_train, y_train, layers = [X_train.shape[0], 10, 10, 1], learning_rate = i, mini_batch_size = 16384,
			epsilon = 1e-8, num_epochs = 1500, print_cost = True)

	print ('\n' + "-------------------------------------------------------" + '\n')

	
	# Plot & compare the costs for different learning_rates 	
	for i in learning_rates:
		plt.plot(np.squeeze(models[i]['costs']), label = str(models[i]['learning_rate']))
	plt.ylabel('cost')
	plt.xlabel('iterations (hundreds)')
	legend = plt.legend(loc='upper center', shadow=True)
	frame = legend.get_frame()
	frame.set_facecolor('0.90')
	plt.show()
	
	for i in learning_rates:
		print("Accuracy for learning rate: " + str(i))
		# Accuracy for Train/dev/test set respectively
		predict_accuracy(X_train, y_train, models[i]['parameters'])
		predict_accuracy(X_dev, y_dev, models[i]['parameters'])
		predict_accuracy(X_test, y_test, models[i]['parameters'])