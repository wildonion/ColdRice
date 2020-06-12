

# coding: utf-8

'''
	Designed By : 
 █     █░ ██▓ ██▓    ▓█████▄  ▒█████   ███▄    █  ██▓ ▒█████   ███▄    █ 
▓█░ █ ░█░▓██▒▓██▒    ▒██▀ ██▌▒██▒  ██▒ ██ ▀█   █ ▓██▒▒██▒  ██▒ ██ ▀█   █ 
▒█░ █ ░█ ▒██▒▒██░    ░██   █▌▒██░  ██▒▓██  ▀█ ██▒▒██▒▒██░  ██▒▓██  ▀█ ██▒
░█░ █ ░█ ░██░▒██░    ░▓█▄   ▌▒██   ██░▓██▒  ▐▌██▒░██░▒██   ██░▓██▒  ▐▌██▒
░░██▒██▓ ░██░░██████▒░▒████▓ ░ ████▓▒░▒██░   ▓██░░██░░ ████▓▒░▒██░   ▓██
	 		>> ICFU by cL34n 3v3RytH!n9 0n 157hAu9us7 <<
'''


import numpy as np
from coldrice.dl.activations import *
from collections import namedtuple

# ----------------------------------------------
# NeuralNetworks class
# father : the begening of everything - it's own
# structure : build the graph
# NOTE : in neural netwroks neurons are the columns in matrix calculations


class NeuralNetworks():
	__slots__ = ['__models', '__biases', '__weights']
	'''
		save a Rice or any models object along with 
		the list of layers object and their configs.
	'''
	def __init__(self):
		self.__models = {} # {model object : {'layers' : self._layers, 'configs' : conf namedtuple} }
		self.__biases = [] # [(model object , layer , biases matrix), ... ]
		self.__weights = [] # [(model object , layer , weights matrix), ... ]
		self.__multilayerperceptron = {} # {model object : [neural_circuit for layer 1, neural_circuit for layer 2, ... , neural_circuit for layer N]}
		self.__wmat = namedtuple('WeightsMat', 'model layer weights')
		self.__bmat = namedtuple('BiasesMat', 'model layer biases')


	def _genWeightsAndBiases(self):
		'''
			generates weights and biases matrices
			for each layer object of our model object.
		'''
		if not self.__weights:
			for m in self.__models:
				for l in range(len(self._models[m]['layers'])):
					if str(self._models[m]['layers'][l].__class__.__name__) == 'Dense':
						if self._models[m]['layers'][l].input_neurons != None:
							self._models[m]['layers'][l].weights = np.random.randn(self.__models[m]['layers'][l].input_neurons, self.__models[m]['layers'][l].hidden_neurons)
						else:
							self._models[m]['layers'][l].weights = np.random.randn(self.__models[m]['layers'][l].hidden_neurons, self.__models[m]['layers'][l+1].hidden_neurons)
						if self._models[m]['layers'][l].use_biase:
							self._models[m]['layers'][l].biases = np.full((1, self.__models[m]['layers'][l].hidden_neurons), 0.1)
							self.__bmat(model=m , layer=self._models[m]['layers'][l] , biases=self._models[m]['layers'][l].biases)
							self.__biases.append(self.__bmat)
						self.__wmat(model=m , layer=self._models[m]['layers'][l] , weights=self._models[m]['layers'][l].weights)
						self.__weights.append(self._wmat)
					elif str(self._models[m]['layers'][l].__class__.__name__) == 'LSTM':
						pass
					elif str(self._models[m]['layers'][l].__class__.__name__) == 'Conv2D':
						pass
					else:
						pass


	def __feedforward(self, x_train, y_train, epochs, batches, model):
		'''
			forward through the network
			If we divide a dataset of 2000 training examples into 500 batches, 
			then 4 iterations will complete 1 epoch. The weights will be updated at the end of each training epoch.
		'''
		iterations = x_train.shape / batches
		for e in range(epochs):
			for i in range(iterations):
				'''
				   x_train.shape >>> ( m, n ) : m rows and n cols in which m is the batches and n is the features
				   so x_train.sahpe[0] is the number of rows which is the number of batches
				'''
				index = np.random.choice(x_train.shape[0], batches, replace=False)
				for l in range(len(model['layers'])):
					if str(model['layers'][l].__class__.__name__) == 'Dense':
						z = np.dot(x_train[index], model['layers'][l].weights) # ((batches , input_neurons) X (input_neurons , hidden_neurons) = (batches , hidden_neurons)) 
						# step 1) : pass z through an activation function based on l.activation ; if l.activation is None use a linear one !
						# step 2) : o = model['layers'][l].activation(z + model['layers'][l].biases)
						model['layers'][l].neural_circuit = o
						self.__multilayerperceptron[model].append(o)
					elif str(model['layers'][l].__class__.__name__) == 'Conv2D':
						pass
					elif str(model['layers'][l].__class__.__name__) == 'LSTM':
						pass
					elif str(model['layers'][l].__class__.__name__) == 'GRU':
						pass
					elif str(model['layers'][l].__class__.__name__) == 'RNN':
						pass
					elif str(model['layers'][l].__class__.__name__) == 'Dropout':
						pass
					elif str(model['layers'][l].__class__.__name__) == 'Flatten':
						pass
					else:
						pass
				pass
			# yield whole info of each iteration and epoch like loss and accuracy value in a tuple
			# call self._backpropagation(output, x_train, y_train, optimizer_name) to update the weights at the end of each epoch
			# TODO : update weights based on model['configs'].optimizer.__class__.__name__ ===> GD , SGD , BGD
		

	def __backpropagation(self, output, x_train, y_train, optimizer_name):
		'''
			train the model by calculating 
			the derivation of each layer 
			using computational graph and jax

			𝚫ᴸ = (Aᴸ - Y) * dZᴸ ..... last layer
			𝚫ⁱ = (𝚫ⁱ⁺¹ . W ᵀ)* dZⁱ ..... other layers
		
			def backward(self, y, rightLayer):
			    if self.isoutputLayer:
			        error = self.A - y
				self.delta = np.atleast_2d(error * self.dZ)
			    else:
				self.delta = np.atleast_2d(
		                np.dot(rightLayer.delta, rightLayer.weight.T)* self.dZ)
			    return self.delta

		        def update(self, learning_rate, left_a):
		            a = np.atleast_2d(left_a)
		            d = np.atleast_2d(self.delta)
			    ad = a.T.dot(d)
			    self.weight -= learning_rate * ad
		'''
		pass
