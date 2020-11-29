

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
from .activations import *
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
						if model['layers'][l].activation.__class__.__name__ == 'relu':
							model['layers'][l].activation = relu()
							o = model['layers'][l].activation(z + model['layers'][l].biases) 
						elif model['layers'][l].activation.__class__.__name__ == 'elu':
							model['layers'][l].activation = elu()
							o = model['layers'][l].activation(z + model['layers'][l].biases)
						elif model['layers'][l].activation.__class__.__name__ == 'sigmoid':
							model['layers'][l].activation = sigmoid()
							o = model['layers'][l].activation(z + model['layers'][l].biases)
						elif model['layers'][l].activation.__class__.__name__ == 'softmax':
							model['layers'][l].activation = softmax()
							o = model['layers'][l].activation(z + model['layers'][l].biases)
						elif model['layers'][l].activation.__class__.__name__ == 'selu':
							model['layers'][l].activation = selu()
							o = model['layers'][l].activation(z + model['layers'][l].biases)
						elif model['layers'][l].activation.__class__.__name__ == 'softplus':
							model['layers'][l].activation = softplus()
							o = model['layers'][l].activation(z + model['layers'][l].biases)
						elif model['layers'][l].activation.__class__.__name__ == 'softsign':
							model['layers'][l].activation = softsign()
							o = model['layers'][l].activation(z + model['layers'][l].biases)
						elif model['layers'][l].activation.__class__.__name__ == 'tanh':
							model['layers'][l].activation = tanh()
							o = model['layers'][l].activation(z + model['layers'][l].biases)
						elif model['layers'][l].activation.__class__.__name__ == 'hardsigmoid':
							model['layers'][l].activation = hardsigmoid()
							o = model['layers'][l].activation(z + model['layers'][l].biases)
						elif model['layers'][l].activation.__class__.__name__ == 'exponential':
							model['layers'][l].activation = exponential()
							o = model['layers'][l].activation(z + model['layers'][l].biases)
						elif model['layers'][l].activation == None:
							model['layers'][l].activation = linear()
							o = model['layers'][l].activation(z + model['layers'][l].biases)
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
				self._backpropagation(model['layers'][l].neural_circuit, x_train, y_train, model['configs'].optimizer.__class__.__name__)
		

	def __backpropagation(self, output, x_train, y_train, optimizer_name):
		'''
			train the model by calculating 
			the derivation of each layer 
			using computational graph and jax
		'''
		pass
