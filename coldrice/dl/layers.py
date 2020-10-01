

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


'''
														======================================
														FEED FORWARD DIAGRAM IN ONE ITERRATION
														======================================

													  one iteration is to train batch_size of data
											  x_train / batch_size iteration needs to complete one epoch.
			



		input_neurons  = 6 >> features as columns for first layer
		hidden_neurons = 9 >> features as columns for next layer
		batch_size 	   = 8 >> number of training data for each iteration
		output_size    = 1 >> number of predicted output ; one column






	X_train =                                                                               Z =
			   [ NEURON1 NEURON2 NEURON3 NEURON4 NEURON5 NEURON6                          	    [ NEURON1 NEURON2 NEURON3 NEURON4 NEURON5 NEURON6 NEURON7 NEURON8 NEURON9
			d1	[  --- 	   ---     ---     ---     ---     ---  ]					   	   	 d1  [  --- 	---     ---     ---     ---     ---     ---     ---     ---  ]
			d2	[  ---     ---     ---     ---     ---     ---  ]                            d2  [  --- 	---     ---     ---     ---     ---     ---     ---     ---  ]
			d3	[  ---     ---     ---     ---     ---     ---  ]						     d3  [  --- 	---     ---     ---     ---     ---     ---     ---     ---  ]
			d4	[  ---     ---     ---     ---     ---     ---  ]      *      ( _ , _ )   =  d4  [  --- 	---     ---     ---     ---     ---     ---     ---     ---  ]
			d5	[  ---     ---     ---     ---     ---     ---  ]						     d5  [  --- 	---     ---     ---     ---     ---     ---     ---     ---  ]
			d6	[  ---     ---     ---     ---     ---     ---  ]						     d6  [  --- 	---     ---     ---     ---     ---     ---     ---     ---  ]
			d7	[  ---     ---     ---     ---     ---     ---  ]						     d7  [  --- 	---     ---     ---     ---     ---     ---     ---     ---  ]
			d8	[  ---     ---     ---     ---     ---     ---  ]                            d8  [  --- 	---     ---     ---     ---     ---     ---     ---     ---  ]
										   						 ] np.arr(8,6)                                                                                            ] np.arr(8,9)

		
		According to the matrix calculation rules we can simply find the rows and 
		columns for the first weights matrix and initialize it randomly.

	
	( _ , _ ) = first_W =
			        	   [ NEURON1 NEURON2 NEURON3 NEURON4 NEURON5 NEURON6 NEURON7 NEURON8 NEURON9 
						d1	[  --- 	   ---     ---     ---     ---     ---     ---     ---     ---  ]
						d2	[  ---     ---     ---     ---     ---     ---     ---     ---     ---  ]
						d3	[  ---     ---     ---     ---     ---     ---     ---     ---     ---  ]
						d4	[  ---     ---     ---     ---     ---     ---     ---     ---     ---  ]
						d5	[  ---     ---     ---     ---     ---     ---     ---     ---     ---  ]
						d6	[  ---     ---     ---     ---     ---     ---     ---     ---     ---  ]
													   						                         ] np.arr(6,9)
		

		We have to pass the Z matrix through an activation function called tanh or 
		any activation functions out there and multiply it by second weights matrix; 
		again, simply we can guess rows and columns of second weights matrix.


	tanh(Z) =																									   O =	
			   [ NEURON1 NEURON2 NEURON3 NEURON4 NEURON5 NEURON6 NEURON7 NEURON8 NEURON9							   [ NEURON1 
			d1  [  --- 	   ---     ---     ---     ---     ---     ---     ---     ---  ]                            d1	[  ---  ]    
			d2  [  --- 	   ---     ---     ---     ---     ---     ---     ---     ---  ]							 d2	[  ---  ]
			d3  [  --- 	   ---     ---     ---     ---     ---     ---     ---     ---  ]							 d3	[  ---  ]
			d4  [  --- 	   ---     ---     ---     ---     ---     ---     ---     ---  ]        *     ( _ , _ )   = d4	[  ---  ]   
			d5  [  --- 	   ---     ---     ---     ---     ---     ---     ---     ---  ]							 d5	[  ---  ]
			d6  [  --- 	   ---     ---     ---     ---     ---     ---     ---     ---  ]                            d6	[  ---  ]
			d7  [  --- 	   ---     ---     ---     ---     ---     ---     ---     ---  ]							 d7	[  ---  ]
			d8  [  --- 	   ---     ---     ---     ---     ---     ---     ---     ---  ]							 d8	[  ---  ]
										   						 						 ] np.arr(8,9)						     ] np.arr(8,1)


		now we can initilize a random matrix by size 9 X 1 for our second weights matrix ; 
		the result of multiplication of tanh(Z) and second_W will give us a 8 X 1 matrix 
		wich by passing it through tanh activation we have our final predictions 
		for each data vector in batch size.


	( _ , _ ) = second_W =                                tanh(O) = 
						   [ NEURON1                               [ NEURON1 
						d1	[  ---  ]							d1	[  ---  ]
						d2  [  ---  ]							d2	[  ---  ]
						d3	[  ---  ]							d3	[  ---  ]
						d4	[  ---  ]						    d4	[  ---  ]  			    >>>>>>> output prediction
						d5	[  ---  ]                ,    		d5	[  ---  ]
						d6	[  ---  ]							d6	[  ---  ]
						d7	[  ---  ]							d7	[  ---  ]
						d8	[  ---  ]							d8	[  ---  ]
						d9	[  ---  ]										 ] np.arr(8,1)
									 ] np.arr(9,1)



		============
		CONCLUSION :
		============

			W1 = (first_layer_input_neurons  , first_layer_hidden_neurons)
			W2 = (first_layer_hidden_neurons , second_layer_hidden_neurons)
			.
			.
			.
			WN = (N-1th_layer_hidden_neurons   , Nth_layer_hidden_neurons)


			KERAS IMPLEMENTATION :
				model.add(Dense(9, input_dim=6, activation='tanh')) ---- input_neurons / features / columns = 6 , hidden_neurons for next layer = 9 and activation function for hidden layer with 9 neurons is tanh
				model.add(Dense(1, activation='sigmoid')) 			---- output neurons = 1 and the activation function for output with 1 neuron is sigmoid




'''


from .nn import NeuralNetworks
import numpy as np


# -------------------------------
# layer classes
# father : NeuralNetworks
# structure : initialize a layer

class Dense(NeuralNetworks):

	'''
		fully connected simple neural netwrok - multilayer percoptrons
	'''

	__slots__ = ['input_neurons', 'hidden_neurons', 'activation', 'use_bias', 'weights', 'biases', 'neural_circuit']

	def __init__(self, hidden_neurons, input_neurons=None, activation=None, use_bias=True):
		super(Dense, self).__init__()
		self.input_neurons 	= input_neurons # features
		self.hidden 		= hidden_neurons # features for next layer
		self.activation 	= activation
		self.use_bias       = use_bias
		self.weights 		= None
		self.neural_circuit = None
		self.biases 		= None
		self.delta 			= None # backpropagation formula


class LSTM(NeuralNetworks):

	'''
		long short term memory - rnn type
	'''

	__slots__ = ['weights', 'units', 'biases', 'neural_circuit']

	def __init__(self):
		self.weights 		= None
		self.units 			= units
		self.biases 		= None
		self.neural_circuit = None
		self.delta 			= None # backpropagation formula

class GRU(NeuralNetworks):

	'''
		gated recurrent units - rnn type
	'''

	__slots__ = ['weights', 'biases', 'neural_circuit']

	def __init__(self):
		self.weights 		= None
		self.biases 		= None
		self.neural_circuit = None
		self.delta 			= None # backpropagation formula


class RNN(NeuralNetworks):

	'''
		recurrent neural network
	'''

	__slots__ = ['weights', 'biases', 'neural_circuit']

	def __init__(self):
		self.weights 		= None
		self.biases  		= None
		self.neural_circuit = None
		self.delta 			= None # backpropagation formula


class Conv2D(NeuralNetworks):
	'''
		convolutional layers
	'''

	__slots__ = ['filters', 'kernel', 'stride', 'padding', 'activation', 'weights', 'biases', 'neural_circuit']

	def __init__(self, filters, kernel, stride, padding, activation):
		self.filters 		= filters
		self.kernel 		= kernel
		self.stride 		= stride
		self.padding 		= padding
		self.activation 	= activation
		self.weights 		= None
		self.biases 		= None
		self.neural_circuit = None
		self.delta 			= None # backpropagation formula


class Conv2DTranspose(NeuralNetworks):
	'''
		convolutional transpose layers
	'''
	
	__slots__ = ['filters', 'kernel', 'stride', 'padding', 'activation', 'weights', 'biases', 'neural_circuit']

	def __init__(self, filters, kernel, stride, padding, activation):
		self.filters 		= filters
		self.kernel 		= kernel
		self.stride 		= stride
		self.padding 		= padding
		self.activation 	= activation
		self.weights 		= None
		self.biases 		= None
		self.neural_circuit = None
		self.delta 			= None # backpropagation formula

		

class Dropout(NeuralNetworks):

	__slots__ = ['weights', 'biases', 'neural_circuit']

	def __init__(self):
		self.weights 		= None
		self.biases 		= None
		self.neural_circuit = None
		self.delta 			= None # backpropagation formula

class Flatten(NeuralNetworks):

	__slots__ = ['weights', 'biases', 'neural_circuit']

	def __init__(self):
		self.weights 		= None
		self.biases 		= None
		self.neural_circuit = None
		self.delta 			= None # backpropagation formula


class MaxPooling2D(NeuralNetworks):

	__slots__ = ['weights', 'biases', 'neural_circuit']

	def __init__(self):
		self.weights 		= None
		self.biases 		= None
		self.neural_circuit = None
		self.delta 			= None # backpropagation formula
