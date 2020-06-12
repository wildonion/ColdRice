
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


from coldrice.dl.layers import *
from coldrice.dl.optimizers import *
import numpy as np
# import matplotlib.pyplot as plt
from coldrice.dl.nn import *
from collections import namedtuple


# -------------------------------------
# Seqential is one the coldrice models
# father : NeuralNetworks
# structure : stack of layer objs

class Sequential(NeuralNetworks):

	
	__slots__ = ['__layers']


	def __init__(self):
		'''
			initializing the model like stack layers declaration
		'''
		super(Sequential, self).__init__()
		self.__layers = []


	def add(layer):
		'''
			add a layer object to the layers stack,
			we can access the hidden neurons for the next 
			layer in this layer like : layer.hidden_neurons
			or accessing the name of the class is like : str(layer.__class__.__name__)
		'''
		self.__layers.append(layer)


	def compile(self, optimizer, loss, metrics):
		'''
			compile the model for generating weight/biases matrices 
			and building model configuration options
		'''
		conf = namedtuple('Configs', 'optimizer loss metrics')
		configuration = conf(optimizer=optimizer, loss=loss, metrics=metrics)
		self.__models[self] = {'layers' : self.__layers, 'configs' : configuration}
		self.__multilayerperceptron[self] = []
		self.__genWeightsAndBiases()


	def summary(self):
		'''
			summarize the model
		'''
		print(self.__models)
		print(self.__weights)



	def fit(self, x_train, y_train, epochs, batches, validation_data=None, validation_split=None, verbose=True):
		'''
			training the model on bastches in every epoch
		'''
		if verbose:
			print(f"[+] Start training on {x_train.shape} samples , x_valid ::> {validation_data[0]} - y_valid ::> {validation_data[1]}\n")
		
		for info in self.__feedforward(x_train, y_train, epochs, batches, self.__models[self]):
			# print some forwarding info like loss and accuracy value from each epoch or iteration
			pass

	def evaluate(self, x_test, y_test):
		'''
			evaluate the model
		'''
		pass
		
	def predict(self, x_test, y_test):
		'''
			output predicts for the input sample 'cons 
		'''
		pass

	def plot(self):
		'''
			plot the training and validation loss/acc of a trained model
		'''
		pass

	def save_weights(self, name):
		'''
			save the wights matrices
		'''
		pass

	def save_model(self, name):
		'''
			save the compiled model
		'''
		# TODO 
		# use pickle

		pass

	def set_wights(self, name):
		'''
			load the wights matrices into the model
		'''

		# TODO 

		# load all saved weights from .txt file using numpy
		# and fill the self.__weights list of NeuralNetworks class

		pass

	def load_model(self, name):
		'''
			load the compiled model
		'''
		pass

	def get_weights(self):
		'''
			return the weights matrices for this model
		'''
		for t in self.__weights:
			if t.model == self:
				return t.weights



# -------------------------------------
# Rice is one the coldrice models
# father : python dict class
# structure : Rice object per layer

class Rice(dict):
    def __init__(self, *args, **kwargs): # a=12 uses **kwargs
        # super(Seqential, self).__init__()
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self # assign this object(class) to dict object
        fntype = type(lambda e:e)
        for key in self:
            if type(self[key]) is fntype:
                fn = self[key]
                # print(fn.__code__)
                # def nfn(*args, **kwargs):
                #     return fn(self, *args, **kwargs)
                self[key] = lambda *args, **kwargs: fn(self, *args, **kwargs) # assigning result to the value of self key which is the dict object of our class
