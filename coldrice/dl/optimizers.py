
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


from .nn import NeuralNetworks


# ----------------------------------------------
# optimizers classes for training (backpropag)
# father : NeuralNetworks
# structure : initialize an optimizer for a model

class sgd(NeuralNetworks):

	__slots__ = ['lr']

	def __init__(self, lr=0.01):
		super(SGD, self).__init__()
		self.lr = lr


class bgd(NeuralNetworks):

	__slots__ = ['lr']

	def __init__(self, lr=0.01):
		super(BGD, self).__init__()
		self.lr = lr


class gd(NeuralNetworks):

	__slots__ = ['lr']

	def __init__(self, lr=0.01):
		super(GD, self).__init__()
		self.lr = lr

class momentum(NeuralNetworks):

	__slots__ = ['lr']

	def __init__(self, lr=0.01):
		super(momentum, self).__init__()
		self.lr = lr
		
class adadelta(NeuralNetworks):

	__slots__ = ['lr']

	def __init__(self, lr=0.01):
		super(adadelta, self).__init__()
		self.lr = lr

class adagrad(NeuralNetworks):

	__slots__ = ['lr']
	
	def __init__(self, lr=0.01):
		super(adagrad, self).__init__()
		self.lr = lr