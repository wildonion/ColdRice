



'''
References:
	https://github.com/wildonion/uniXerr/blob/master/references.txt
	https://drive.google.com/drive/folders/1elbMIrg5_NlNMzrKAX_KNjJzgD-MvNQd
	https://realpython.com/python-type-checking/
'''


import numpy as np
from coldrice.dl.models import Rice
from coldrice.dl.models import Sequential
from coldrice.dl.layers import Dense, Conv2D, LSTM
from coldrice.dl.optimizers import gd



# =================
# KERAS SYNTAX
# =================
cr = Sequential()
cr.add(Dense(input_neurons=18, hidden_neurons=34, activation='relu'))
cr.add(Conv2D(32, (1,1), stride=1, padding='valid', activation='relu'))
cr.add(Dense(hidden_neurons=34, activation='sigmoid'))
cr.compile(gd(lr=0.05), loss='sos', metrics=['accuracy'])
cr.summary()
print(cr.get_weights())



# ==============================
# COLDRICE ABSTRACT SYNTAX
# ==============================
@cr.build # build the graph 
def model():
    network.mlp = layer1 = 23 <-> layer2 = 34 <-> output = 2
    network.conv2d = layer1 = mlp(36) <-> layer2 = [32, (1,1), 1, 'valid', 'relu']

cr.compile(model()) # compile to protobuf
cr.execute(model()) # execute the graph


# =======================
# COLDRICE OOP SYNTAX
# =======================
Network = namedtuple('house_price_prediction_model', ['layer1', 'layer2'])
rice = Network( 

				# g=2 uses **kwargs concept; self represents the instance of the class
				# print(o.fn(4, 78))
				# print(o.__dict__)
				# print(dir(o)) # everything in python is an object so we can see the object attributes using dir() method

			  	Rice(input_neurons=12, fn=lambda self, s, g=2:self.a+g*s), 
			  	Rice(input_neurons=32, fn=lambda self, s, g=90:self.a+g*s),
			  	Rice(input_neurons=64, fn=lambda self, s, g=90:self.a+g*s)
			  )

configs = namedtuple('weights_layer1_configs', 'shape distribution')
layer1_w = configs(shape=(24, 128), distribution='uniform')
# >>> layer1_w.shape 
# >>> (24, 128)
# >>> layer1_w.distribution 
# >>> 'uniform'

rice.layer1.name = 'mlp__Dense'
rice.layer1.input_neurons = 24
rice.layer1.activation = 'elu'
rice.layer1.weights = layer1_w


class weights_layer2_configs:
	__slots__ = ['shape', 'distribution']
	def __init__(self, shape, distribution):
		self.shape = shape
		self.distribution = distribution


layer2_w = weights_layer2_configs(shape=(128, 64), distribution='normal')
# >>> layer2_w.shape 
# >>> (128, 64)
# >>> layer2_w.distribution 
# >>> 'normal'

rice.layer2.name = '__Dense'
rice.layer2.input_neurons = 128
rice.layer2.activation = "relu"
rice.layer2.weights = layer2_w
