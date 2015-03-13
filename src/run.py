import numpy as np
from PIL import Image
from numpy import array

## working for 0,1,2,3,4,5,6,7,8,9 --- update on 13/03/15 evening

biases = np.load('b.npy')
theta = np.load('t.npy')

def feedforward(a):
	global theta
	global biases
	for b, w in zip(biases,theta):
		a = sigmoid_vec(np.dot(w, a)+b)
	return a


def sigmoid(z):
	#The sigmoid function
	return 1.0/(1.0+np.exp(-z))

# vectorizing using numpy
sigmoid_vec = np.vectorize(sigmoid)
######### finding the answer
im = Image.open("invseven.jpg")
arr = array(im,dtype='f')
arr = arr/255
x = np.reshape(arr,(784,1))
result = feedforward(x)
result = np.around(result,decimals=4)
np.set_printoptions(precision=4)

print np.array(result)