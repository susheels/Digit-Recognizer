import random
import numpy as np


L = 0
biases = []
theta = []
def main(sizes):
	num_layers = len(sizes)
	global biases
	global theta
	global L
	L = num_layers
	biases = [np.random.randn(y, 1) for y in sizes[1:]]
	theta = [np.random.randn(y, x) 
			for x, y in zip(sizes[:-1], sizes[1:])]
	
def feedforward(a):
		for b, w in zip(biases,theta):
			a = sigmoid_vec(np.dot(w, a)+b)
		return a

def MB_GD(training_data,max_iter,mini_batch_size,alpha,test_data=None):
	global theta
	global biases
	if test_data: n_test = len(test_data)
	n = len(training_data)
	for j in xrange(max_iter):
		random.shuffle(training_data)
		mini_batches = [
			training_data[k:k+mini_batch_size]
			for k in xrange(0, n, mini_batch_size)]
		for mini_batch in mini_batches:
			update(mini_batch,alpha)
		if test_data:
			print "Iter {0}: {1} / {2}".format(
				j, evaluate(test_data), n_test)
		else:
			print "Iter {0} complete".format(j)
	np.save('t',theta)
	np.save('b',biases)

def update(mini_batch,alpha):
	global theta
	global biases
	
	nabla_b = [np.zeros(b.shape) for b in biases]
	nabla_w = [np.zeros(w.shape) for w in theta]
	for x, y in mini_batch:    	
		delta_nabla_b, delta_nabla_w = backprop(x, y)
		nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
		nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
	
	theta = [w-(alpha/len(mini_batch))*nw 
					for w, nw in zip(theta, nabla_w)]
	
	biases = [b-(alpha/len(mini_batch))*nb 
				   for b, nb in zip(biases, nabla_b)]


def backprop(x, y):
	global biases
	global theta
	global L    
	nabla_b = [np.zeros(b.shape) for b in biases]
	nabla_w = [np.zeros(w.shape) for w in theta]
	# feedforward
	activation = x
	activations = [x] # list to store all the activations
	zs = [] # list to store all the z vectors
	for b, w in zip(biases,theta):
		z = np.dot(w, activation)+b
		zs.append(z)
		activation = sigmoid_vec(z)
		activations.append(activation)
	# back prop
	delta = cost_derivative(activations[-1], y) * \
		sigmoid_prime_vec(zs[-1])
	nabla_b[-1] = delta
	nabla_w[-1] = np.dot(delta, activations[-2].transpose())
   
	for l in xrange(2, L):
		z = zs[-l]
		spv = sigmoid_prime_vec(z)
		delta = np.dot(theta[-l+1].transpose(), delta) * spv
		nabla_b[-l] = delta
		nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
	return (nabla_b, nabla_w)

def evaluate(test_data):
	
	test_results = [(np.argmax(feedforward(x)), y) 
					for (x, y) in test_data]
	return sum(int(x == y) for (x, y) in test_results)
	
def cost_derivative(output_activations, y):
	
	return (output_activations-y) 


def sigmoid(z):
	#The sigmoid function
	return 1.0/(1.0+np.exp(-z))

# vectorizing using numpy
sigmoid_vec = np.vectorize(sigmoid)

def sigmoid_prime(z):
	#Derivative of the sigmoid function
	return sigmoid(z)*(1-sigmoid(z))
	
# vectorizing using numpy
sigmoid_prime_vec = np.vectorize(sigmoid_prime)

def output(input_data):
	print feedforward(input_data)
	



	
			

