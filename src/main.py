import mnist_loader
import backP
import numpy as np
from PIL import Image
from numpy import array

training_data , validation_data , test_data = mnist_loader.load_data_wrapper();

backP.main([784,30,10]);
backP.MB_GD(training_data, 6, 10, 3.0,test_data);
#result = backP.evaluate(test_data);

######### finding the answer
im = Image.open("invone.jpg")
arr = array(im,dtype='f')
arr = arr/255
x = np.reshape(arr,(784,1))
result = backP.feedforward(x)
result = np.around(result,decimals=4)
np.set_printoptions(precision=4)

print np.array(result)