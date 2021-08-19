import matplotlib.pyplot as plt
import numpy as np

from bin.neural_n import NeuralNetwork

N_train = 3   # Number of training examples from file
train_data = np.zeros([N_train, 785])
with open('data_set/mnist_train.csv', 'r') as data:
    for x in range(N_train):
        train_data[x] = np.fromstring(next(data), sep=',')
train_data = train_data.astype(np.int)

N_test = 3   # Number of testing examples from file
test_data = np.zeros([N_test, 785])
with open('data_set/mnist_test.csv', 'r') as data:
    for x in range(N_test):
        test_data[x] = np.fromstring(next(data), sep=',')
test_data = test_data.astype(np.int)

fac = 1 / 255
train_imgs = (train_data[:, 1:]) * fac
test_imgs = (test_data[:, 1:]) * fac

train_labels = train_data[:, :1].transpose()[0,:]
test_labels = (test_data[:, :1]).transpose()[0,:]

'''img = test_imgs[0].reshape((28,28))
plt.imshow(img, cmap='Greys')
plt.show()'''

'''from mnist import MNIST

mndata = MNIST('dataset')
images_train, labels_train = mndata.load_training()
images_test, labels_test = mndata.load_testing()'''

'''from sklearn.datasets import fetch_openml
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

x, y = fetch_openml('mnist_784', version=1, return_X_y=True)
x = (x/255).astype('float32')
y = to_categorical(y)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, random_state=42)
net = NeuralNetwork(784,300,10, learning_rate=0.01, epochs=1)
net.train(list(zip(x_train, y_train))[500], visualize_cost=False)
print(net.evaluate(list(zip(x_train, y_train))[100]))
print(net.accuracy(list(zip(x_train, y_train))[100]))'''


'''net = NeuralNetwork(5,4,3, learning_rate=0.1, epochs=1)
net.train([([1,2,3,4,5], [0,1,0])] * 400, visualize_cost=True)
print(net.evaluate([([1,2,3,4,5], [0,1,0])] * 100))
print(net.accuracy([([1,2,3,4,5], [0,1,0])] * 100))'''
