import matplotlib.pyplot as plt
import numpy as np

from bin.neural_n import NeuralNetwork as NN

N_train = 5000   # Number of training examples from file
train_data = np.zeros([N_train, 785])
with open('data_set/mnist_train.csv', 'r') as data:
    for x in range(N_train):
        train_data[x] = np.fromstring(next(data), sep=',')

N_test = 3   # Number of testing examples from file
test_data = np.zeros([N_test, 785])
with open('data_set/mnist_test.csv', 'r') as data:
    for x in range(N_test):
        test_data[x] = np.fromstring(next(data), sep=',')

fac = 1 / 255
train_imgs = (train_data[:, 1:]) * fac
test_imgs = (test_data[:, 1:]) * fac

train_labels = train_data[:, :1].transpose()[0,:].astype(np.int)
test_labels = test_data[:, :1].transpose()[0,:].astype(np.int)

net = NN(784,300,10, learning_rate=0.001, epochs=1)
train_pairs = list(zip(train_imgs, NN.one_hot_encoder(train_labels, 10)))
net.train(train_pairs, visualize_cost=True)

'''img = test_imgs[0].reshape((28,28))
plt.imshow(img, cmap='Greys')
plt.show()'''