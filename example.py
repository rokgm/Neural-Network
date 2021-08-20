import matplotlib.pyplot as plt
import numpy as np

from bin.neural_n import NeuralNetwork as NN

N_train = 600   # Number of training examples from file
train_data = np.zeros([N_train, 785])
with open('data_set/mnist_train.csv', 'r') as data:
    for x in range(N_train):
        train_data[x] = np.fromstring(next(data), sep=',')

N_test = 100   # Number of testing examples from file
test_data = np.zeros([N_test, 785])
with open('data_set/mnist_test.csv', 'r') as data:
    for x in range(N_test):
        test_data[x] = np.fromstring(next(data), sep=',')

fac = 1 / 255
train_imgs = (train_data[:, 1:]) * fac
test_imgs = (test_data[:, 1:]) * fac

train_labels = train_data[:, :1].transpose()[0,:].astype(np.int)
test_labels = test_data[:, :1].transpose()[0,:].astype(np.int)

train_pairs = list(zip(train_imgs, NN.one_hot_encoder(train_labels, 10)))
test_pairs = list(zip(test_imgs, NN.one_hot_encoder(test_labels, 10)))

net = NN(784,600,10, learning_rate=0.01, epochs=1)
net.train(train_pairs, visualize_cost=True)
#net.save_network('net1.npz')

print(net.evaluate(test_pairs))
print(net.accuracy(test_pairs))

'''print(train_pairs[5070][1])
plt.imshow(train_pairs[5070][0].reshape(28,28), cmap='Greys')
plt.plot()
plt.show()'''