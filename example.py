from bin.neural_n import NeuralNetwork

'''# If enough ram is available can ce loaded with np.loadtxt
#  but might return MemoryError: cannot allocate memory for array
train_data = np.empty([60000,785])
row = 0
with open('dataset/mnist_train.csv', 'r') as data:
    for line in data:
        train_data[row] = np.fromstring(line, sep=',')

test_data = np.empty([10000,785])
row = 0
with open('dataset/mnist_test.csv', 'r') as data:
    for line in data:
        test_data[row] = np.fromstring(line, sep=',')

fac = 1 / 255    # Can be 0.99/255 + 0.01 to avoid 0 values, which can prevent weight updates

train_imgs = np.asfarray(train_data[:, 1:]) * fac
test_imgs = np.asfarray(test_data[:, 1:]) * fac

train_labels = np.asfarray(train_data[:, :1])
test_labels = np.asfarray(test_data[:, :1])

lr = np.arange(10)

train_labels_one_hot = (lr==train_labels).astype(np.int)
test_labels_one_hot = (lr==test_labels).astype(np.int)
print(train_labels_one_hot[243])
print(test_labels_one_hot[7518])'''

'''from mnist import MNIST

mndata = MNIST('dataset')
images_train, labels_train = mndata.load_training()
images_test, labels_test = mndata.load_testing()'''

from sklearn.datasets import fetch_openml
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

from bin.neural_n import NeuralNetwork

x, y = fetch_openml('mnist_784', version=1, return_X_y=True)
x = (x/255).astype('float32')
y = to_categorical(y)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, random_state=42)
net = NeuralNetwork(784,300,10, learning_rate=0.01, epochs=1)
net.train(list(zip(x_train, y_train))[500], visualize_cost=False)
print(net.evaluate(list(zip(x_train, y_train))[100]))
print(net.accuracy(list(zip(x_train, y_train))[100]))


'''net = NeuralNetwork(5,4,3, learning_rate=0.1, epochs=1)
net.train([([1,2,3,4,5], [0,1,0])] * 400, visualize_cost=True)
print(net.evaluate([([1,2,3,4,5], [0,1,0])] * 100))
print(net.accuracy([([1,2,3,4,5], [0,1,0])] * 100))'''
