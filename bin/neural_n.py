import numpy as np

from sigmoid import sigmoid, sigmoid_der
from cost import mean_squared_error, mean_squared_error_der


class NeuralNetwork():

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001):
        """Create neural network structure.

        Args:
            input_size (int): number of input neurons
            hidden_size (int): number of hidden neurons
            output_size (int): number of output neurons
        """     
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.weights1 = np.random.rand(self.hidden_size, self.input_size) * np.sqrt(1. / self.input_size)
        self.weights2 = np.random.randn(self.output_size, self.hidden_size) * np.sqrt(1. / self.hidden_size)
        self.biases1 = np.random.randn(self.hidden_size) * np.sqrt(1. / self.input_size)
        self.biases2 = np.random.randn(self.output_size) * np.sqrt(1. / self.hidden_size)

    def __repr__(self):
        return 'NeuralNetwork(input_size={0.input_size}, hidden_size={0.hidden_size}, output_size={0.output_size})'.format(self)

    def add_input(self, lst):
        """Add input data to neural network.

        Args:
            lst (list)
        """
        lst = np.array(lst) 
        if self.input_size == len(lst):
            self.input = lst
        else:
            raise ValueError('Size of input and number of input neurons do not match.')

    def add_output(self, lst):
        """Add output data to neural network.

        Args:
            lst (list)
        """
        lst = np.array(lst) 
        if self.output_size == len(lst):
            self.output = lst
        else:
            raise ValueError('Size of output and number of output neurons do not match.')

    def feed_forward(self):       
        """Propagates input through the network.
        """
        self.hidden_z = np.dot(self.weights1, np.transpose(self.input)) + self.biases1
        self.hidden_a = sigmoid(self.hidden_z)
        self.output_z = np.dot(self.weights2, self.hidden_a) + self.biases2
        self.output_a = sigmoid(self.output_z)
        return self.output_a

    def backpropagation(self):
        delta_output = np.multiply(mean_squared_error_der(self.output_a, self.output), sigmoid_der(self.output_z))
        print(delta_output)
        delta_hidden = np.multiply(np.dot(np.transpose(self.weights2), delta_output), sigmoid_der(self.hidden_z))
        print(delta_hidden)
        delta_bias2 = delta_output
        delta_bias1 = delta_hidden
        print(np.shape(delta_output), np.shape(self.hidden_a))
        delta_weights2 = np.dot(np.transpose(delta_output), self.hidden_a)              # popravi dimenzije, spremeni v matmul?


net = NeuralNetwork(5,4,3)
net.add_input([2,5,6,7,8])
net.add_output([0,1,0])

print(NeuralNetwork.feed_forward(net))
NeuralNetwork.backpropagation(net)