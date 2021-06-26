import numpy as np

from sigmoid import sigmoid, deriv_sigmoid


class NeuralNetwork():

    def __init__(self, input_size, hidden_size, output_size):
        """Create neural network structure.

        Args:
            input_size (int): number of input neurons
            hidden_size (int): number of hidden neurons
            output_size (int): number of output neurons
        """     
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights1 = np.random.rand(self.hidden_size, self.input_size) * np.sqrt(1. / self.input_size)
        self.weights2 = np.random.randn(self.output_size, self.hidden_size) * np.sqrt(1. / self.hidden_size)

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

    def add_output(self, lst):          # popravi, ali rabim cel lst ali samo float?
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

        Returns:
            np.array: output of network
        """        
        return sigmoid(np.matmul(self.weights2, sigmoid(np.matmul(self.weights1, self.input))))

net = NeuralNetwork(4,3,2)
net.add_input([2,2,2,2])
net.add_output([2,2])

print(NeuralNetwork.feed_forward(net))