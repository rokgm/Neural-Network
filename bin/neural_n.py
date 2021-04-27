import numpy as np


class NeuralNetwork():

    def __init__(self, input_n, hidden_n, output_n):
        """
        Create neural network structure.

        Args:
            input_n (int): number of input neurons
            hidden_n (int): number of hidden neurons
            output_n (int): number of output neurons
        """     
        self.input_n = input_n
        self.hidden_n = hidden_n
        self.output_n = output_n
        self.weights1 = np.random.rand(self.hidden_n, self.input_n) * np.sqrt(1. / self.input_n)
        self.weights2 = np.random.randn(self.output_n, self.hidden_n) * np.sqrt(1. / self.hidden_n)

    def __repr__(self):
        return 'NeuralNetwork(input_n={0.input_n}, hidden_n={0.hidden_n}, output_n={0.output_n})'.format(self)

    def add_input(self, lst):
        """
        Add input data to neural network.

        Args:
            lst (list)
        """
        lst = np.array(lst) 
        if self.input_n == len(lst):
            self.input = lst
        else:
            raise ValueError('Size of input and number of input neurons do not match.')

    def add_output(self, lst):
        """
        Add output data to neural network.

        Args:
            lst (list)
        """
        lst = np.array(lst) 
        if self.output_n == len(lst):
            self.output = lst
        else:
            raise ValueError('Size of output and number of input neurons do not match.')

net = NeuralNetwork(4,3,2)
net.add_input([2,2,2,2])
net.add_output([2,2])