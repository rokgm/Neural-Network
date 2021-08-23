import numpy as np
import matplotlib.pyplot as plt
import tqdm

from bin.cost import mean_squared_error, mean_squared_error_der
from bin.sigmoid import sigmoid, sigmoid_der


class NeuralNetwork():

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001, epochs=1):
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
        self.epochs = epochs

        weights1 = np.random.rand(self.hidden_size, self.input_size) * np.sqrt(1. / self.input_size)
        weights2 = np.random.randn(self.output_size, self.hidden_size) * np.sqrt(1. / self.hidden_size)
        biases1 = np.random.randn(self.hidden_size) * np.sqrt(1. / self.input_size)
        biases2 = np.random.randn(self.output_size) * np.sqrt(1. / self.hidden_size)
        
        self.weights1 = weights1
        self.weights2 = weights2
        self.biases1 = biases1
        self.biases2 = biases2

    def __repr__(self):
        return 'NeuralNetwork(input_size={0.input_size}, hidden_size={0.hidden_size}, output_size={0.output_size}, learning_rate={0.learning_rate}, epochs={0.epochs})'.format(self)

    def add_input(self, lst):
        """Add input data to neural network. Normalize input before adding.

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
            lst (list): list of 0 with one 1 for desired output
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

    def backpropagate(self):
        """Computes gradient of weights and biases.
        """        
        delta_output = np.multiply(mean_squared_error_der(self.output_a, self.output), sigmoid_der(self.output_z))
        delta_hidden = np.multiply(np.dot(np.transpose(self.weights2), delta_output), sigmoid_der(self.hidden_z))
        delta_bias2 = delta_output
        delta_bias1 = delta_hidden
        delta_weights2 = np.outer(delta_output, self.hidden_a)
        delta_weights1 = np.outer(delta_hidden, self.input)
        return delta_weights1, delta_bias1, delta_weights2, delta_bias2

    def update_weights_biases(self):
        """Updates weights and biases of network with backpropagate.
        """        
        changes = self.backpropagate()
        self.weights1 -= self.learning_rate * changes[0]
        self.biases1 -= self.learning_rate * changes[1]
        self.weights2 -= self.learning_rate * changes[2]
        self.biases2 -= self.learning_rate * changes[3]

    def train(self, input_output_pairs, visualize_cost=False):  
        """Trains network with pairs of input and desired output vectors.

        Args:
            input_output_pairs (tuple of 2 vectors)
            visualize_cost (bool): Plot graph cost(iterations). Number of input_output_pairs must be >= 100.
        """ 
        if visualize_cost:

            if len(input_output_pairs) < 400:
                raise ValueError('For visualize_cost number of input-output pairs must be atleast 400.')

            plot_cost = np.array([])                  
            plot_epochs = np.array([])
            k = 0
            x_scaling_const = self.epochs / 400
            for epoch in tqdm.tqdm(range(self.epochs), 'Epochs'):
                for section in tqdm.tqdm(np.array_split(input_output_pairs, 400 / self.epochs), 'Epoch'):
                    for x, y in section:
                        self.add_input(x)
                        self.add_output(y)
                        self.feed_forward()
                        self.update_weights_biases()
                    plot_cost = np.append(plot_cost, mean_squared_error(self.output_a, self.output))
                    plot_epochs = np.append(plot_epochs, k * x_scaling_const)
                    k += 1
            plt.plot(plot_epochs, plot_cost)
            plt.xlabel('Epochs')
            plt.ylim(0,0.5)
            plt.ylabel('Mean Squared Error')
            plt.show()

        else:
            for epoch in tqdm.tqdm(range(self.epochs), 'Epochs'):
                for x, y in tqdm.tqdm(input_output_pairs, 'Iterations'):
                    self.add_input(x)
                    self.add_output(y)
                    self.feed_forward()
                    self.update_weights_biases()

    def evaluate(self, input_output_pairs):
        """Evaluates network with mean squared error of predicted vector of input and desired output vector.

        Args:
            input_output_pairs (tuple of 2 vectors)

        Returns:
            float: mean squared error
        """        
        f_forw = np.array([])
        desired = np.array([])
        for x, y in input_output_pairs:
            self.add_input(x)
            self.feed_forward()
            f_forw = np.append(f_forw, [self.output_a])
            desired = np.append(desired, y)
        return mean_squared_error(f_forw, desired)

    def accuracy(self, input_output_pairs):
        """Percentage of accurate network predictions.

        Args:
            input_output_pairs (tuple of 2 vectors)

        Returns:
            float: percentage of correct predictions
        """ 
        pred = np.array([])
        for x, y in input_output_pairs:
            self.add_input(x)
            self.feed_forward()
            pred = np.append(pred, np.argmax(self.output_a) == np.argmax(y))
        return '{} %'.format(np.mean(pred) * 100)
    
    def predict(self, x):
        '''Returns index of maximum of predicted output vector.
        '''
        self.add_input(x)
        return np.argmax(self.feed_forward())

    def save_network(self, filename):
        """Saves network structure and parameters.

        Args:
            filename (str): name of file to save
        """        
        np.savez(filename, weights1=self.weights1, weights2=self.weights2, biases1=self.biases1, biases2=self.biases2)

    @classmethod
    def load_network(cls, filename):
        """Load network structure and parameters saved by save_network.

        Args:
            filename (str): name of file to load

        Returns:
            instance of NeuralNetwork
        """        
        params = np.load(filename)
        weights1, weights2, biases1, biases2 = params['weights1'], params['weights2'], params['biases1'], params['biases2']
        
        network = cls(np.shape(weights1)[1], np.shape(weights2)[1], np.shape(weights2)[0])
        network.weights1 = weights1
        network.weights2 = weights2
        network.biases1 = biases1
        network.biases2 = biases2
        return network

    @staticmethod
    def one_hot_encoder(indx_lst, vect_len):
        """Creates vector of zeros with one 1 value at given index.

        Args:
            indx_lst (int)
            vect_len (int): lenght of vector

        Returns:
            np.array
        """        
        indx_lst = np.array(indx_lst)
        hot_enc = np.zeros((indx_lst.size, vect_len))
        hot_enc[np.arange(indx_lst.size), indx_lst] = 1
        return hot_enc