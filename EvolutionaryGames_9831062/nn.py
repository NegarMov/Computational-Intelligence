import numpy as np

class NeuralNetwork():

    def __init__(self, layer_sizes):
        # layer_sizes example: [5, 20, 1]
        self.W1 = np.random.randn(layer_sizes[1], layer_sizes[0]) * np.sqrt(2 / layer_sizes[0])
        self.b1 = np.zeros((layer_sizes[1], 1))

        self.W2 = np.random.randn(layer_sizes[2], layer_sizes[1]) * np.sqrt(2 / layer_sizes[1])
        self.b2 = np.zeros((layer_sizes[2], 1))

    def activation(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        # x example: np.array([[0.1], [0.2], [0.3]])
        z1 = np.dot(self.W1, x) + self.b1
        a1 = self.activation(z1)

        z2 = np.dot(self.W2, a1) + self.b2
        a2 = self.activation(z2)

        return a2
