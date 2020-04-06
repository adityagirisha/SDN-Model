import math, random
import numpy as np

class NN:

    def __init__(self, input_dim=None, output_dim=None, hidden_layers=None, seed=1):
        if (input_dim is None) or (output_dim is None) or (hidden_layers is None):
            raise Exception("Invalid arguments given!")
        self.input_dim = input_dim # number of input nodes
        self.output_dim = output_dim # number of output nodes
        self.hidden_layers = hidden_layers # number of hidden nodes @ each layer
        self.network = self._build_network(seed=seed)

    
    def setNetworkDetails(self,payload):
        self.network=payload["network"]
        self.input_dim=payload["input_dim"]
        self.output_dim=payload["output_dim"]
        self.hidden_layers=payload["hidden_layers"]
        print("network uploaded")
        
    def getNetworkDetails(self):
        payload={
            "network":self.network,
            "input_dim":self.input_dim,
            "output_dim":self.output_dim,
            "hidden_layers":self.hidden_layers
        }
        return payload
    
    
    def train(self, X, y, eta=0.5, n_epochs=200):
        for epoch in range(n_epochs):
            for (x_, y_) in zip(X, y):
                self._forward_pass(x_) 
                yhot_ = self._one_hot_encoding(y_, self.output_dim) 
                self._backward_pass(yhot_) 
                self._update_weights(x_, eta) 

    
    def predict(self, X):
        ypred = np.array([np.argmax(self._forward_pass(x_)) for x_ in X], dtype=np.int)
        return ypred


    
    def _build_network(self, seed=1):
        random.seed(seed)
        def _layer(input_dim, output_dim):
            layer = []
            for i in range(output_dim):
                weights = [random.random() for _ in range(input_dim)] 
                node = {"weights": weights, 
                        "output": None, 
                        "delta": None} 
                layer.append(node)
            return layer

        
        network = []
        if len(self.hidden_layers) == 0:
            network.append(_layer(self.input_dim, self.output_dim))
        else:
            network.append(_layer(self.input_dim, self.hidden_layers[0]))
            for i in range(1, len(self.hidden_layers)):
                network.append(_layer(self.hidden_layers[i-1], self.hidden_layers[i]))
            network.append(_layer(self.hidden_layers[-1], self.output_dim))
        return network

    
    def _forward_pass(self, x):
        transfer = self._sigmoid
        x_in = x
        for layer in self.network:
            x_out = []
            for node in layer:
                node['output'] = transfer(self._dotprod(node['weights'], x_in))
                x_out.append(node['output'])
            x_in = x_out 
        return x_in

    
    def _backward_pass(self, yhot):
        transfer_derivative = self._sigmoid_derivative
        n_layers = len(self.network)
        for i in reversed(range(n_layers)):
            if i == n_layers - 1:
                for j, node in enumerate(self.network[i]):
                    err = node['output'] - yhot[j]
                    node['delta'] = err * transfer_derivative(node['output'])
            else:
                for j, node in enumerate(self.network[i]):
                    err = sum([node_['weights'][j] * node_['delta'] for node_ in self.network[i+1]])
                    node['delta'] = err * transfer_derivative(node['output'])

    
    def _update_weights(self, x, eta):
        for i, layer in enumerate(self.network):
            if i == 0: inputs = x
            else: inputs = [node_['output'] for node_ in self.network[i-1]]
            for node in layer:
                for j, input in enumerate(inputs):
                    node['weights'][j] += - eta * node['delta'] * input

   
    def _dotprod(self, a, b):
        return sum([a_ * b_ for (a_, b_) in zip(a, b)])

    
    def _sigmoid(self, x):
        return 1.0/(1.0+math.exp(-x))

    
    def _sigmoid_derivative(self, sigmoid):
        return sigmoid*(1.0-sigmoid)

    
    def _one_hot_encoding(self, idx, output_dim):
        x = np.zeros(output_dim, dtype=np.int)
        x[idx] = 1
        return x