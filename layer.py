# from utils import relu_backward, sigmoid_derivative, sigmoid, relu, tanh_backward
import numpy as np

class Layer:
    def __init__(self, depth, activation='relu'):
        self.depth = depth
        self.weights = []
        self.biases = []
        self.z = []
        self.delW = []
        self.delB = []

    def update(self, lr, m):
        self.weights = self.weights - (lr/m)*self.delW
        self.biases = self.biases - (lr/m)*self.delB
    
    def say(self):
        pass

     
    
       