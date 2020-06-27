from collector import UnsplashCollector
from layer import Layer
from model import DeepNet
import numpy as np

url_map = {
    'dog': 'https://unsplash.com/collections/9718726/dog',
    'cat': 'https://unsplash.com/collections/139386/cats'
}

final_activations = len(list(url_map.keys()))

PATH = 'C:\Program Files (x86)\chromedriver.exe'

def sigmoid(z):
    res = 1/(1+np.exp(-z))
    return res

def compute_cost_derivative(prediction, actual, m):
    return (prediction-actual)/m

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def compute_cost(prediction, actual, m):
    return np.sum((prediction - actual)**2)/(2*m)

if __name__ == '__main__':
    collector = UnsplashCollector(url_map, PATH)
    training_data = collector.collect()
    np.random.shuffle(training_data)
    split = np.split(training_data, 2)
    training_data = split[0]
    testing_data = split[1]

    model = DeepNet([Layer(100), Layer(50), Layer(final_activations)], training_data)
    model.stochastic_descent(35, 10, 2.5)
    model.test(testing_data)

