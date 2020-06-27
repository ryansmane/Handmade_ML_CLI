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

if __name__ == '__main__':
    collector = UnsplashCollector(url_map, PATH)
    training_data = collector.collect()
    np.random.shuffle(training_data)
    split = np.split(training_data, 2)
    training_data = split[0]
    testing_data = split[1]

    model = DeepNet([Layer(100), Layer(50), Layer(final_activations)], training_data)
    model.stochastic_descent(25, 10, 2.5)
    (x, y, y_hat) = model.test(testing_data)
    model.display_labels(x, y, y_hat, url_map, 7)

