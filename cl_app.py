from collector import UnsplashCollector
from layer import Layer
from model import DeepNet
from util import get_layers, cli, split_data
import numpy as np

url_map = {
    'woodland': 'https://unsplash.com/collections/444531/woodland-animals',
    'reptile': 'https://unsplash.com/collections/2595311/reptiles',
    'bird': 'https://unsplash.com/collections/1451031/bird-.'
}

final_activations = len(list(url_map.keys()))

PATH = 'C:\Program Files (x86)\chromedriver.exe'

if __name__ == '__main__':
    collector = UnsplashCollector(url_map, PATH)
    training_data = collector.collect()
    np.random.shuffle(training_data)
    (training_data, testing_data) = split_data(training_data)
    cli(training_data, testing_data, final_activations, url_map)

