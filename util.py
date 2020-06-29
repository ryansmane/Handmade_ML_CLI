from layer import Layer
from model import DeepNet
import ast
from pydoc import locate
import numpy as np

def get_layers(layers, num_categories):
    res = [Layer(depth) for depth in layers]
    res.append(Layer(num_categories))
    return res
    
def test_for_type(x, type):
    return 

def query_for_type(q, t):
    res = input(q)
    if t == 'tuple':
        try:
            trans = ast.literal_eval(res)
            if isinstance(trans, tuple):
                return trans
            else:
                print('Not a tuple.')
                query_for_type(q, t)
        except ValueError:
            print('Invalid type')
            query_for_type(q, t)
    elif t == 'int':
        try:
            i = int(res)
            if float(res) == int(res):
                return i
            else:
                print('Not an integer')
                query_for_type(q, t)
        except ValueError:
            print('Not an integer')
            query_for_type(q, t)
    elif t == 'float':
        try:
            i = float(res)
            return i 
        except ValueError:
            print('Not a float')
            query_for_type(q, t)
    else:
        return False

def cli(training_data, testing_data, final_activations, url_map):
    q1 = "Enter hidden layers (tuple): \nE.g., (100, 50) 2 layers with 100 and 50 activations respectively \n"
    q2 = "No. of Epochs: "
    q3 = "Batch size: "
    q4 = "Learning rate: "
    q5 = "How many predictions would you like plotted? \n"
    wish = query_for_type(q1, 'tuple')
    print("Enter hyperparameters as follows: ")
    epochs = query_for_type(q2, 'int')
    batch_size = query_for_type(q3, 'int')
    learning_rate = query_for_type(q4, 'float')
    layers = get_layers(wish, final_activations)
    model = DeepNet(layers, training_data)
    model.stochastic_descent(int(epochs), int(batch_size), float(learning_rate))
    ans = input("Do you wish to run an evaluation? (y/n)")
    if ans == 'y':
        show_limit = query_for_type(q5, 'int')
        (x, y, y_hat) = model.test(testing_data)
        model.display_labels(x, y, y_hat, url_map, int(show_limit))
        single = input('Do you wish to try a single image? (y/n)')
        if single == 'y':
            path = input('Enter -in_directory- path to image file: \n')
            model.test_single(path, url_map)
        a = input("Do you wish to try a different model? (y/n)")
        if a == 'y':
            cli(training_data, testing_data, final_activations, url_map)
        elif a =='n':
            print('Exiting...')
    elif ans == 'n':
        single = input('Do you wish to try a single image? (y/n)')
        if single == 'y':
            path = input('Enter -in_directory- name of file: \n E.g., bird3.png')
            model.test_single(path, url_map)
        a = input("Do you wish to try a different model? (y/n)")
        if a == 'y':
            cli(training_data, testing_data, final_activations, url_map)
        elif a =='n':
            print('Exiting...')

def split_data(training_data):
    try:
        l = training_data.shape[0]
        s = input("No. of testing examples: ")
        if int(s) == float(s) and int(s) > 0:
            s = int(s)
            fraction = int(l/s)
            parts = np.split(training_data, fraction)
            testing_data = parts[-1]
            tup=tuple(parts[x] for x in range(fraction))
            res = np.concatenate(tup)
            return (res, testing_data)
        else:
            print('Invalid input (test set cannot be non-empty)')
            split_data(training_data)
    except ValueError:
        print("An uneven split occured. Try a nicer number.")
        split_data(training_data)
