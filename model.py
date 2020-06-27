import numpy as np
def sigmoid(z):
    res = 1/(1+np.exp(-z))
    return res

def compute_cost_derivative(prediction, actual, m):
    return (prediction-actual)/m

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def compute_cost(prediction, actual, m):
    return np.sum((prediction - actual)**2)/(2*m)

class DeepNet():
    def __init__(self, layers, training_data):
        self.layers = layers
        self.num_layers = len(layers)
        self.num_features = len(training_data[0][0])
        self.num_training_examples = training_data.shape[0]
        self.initialize_layers()
        self.activations = []
        self.zs = []
        self.training_data = training_data

    def initialize_layers(self):
        features = self.num_features
        for l in self.layers:
            l.weights = np.random.randn(l.depth, features)*np.sqrt(l.depth/features)
            l.biases = np.zeros((l.depth, 1))
            features = l.depth
    
    def feed_forward(self, x, y):
        activations = [x]
        zs = []
        prev_act = x
        for l in self.layers:
            z = np.dot(l.weights, prev_act)
            a = sigmoid(z + l.biases)
            prev_act = a
            zs.append(z)
            activations.append(a)
        activations = np.array(activations)
        zs = np.array(zs)
        return (activations, zs)
  

    def stochastic_descent(self, epochs, batch_size, lr):
        for e in range(epochs):
            np.random.shuffle(self.training_data)
            examples = self.num_training_examples
            mini_batches = np.array([self.training_data[k:k+batch_size] for k in range(0, examples, batch_size)])
            current_y = []
            for b in self.layers:
                b.say()
            for j, batch in enumerate(mini_batches):
                x,y = zip(*batch)
                x=np.array(x).T
                y=np.array(y).T
                current_y = y
                (activations, zs) = self.feed_forward(x,y)
                cost_derivative = compute_cost_derivative(activations[-1], y, batch_size)
                delta = cost_derivative * sigmoid_prime(zs[-1])
                self.layers[-1].delW = np.dot(delta, activations[-2].T)
                self.layers[-1].delB = np.sum(delta, axis=1, keepdims=True)
                for l in reversed(range(1, self.num_layers)):
                    layer = self.layers[l-1]
                    delta = np.dot(self.layers[l].weights.T, delta) * sigmoid_prime(zs[l-1])
                    layer.delW = np.dot(delta, activations[l-1].T)
                    layer.delB = np.sum(delta, axis=1, keepdims=True)
                for p in self.layers:
                    p.update(lr, batch_size)
            print(f'cost after epoch {e}: {compute_cost(activations[-1], current_y, batch_size)}')

    def test(self, data):
        x,y = zip(*data)
        x=np.array(x).T
        y=np.array(y).T
        (activations, zs) = self.feed_forward(x,y)
        print(activations[-1][0:3, 0:3])
        print(y[0:3,0:3])
        print(f'cost: {compute_cost(activations[-1], y, x.shape[1])}')


                

                
