import pickle
from collections import OrderedDict
from util import *


class TwoLayerNet:
    def __init__(self, input_size, hidden_size_list, output_size, l2_lambda=0):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.l2_lambda = l2_lambda
        self.params = {}
        self.init_weight()
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num + 1):
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                      self.params['b' + str(idx)])
            self.layers['Activation_function' + str(idx)] = Relu()

        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                  self.params['b' + str(idx)])

        self.last_layer = SoftmaxWithLoss()

    def init_weight(self):
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = np.sqrt(2.0 / all_size_list[idx - 1])
            W = scale * np.random.randn(all_size_list[idx - 1], all_size_list[idx])
            b = np.zeros(all_size_list[idx])
            self.params['W' + str(idx)] = W
            self.params['b' + str(idx)] = b

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        weight_decay = 0.5 * self.l2_lambda * sum(np.sum(W ** 2) for key, W in self.params.items() if 'W' in key)
        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.mean(y == t)
        return accuracy

    def gradient(self, x, t):
        self.loss(x, t)
        dout = self.last_layer.backward()
        for layer in reversed(self.layers.values()):
            dout = layer.backward(dout)
        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            affine_layer = self.layers['Affine' + str(idx)]
            grads['W' + str(idx)] = affine_layer.dW + self.l2_lambda * affine_layer.W
            grads['b' + str(idx)] = affine_layer.db

        return grads


    def save_params(self, file_name="params.pkl"):
        with open(file_name, 'wb') as f:
            pickle.dump(self.params, f)
        print("\n ! The parameters have been saved in params.pkl !\n")

    def load_params(self, file_name='params.pkl'):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for i, key in enumerate(['Affine1', 'Affine2']):
            affine_layer = self.layers[key]
            affine_layer.W = params['W' + str(i + 1)]
            affine_layer.b = params['b' + str(i + 1)]


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx
