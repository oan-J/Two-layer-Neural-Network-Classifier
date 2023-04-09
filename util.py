import numpy as np


def shuffle(x, t):
    permutation = np.random.permutation(x.shape[0])
    x = x[permutation]
    t = t[permutation]
    return x, t


def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    return softmax_x


def cross_entropy_loss(y, t):
    if t.ndim == 1:
        t = np.eye(y.shape[1])[t]
    delta = 1e-7
    batch_size = y.shape[0]
    loss = -np.sum(t * np.log(y + delta)) / batch_size
    return loss


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        out = np.dot(self.x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = dx.reshape(*self.original_x_shape)
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_loss(self.y, self.t)
        return self.loss

    def backward(self):
        batch_size = self.t.shape[0]
        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx = dx / batch_size
        return dx
