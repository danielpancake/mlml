import numpy as np


class Layer:
    def __init__(self):
        self.ctx = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    def backward(self, grad: np.ndarray) -> np.ndarray:
        pass


class LayerWithParams(Layer):
    def __init__(self):
        super().__init__()
        self.params = {}
        self.grads = {}

    def update(self, lr: float):
        for k in self.params:
            if k in self.grads:
                self.params[k] -= lr * self.grads[k]
