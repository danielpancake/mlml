import numpy as np

from warnings import warn


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
            else:
                warn(f"Parameter {k} has no gradient")


class LossFunction:
    def __init__(self):
        self.ctx = None

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return self.forward(y_pred, y_true)

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        pass

    def backward(self) -> np.ndarray:
        pass
