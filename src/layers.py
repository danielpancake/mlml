import numpy as np

if __name__ == "__main__":
    from core import Layer, LayerWithParams
else:
    from .core import Layer, LayerWithParams


class Linear(LayerWithParams):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.params["W"] = np.random.randn(in_dim, out_dim)
        self.params["b"] = np.random.randn(1, out_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.ctx = x
        return np.matmul(x, self.params["W"]) + self.params["b"]

    def backward(self, grad: np.ndarray) -> np.ndarray:
        self.grads["W"] = np.matmul(self.ctx.T, grad)
        self.grads["b"] = np.sum(grad, axis=0)
        return np.matmul(grad, self.params["W"].T)


class ReLU(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.ctx = x > 0
        return x * self.ctx

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self.ctx


class Softmax(Layer):
    def __init__(self, axis: int = 1):
        super().__init__()
        # Commonly, axis is `len(x.shape) - 1` assuming batch is the first dimension
        self.axis = axis

    def forward(self, x: np.ndarray) -> np.ndarray:
        exp = np.exp(x - np.max(x, axis=self.axis, keepdims=True))
        out = exp / np.sum(exp, axis=self.axis, keepdims=True)

        self.ctx = out
        return out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        s = self.ctx
        return grad * s * (1 - s)
