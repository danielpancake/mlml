import numpy as np

if __name__ == "__main__":
    from core import Layer, LayerWithParams
else:
    from .core import Layer, LayerWithParams


class Linear(LayerWithParams):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.params["W"] = np.random.randn(out_dim, in_dim) * np.sqrt(2 / in_dim)
        self.params["b"] = np.zeros((out_dim, 1))

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.ctx = x
        return np.matmul(self.params["W"], x) + self.params["b"]

    def backward(self, grad: np.ndarray) -> np.ndarray:
        self.grads["W"] = np.matmul(grad, self.ctx.T)
        self.grads["b"] = np.sum(grad, axis=1, keepdims=True)
        return np.matmul(self.params["W"].T, grad)


class ReLU(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.ctx = x > 0
        return x * self.ctx

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self.ctx


class Softmax(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        exp = np.exp(x - np.max(x))
        inv_sum = 1 / np.sum(exp, keepdims=True)
        out = exp * inv_sum

        self.ctx = (exp, inv_sum, out)
        return out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        exp, inv_sum, out = self.ctx
        return out * (grad - np.sum(grad * exp, keepdims=True) * inv_sum)
