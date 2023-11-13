import numpy as np

if __name__ == "__main__":
    from core import Layer, LayerWithParams
    from utils import im2col
else:
    from .core import Layer, LayerWithParams
    from .utils import im2col


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


class Sigmoid(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        out = 1 / (1 + np.exp(-x))
        self.ctx = out
        return out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        out = self.ctx
        return grad * out * (1 - out)


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


class Flatten(Layer):
    def __init__(self, in_shape: tuple):
        super().__init__()
        self.ctx = in_shape

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x.reshape(-1, 1)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad.reshape(self.ctx)


class Conv2d(LayerWithParams):
    def __init__(
        self,
        in_channels: int,
        num_filters: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        """
        Done with the help of this article:
        https://blog.ca.meron.dev/Vectorized-CNN/
        """
        super().__init__()

        self.params["W"] = np.random.randn(
            num_filters, in_channels, kernel_size, kernel_size
        )

        self.in_channels = in_channels
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Make some aliases for brevity
        K, P, S = self.kernel_size, self.padding, self.stride
        B, C, H, W = x.shape

        # Compute output dimensions
        out_h = (H - K + 2 * P) // S + 1
        out_w = (W - K + 2 * P) // S + 1

        cols = im2col(x, (B, C, out_h, out_w), K, P, S)

        # Cache for backward pass
        self.ctx = (x, cols)

        # Compute generalized matrix multiplication
        out = np.einsum("BCHWKk,FCKk->BFHW", cols, self.params["W"])

        return out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        x, cols = self.ctx

        # Make some aliases for brevity
        K, P, S = self.kernel_size, self.padding, self.stride
        P = K - 1 if P == 0 else P

        # Compute gradients w.r.t. the filters
        self.grads["W"] = np.einsum("BCHWKk,BFHW->FCKk", cols, grad)

        # Compute gradients w.r.t. the input
        grad_cols = im2col(grad, x.shape, K, P, 1, S - 1)
        rot_W = np.rot90(self.params["W"], 2, axes=(2, 3))

        return np.einsum("BFHWKk,FCKk->BCHW", grad_cols, rot_W)
