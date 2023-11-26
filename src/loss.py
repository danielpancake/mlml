import numpy as np

if __name__ == "__main__":
    from core import LossFunction
else:
    from .core import LossFunction


class CrossEntropy(LossFunction):
    eps = 1e-9

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        assert (
            y_pred.shape == y_true.shape
        ), f"y_pred and y_true must have the same shape, got {y_pred.shape} and {y_true.shape}"

        self.ctx = (y_pred, y_true)
        return -np.sum(y_true * np.log(y_pred + self.eps))

    def backward(self) -> np.ndarray:
        y_pred, y_true = self.ctx
        return -y_true / (y_pred + self.eps)


class SSE(LossFunction):
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        assert (
            y_pred.shape == y_true.shape
        ), f"y_pred and y_true must have the same shape, got {y_pred.shape} and {y_true.shape}"

        self.ctx = (y_pred, y_true)
        return np.sum((y_pred - y_true) ** 2)

    def backward(self) -> np.ndarray:
        y_pred, y_true = self.ctx
        return 2 * (y_pred - y_true)
