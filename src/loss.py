import numpy as np

if __name__ == "__main__":
    from core import LossFunction
else:
    from .core import LossFunction


class CrossEntropy(LossFunction):
    eps = 1e-15

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        assert (
            y_pred.shape == y_true.shape
        ), "y_pred and y_true must have the same shape"

        self.ctx = (y_pred, y_true)
        return -np.sum(y_true * np.log(y_pred + self.eps))

    def backward(self) -> np.ndarray:
        y_pred, y_true = self.ctx
        return -y_true / (y_pred + self.eps)
