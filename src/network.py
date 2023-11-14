import numpy as np

from tqdm import tqdm

if __name__ == "__main__":
    from core import LayerWithParams, LossFunction
else:
    from .core import LayerWithParams, LossFunction


class Network:
    def __init__(self, layers: list, loss: LossFunction, metric: callable):
        self.layers = layers
        self.layers_trainable = [
            layer for layer in layers if isinstance(layer, LayerWithParams)
        ]

        self.loss = loss
        self.metric = metric

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def update(self, lr: float):
        for layer in self.layers_trainable:
            layer.update(lr)

    @staticmethod
    def split_train_val(x: np.ndarray, y: np.ndarray, train_val_split: float) -> tuple:
        if train_val_split <= 0 or train_val_split >= 1 or np.isnan(train_val_split):
            raise ValueError("train_val_split must be between 0 and 1")

        train_val_split_mask = np.random.rand(x.shape[0]) < train_val_split

        x_train, y_train = x[train_val_split_mask], y[train_val_split_mask]
        x_val, y_val = x[~train_val_split_mask], y[~train_val_split_mask]

        return x_train, y_train, x_val, y_val

    def train_step(self, x: np.ndarray, y: np.ndarray, lr: float) -> float:
        out = self.forward(x)

        loss = self.loss(out, y)
        grad = self.loss.backward()

        self.backward(grad)
        self.update(lr)

        return loss

    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lr: float,
        epochs: int,
        train_val_split: float = 0.8,
        val_x: np.ndarray = None,
        val_y: np.ndarray = None,
        shuffle: bool = False,
    ):
        # Shuffle dataset
        if shuffle:
            shuffle_mask = np.random.permutation(len(x))
            x = x[shuffle_mask]
            y = y[shuffle_mask]

        for epoch in range(epochs):
            # Split into training and validation set
            if val_x is not None and val_y is not None:
                x_train, y_train = x, y
                x_val, y_val = val_x, val_y
            else:
                x_train, y_train, x_val, y_val = self.split_train_val(
                    x, y, train_val_split
                )
            x_train_size, x_val_size = x_train.shape[0], x_val.shape[0]

            update_interval = max(1, x_train_size // 100)

            print(f"Epoch {epoch + 1}/{epochs}")

            train_losses = []
            e_tqdm = tqdm(range(x_train_size))

            for i in e_tqdm:
                train_loss = self.train_step(x_train[i], y_train[i], lr)
                train_losses.append(train_loss)

                if i % update_interval == 0:
                    e_tqdm.set_postfix({"loss": np.mean(train_losses)})

            # Validation
            if x_val_size > 0:
                print("Evaluating...", end=" ")
                acc = self.evaluate(x_val, y_val)
                print(f"Validation score: {acc * 100:.5f}%")

            e_tqdm.close()

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        metric_vals = []

        for i in range(len(x)):
            out = self.forward(x[i])
            metric_vals.append(self.metric(out, y[i]))

        return np.mean(metric_vals)
