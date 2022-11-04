from flax import linen as nn


class CNN(nn.Module):
    """A simple CNN model."""
    train: bool
    out_features: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=256, kernel_size=(3, 3))(x)
        # x = nn.BatchNorm(use_running_average=not self.train)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2))
        x = nn.Conv(features=256, kernel_size=(3, 3))(x)
        # x = nn.BatchNorm(use_running_average=not self.train)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.out_features)(x)
        return x