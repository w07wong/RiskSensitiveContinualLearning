from flax import linen as nn


class MLPSimple(nn.Module):
    """MLP model to evaluate gradient alignment.'"""
    out_features: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=1)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.out_features)(x)
        return x