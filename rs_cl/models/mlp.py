from flax import linen as nn


class MLP(nn.Module):
    """MLP model from 'Re-evaluating Continual Learning Scenarios- A 
    Categorization and Case for Strong Baselines'"""
    out_features: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=400)(x)
        x = nn.relu(x)
        x = nn.Dense(features=400)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.out_features)(x)
        return x