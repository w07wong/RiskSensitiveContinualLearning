import jax.numpy as jnp

from .interface import RiskFunctionalInterface


class MeanVariance(RiskFunctionalInterface):
    """
        Parameters:
            c - variance penalty
            reduction - (mean, sum, none)
                'none': no reduction is applied
                'mean': the weighted mean of the output is taken
                'sum': the output will be summed
    """
    def __init__(self, c=0.1, reduction='mean'):
        super().__init__()
        self.c = c
        self.reduction = reduction
    
    def forward(self, loss):
        var = jnp.var(loss)
        mean = jnp.mean(loss)
 
        if self.reduction == 'mean':
            return mean + jnp.multiply(self.c, var)
        elif self.reduction == 'sum':
            return jnp.sum(loss + jnp.multiply(self.c, var))
        elif self.reduction == 'none':
            return loss + jnp.multiply(self.c, var)
        else:
            raise Exception('Only mean, sum, none reduction types supported.')