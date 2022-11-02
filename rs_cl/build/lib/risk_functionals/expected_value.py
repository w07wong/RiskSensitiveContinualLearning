import jax.numpy as jnp

from .interface import RiskFunctionalInterface


class ExpectedValue(RiskFunctionalInterface):
    """
        Parameters:
            percent - percent (0.0, 1.0] of losses to optimize over
            reduction - (mean, sum, none)
                'none': no reduction is applied
                'mean': the weighted mean of the output is taken
                'sum': the output will be summed
    """
    def __init__(self, percent=1.0, reduction='mean'):
        super().__init__()
        assert (percent > 0.0 and percent <= 1.0)
        self.percent = percent
        self.reduction = reduction
    
    def forward(self, loss):
        if self.percent != 1.0:
          num_samples = int(len(loss) * self.percent)
          loss = loss[:num_samples]

        if self.reduction == 'mean':
            return jnp.mean(loss) 
        elif self.reduction == 'sum':
            return jnp.sum(loss)
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Only mean, sum, none reduction types supported.')