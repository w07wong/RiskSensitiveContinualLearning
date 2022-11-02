import jax.numpy as jnp

from .interface import RiskFunctionalInterface


class TrimmedRisk(RiskFunctionalInterface):
    """
        Parameters:
            alpha - quantile of tails to ignore
            reduction - (mean, sum, none)
                'none': no reduction is applied
                'mean': the weighted mean of the output is taken
                'sum': the output will be summed
    """
    def __init__(self, alpha=0.05, reduction='mean'):
        super().__init__()
        assert alpha >= 0 and alpha <= 0.5, 'alpha must be in [0, 0.5]'
        self.alpha = alpha
        self.reduction = reduction
    
    """Gets losses not on the tail ends of the distribution."""
    def _get_untrimmed_losses(self, loss):
        sorted_indices = jnp.argsort(loss, axis=0)
        empirical_cdf = jnp.argsort(sorted_indices, axis=0) / (len(loss) - 1)
        return jnp.where(
            (empirical_cdf >= self.alpha) & (empirical_cdf <= 1 - self.alpha),
            size=len(loss))[0]
    
    def forward(self, loss):
        untrimmed_losses = self._get_untrimmed_losses(loss)
        
        if self.reduction == 'mean':
            return (jnp.sum(jnp.take(loss, untrimmed_losses, axis=0)) / 
                    ((1 - 2 * self.alpha) * len(loss)))
        elif self.reduction == 'sum':
            return jnp.sum(jnp.take(loss, untrimmed_losses, axis=0))
        elif self.reduction == 'none':
            return jnp.take(loss, untrimmed_losses, axis=0)
        else:
            raise Exception('Only mean, sum, none reduction types supported.')