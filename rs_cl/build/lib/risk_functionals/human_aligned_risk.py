import jax.numpy as jnp

from .interface import RiskFunctionalInterface


class HumanAlignedRisk(RiskFunctionalInterface):
    """
        Parameters:
            a - weightage function parameter a
            b - weightage function parameter b
            reduction - (mean, sum, none)
                'none': no reduction is applied
                'mean': the weighted mean of the output is taken
                'sum': the output will be summed
    """
    def __init__(self, a=0.4, b=0.3, reduction='mean'):
        assert(reduction == 'none' or reduction == 'mean' or reduction == 'sum')
        super().__init__()
        self.a = a
        self.b = b
        self.reduction = reduction
      
    def _cpt_poly_derivative(self, a, b, Fx):
        return (3 - 3 * b) / (a**2 - a + 1) * (3 * Fx**2 - 2 * (a + 1) * Fx + a) + 1

    def forward(self, loss):
        empirical_cdf = jnp.argsort(jnp.argsort(loss, axis=0)) / len(loss)
        weighted_cdf = self._cpt_poly_derivative(self.a, self.b, empirical_cdf)
        loss *= weighted_cdf

        if self.reduction == 'mean':
            return jnp.mean(loss)
        elif self.reduction == 'sum':
            return jnp.sum(loss)
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Only mean, sum, none reduction types supported.')