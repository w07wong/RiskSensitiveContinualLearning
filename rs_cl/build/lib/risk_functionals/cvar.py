import jax.numpy as jnp

from .interface import RiskFunctionalInterface


"""Class for CVaR and inverted CVaR risk."""
class CVaR(RiskFunctionalInterface):
    """
        Parameters:
            alpha - CVaR alpha
            inverted - To use CVaR, set inverted to False. To use inverted CVaR, set inverted to True.
            reduction - (mean, sum, none)
                'none': no reduction is applied
                'mean': the weighted mean of the output is taken
                'sum': the output will be summed
    """
    def __init__(self, alpha=0.05, inverted=False, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.inverted = inverted
        self.reduction = reduction
    
    """Computes the value at risk specified by the 'a' parameter."""
    def _value_at_risk(self, loss):
        sorted_loss = jnp.sort(loss, axis=0)
        sorted_indices = jnp.argsort(loss, axis=0)
        empirical_cdf = jnp.argsort(sorted_indices, axis=0) / (len(loss) - 1)
        sorted_cdf = jnp.sort(empirical_cdf, axis=0).flatten()
        value_at_risk_idx = jnp.searchsorted(sorted_cdf, 1 - self.alpha, side='left')
        return value_at_risk_idx, sorted_loss[value_at_risk_idx]
    
    def forward(self, loss):
        multiplier = 1
        if self.inverted:
            loss *= -1
            multiplier = -1
            
        value_at_risk_idx, value_at_risk = self._value_at_risk(loss)
        num_elements_greater = len(loss) - value_at_risk_idx
        values_at_risk = jnp.where(loss >= value_at_risk, size=len(loss))[0]
        # print(num_elements_greater, values_at_risk)

        if self.reduction == 'mean':
            risk = multiplier * (jnp.sum(
                jnp.take(loss, values_at_risk, axis=0)) / num_elements_greater)
            loss *= multiplier # Undo modifier to loss passed in.
            return risk
        elif self.reduction == 'sum':
            risk = multiplier * jnp.sum(jnp.take(loss, values_at_risk, axis=0))
            loss *= multiplier
            return risk
        elif self.reduction == 'none':
            risk = values_at_risk
            loss *= multiplier
            return risk
        else:
            raise Exception('Only mean, sum, none reduction types supported.')