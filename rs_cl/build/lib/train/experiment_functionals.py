import jax.numpy as jnp

from risk_functionals.expected_value import ExpectedValue
from risk_functionals.cvar import CVaR
from risk_functionals.entropic_risk import EntropicRisk
from risk_functionals.human_aligned_risk import HumanAlignedRisk
from risk_functionals.mean_variance import MeanVariance
from risk_functionals.trimmed_risk import TrimmedRisk


risk_functionals = {
    # 'Human-Aligned Risk a=0.4, b=0.3': lambda losses: HumanAlignedRisk(a=0.4, b=0.3, reduction='none').forward(losses),
    'Expected Value': lambda losses: ExpectedValue(reduction='none').forward(losses),
    'CVaR 0.1': lambda losses: CVaR(alpha=0.1, inverted=False).forward(losses),
    # 'CVaR 0.2': lambda losses: CVaR(alpha=0.2, inverted=False).forward(losses),
    'CVaR 0.3': lambda losses: CVaR(alpha=0.3, inverted=False).forward(losses),
    # 'CVaR 0.4': lambda losses: CVaR(alpha=0.4, inverted=False).forward(losses),
    'CVaR 0.5': lambda losses: CVaR(alpha=0.5, inverted=False).forward(losses),
    # 'CVaR 0.6': lambda losses: CVaR(alpha=0.6, inverted=False).forward(losses),
    'CVaR 0.7': lambda losses: CVaR(alpha=0.7, inverted=False).forward(losses),
    # 'CVaR 0.8': lambda losses: CVaR(alpha=0.8, inverted=False).forward(losses),
    'CVaR 0.9': lambda losses: CVaR(alpha=0.9, inverted=False).forward(losses),
    'Inverted CVaR 0.1': lambda losses: CVaR(alpha=0.1, inverted=True).forward(losses),
    # 'Inverted CVaR 0.2': lambda losses: CVaR(alpha=0.2, inverted=True).forward(losses),
    'Inverted CVaR 0.3': lambda losses: CVaR(alpha=0.3, inverted=True).forward(losses),
    # 'Inverted CVaR 0.4': lambda losses: CVaR(alpha=0.4, inverted=True).forward(losses),
    'Inverted CVaR 0.5': lambda losses: CVaR(alpha=0.4, inverted=True).forward(losses),
    # 'Inverted CVaR 0.6': lambda losses: CVaR(alpha=0.6, inverted=True).forward(losses),
    'Inverted CVaR 0.7': lambda losses: CVaR(alpha=0.4, inverted=True).forward(losses),
    # 'Inverted CVaR 0.8': lambda losses: CVaR(alpha=0.8, inverted=True).forward(losses),
    'Inverted CVaR 0.9': lambda losses: CVaR(alpha=0.4, inverted=True).forward(losses),
    # 'Entropic Risk t=-3': lambda losses: EntropicRisk(t=-3).forward(losses),   
    'Entropic Risk t=-2': lambda losses: EntropicRisk(t=-2).forward(losses),
    # 'Entropic Risk t=-1.5': lambda losses: EntropicRisk(t=-1.5).forward(losses),
    'Entropic Risk t=-1': lambda losses: EntropicRisk(t=-1).forward(losses),
    'Entropic Risk t=-0.5': lambda losses: EntropicRisk(t=-0.5).forward(losses),
    # 'Entropic Risk t=-0.1': lambda losses: EntropicRisk(t=-0.1).forward(losses),
    # 'Entropic Risk t=0.1': lambda losses: EntropicRisk(t=0.1).forward(losses),
    'Entropic Risk t=0.5': lambda losses: EntropicRisk(t=0.5).forward(losses),
    'Entropic Risk t=1': lambda losses: EntropicRisk(t=1).forward(losses),
    # 'Entropic Risk t=1.5': lambda losses: EntropicRisk(t=1.5).forward(losses),
    'Entropic Risk t=2': lambda losses: EntropicRisk(t=2).forward(losses),
    # 'Entropic Risk t=3': lambda losses: EntropicRisk(t=3).forward(losses),
    'Human-Aligned Risk a=0.4, b=0.3': lambda losses: HumanAlignedRisk(a=0.4, b=0.3).forward(losses),
    # 'Human-Aligned Risk a=0.4, b=0.6': lambda losses: HumanAlignedRisk(a=0.4, b=0.6).forward(losses),
    # 'Human-Aligned Risk a=0.4, b=0.9': lambda losses: HumanAlignedRisk(a=0.4, b=0.9).forward(losses),
    # 'Human-Aligned Risk a=0.1, b=0.3': lambda losses: HumanAlignedRisk(a=0.1, b=0.3).forward(losses),
    # 'Human-Aligned Risk a=0.5, b=0.3': lambda losses: HumanAlignedRisk(a=0.5, b=0.3).forward(losses),
    'Human-Aligned Risk a=0.7, b=0.3': lambda losses: HumanAlignedRisk(a=0.7, b=0.3).forward(losses),
    # 'Human-Aligned Risk a=0.7, b=0.7': lambda losses: HumanAlignedRisk(a=0.7, b=0.7).forward(losses),
    'Human-Aligned Risk a=0.2, b=0.2': lambda losses: HumanAlignedRisk(a=0.2, b=0.2).forward(losses),
    'Mean Variance c=-5': lambda losses: MeanVariance(c=-5).forward(losses),
    'Mean Variance c=-1': lambda losses: MeanVariance(c=-1).forward(losses),
    'Mean Variance c=-0.1': lambda losses: MeanVariance(c=-0.1).forward(losses),
    'Mean Variance c=0.1': lambda losses: MeanVariance(c=0.1).forward(losses),
    # 'Mean Variance c=0.5': lambda losses: MeanVariance(c=0.5).forward(losses),
    'Mean Variance c=1.0': lambda losses: MeanVariance(c=1.0).forward(losses),
    'Mean Variance c=5.0': lambda losses: MeanVariance(c=5.0).forward(losses),
    # 'Mean Variance c=10.0': lambda losses: MeanVariance(c=10.0).forward(losses),
    # 'Mean Variance c=20.0': lambda losses: MeanVariance(c=20.0).forward(losses),
    # 'Mean Variance c=50.0': lambda losses: MeanVariance(c=50.0).forward(losses),
    # 'Trimmed Risk 0.05': lambda losses: TrimmedRisk(alpha=0.05).forward(losses),
    'Trimmed Risk 0.1': lambda losses: TrimmedRisk(alpha=0.1).forward(losses),
    # 'Trimmed Risk 0.15': lambda losses: TrimmedRisk(alpha=0.15).forward(losses),
    'Trimmed Risk 0.2': lambda losses: TrimmedRisk(alpha=0.2).forward(losses),
    # 'Trimmed Risk 0.25': lambda losses: TrimmedRisk(alpha=0.25).forward(losses),
    'Trimmed Risk 0.3': lambda losses: TrimmedRisk(alpha=0.3).forward(losses),
    # 'Trimmed Risk 0.35': lambda losses: TrimmedRisk(alpha=0.35).forward(losses),
    'Trimmed Risk 0.4': lambda losses: TrimmedRisk(alpha=0.4).forward(losses),
    # 'Trimmed Risk 0.45': lambda losses: TrimmedRisk(alpha=0.45).forward(losses),
}