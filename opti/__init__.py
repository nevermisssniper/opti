"""OPTI: Multi-Domain Performance Optimizer.

A linear regression system that models cross-domain performance interactions
(mental, physical, cognitive) and recommends minimum behavioral changes to
maximize predicted next-day performance.
"""
from opti.data import SyntheticDataConfig, generate_synthetic_logs
from opti.features import FeatureBuilder
from opti.model import OptiModel
from opti.optimizer import BehaviorOptimizer, OptimizationResult

__version__ = "0.1.0"
__all__ = [
    "SyntheticDataConfig",
    "generate_synthetic_logs",
    "FeatureBuilder",
    "OptiModel",
    "BehaviorOptimizer",
    "OptimizationResult",
]
