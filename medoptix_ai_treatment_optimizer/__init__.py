"""
MedOptix AI Treatment Optimizer
A system for predicting patient dropout risk and forecasting treatment adherence.
"""

from .app import app
from .train_models import train_models

__all__ = ['app', 'train_models'] 