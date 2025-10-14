"""
Times Network CTR Optimizer
Enterprise-grade CTR prediction system

Author: Prateek (IIT Patna MTech AI)
Email: prat.cann.170701@gmail.com
"""

__version__ = "1.0.0"
__author__ = "Prateek"
__email__ = "prat.cann.170701@gmail.com"

from .predictor import CTRPredictor
from .models.wide_deep import WideDeepModel

__all__ = ["CTRPredictor", "WideDeepModel"]
