"""
AI Betting - AI-powered betting prediction system
"""

__version__ = "0.1.0"
__author__ = "Andy Cheng"

from aibetting.predictor import BettingPredictor
from aibetting.models import Match, Prediction

__all__ = ["BettingPredictor", "Match", "Prediction"]
