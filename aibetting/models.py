"""
Data models for the AI betting system
"""
from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class Match:
    """Represents a sports match for betting prediction"""
    
    home_team: str
    away_team: str
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    date: Optional[datetime] = None
    sport: str = "soccer"
    
    # Team statistics
    home_win_rate: float = 0.5
    away_win_rate: float = 0.5
    home_avg_goals: float = 1.5
    away_avg_goals: float = 1.5
    
    def __post_init__(self):
        """Validate match data"""
        if self.home_win_rate < 0 or self.home_win_rate > 1:
            raise ValueError("Win rate must be between 0 and 1")
        if self.away_win_rate < 0 or self.away_win_rate > 1:
            raise ValueError("Win rate must be between 0 and 1")


@dataclass
class Prediction:
    """Represents a betting prediction"""
    
    match: Match
    home_win_probability: float
    away_win_probability: float
    draw_probability: float
    recommended_bet: str
    confidence: float
    expected_value: float = 0.0
    
    def __post_init__(self):
        """Validate prediction data"""
        total_prob = self.home_win_probability + self.away_win_probability + self.draw_probability
        if not (0.99 <= total_prob <= 1.01):  # Allow small floating point errors
            raise ValueError(f"Probabilities must sum to 1.0, got {total_prob}")
        
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError("Confidence must be between 0 and 1")
    
    def __str__(self):
        """String representation of prediction"""
        return (
            f"Prediction for {self.match.home_team} vs {self.match.away_team}:\n"
            f"  Home Win: {self.home_win_probability:.2%}\n"
            f"  Away Win: {self.away_win_probability:.2%}\n"
            f"  Draw: {self.draw_probability:.2%}\n"
            f"  Recommended: {self.recommended_bet}\n"
            f"  Confidence: {self.confidence:.2%}"
        )
