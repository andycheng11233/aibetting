"""
AI-powered betting prediction engine
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import List, Tuple
from aibetting.models import Match, Prediction

# Constants
MIN_TRAINING_SAMPLES = 5
DEFAULT_N_ESTIMATORS = 100
DEFAULT_MAX_DEPTH = 10
RANDOM_STATE = 42


class BettingPredictor:
    """
    AI-powered betting predictor using machine learning
    """
    
    def __init__(self, n_estimators: int = DEFAULT_N_ESTIMATORS, 
                 max_depth: int = DEFAULT_MAX_DEPTH,
                 random_state: int = RANDOM_STATE):
        """
        Initialize the betting predictor
        
        Args:
            n_estimators: Number of trees in random forest (default: 100)
            max_depth: Maximum depth of trees (default: 10)
            random_state: Random state for reproducibility (default: 42)
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
        self.is_trained = False
    
    def _extract_features(self, match: Match) -> np.ndarray:
        """
        Extract features from a match for prediction
        
        Args:
            match: Match object
            
        Returns:
            Feature array
        """
        features = [
            match.home_win_rate,
            match.away_win_rate,
            match.home_avg_goals,
            match.away_avg_goals,
            match.home_win_rate - match.away_win_rate,  # Win rate difference
            match.home_avg_goals - match.away_avg_goals,  # Goal difference
        ]
        return np.array(features).reshape(1, -1)
    
    def train(self, matches: List[Match]) -> None:
        """
        Train the predictor on historical match data
        
        Args:
            matches: List of historical matches with results
        """
        if not matches:
            raise ValueError("Training data cannot be empty")
        
        # Prepare training data
        X = []
        y = []
        
        for match in matches:
            if match.home_score is None or match.away_score is None:
                continue
            
            features = self._extract_features(match)[0]
            X.append(features)
            
            # Determine outcome (0: away win, 1: draw, 2: home win)
            if match.home_score > match.away_score:
                y.append(2)
            elif match.home_score < match.away_score:
                y.append(0)
            else:
                y.append(1)
        
        if len(X) < MIN_TRAINING_SAMPLES:
            raise ValueError(f"Need at least {MIN_TRAINING_SAMPLES} matches with results for training")
        
        X = np.array(X)
        y = np.array(y)
        
        # Train random forest classifier
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        self.model.fit(X, y)
        self.is_trained = True
    
    def predict(self, match: Match) -> Prediction:
        """
        Predict the outcome of a match
        
        Args:
            match: Match to predict
            
        Returns:
            Prediction object with probabilities and recommendation
        """
        features = self._extract_features(match)
        
        if self.is_trained and self.model is not None:
            # Use trained model
            probabilities = self.model.predict_proba(features)[0]
            away_win_prob = probabilities[0]
            draw_prob = probabilities[1]
            home_win_prob = probabilities[2]
        else:
            # Use simple heuristic if not trained
            home_win_prob, draw_prob, away_win_prob = self._simple_prediction(match)
        
        # Determine recommended bet
        probs = {
            'home': home_win_prob,
            'draw': draw_prob,
            'away': away_win_prob
        }
        recommended = max(probs, key=probs.get)
        confidence = max(probs.values())
        
        return Prediction(
            match=match,
            home_win_probability=float(home_win_prob),
            away_win_probability=float(away_win_prob),
            draw_probability=float(draw_prob),
            recommended_bet=recommended,
            confidence=float(confidence)
        )
    
    def _simple_prediction(self, match: Match) -> Tuple[float, float, float]:
        """
        Simple heuristic-based prediction when model is not trained
        
        Args:
            match: Match to predict
            
        Returns:
            Tuple of (home_win_prob, draw_prob, away_win_prob)
        """
        # Base probabilities
        home_strength = (match.home_win_rate + match.home_avg_goals / 3.0) / 2.0
        away_strength = (match.away_win_rate + match.away_avg_goals / 3.0) / 2.0
        
        # Normalize to probabilities
        total = home_strength + away_strength + 0.25  # 0.25 for draw baseline
        home_win_prob = home_strength / total
        away_win_prob = away_strength / total
        draw_prob = 0.25 / total
        
        # Normalize to sum to 1.0
        total_prob = home_win_prob + away_win_prob + draw_prob
        home_win_prob /= total_prob
        away_win_prob /= total_prob
        draw_prob /= total_prob
        
        return home_win_prob, draw_prob, away_win_prob
    
    def predict_batch(self, matches: List[Match]) -> List[Prediction]:
        """
        Predict outcomes for multiple matches
        
        Args:
            matches: List of matches to predict
            
        Returns:
            List of predictions
        """
        return [self.predict(match) for match in matches]
