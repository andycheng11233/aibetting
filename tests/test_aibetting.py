"""
Unit tests for AI betting system
"""
import unittest
from aibetting.models import Match, Prediction
from aibetting.predictor import BettingPredictor


class TestMatch(unittest.TestCase):
    """Test Match model"""
    
    def test_match_creation(self):
        """Test creating a match"""
        match = Match(
            home_team="Team A",
            away_team="Team B",
            home_win_rate=0.6,
            away_win_rate=0.5
        )
        self.assertEqual(match.home_team, "Team A")
        self.assertEqual(match.away_team, "Team B")
        self.assertEqual(match.home_win_rate, 0.6)
        self.assertEqual(match.away_win_rate, 0.5)
    
    def test_match_validation(self):
        """Test match validation"""
        with self.assertRaises(ValueError):
            Match("Team A", "Team B", home_win_rate=1.5)  # Invalid rate
        
        with self.assertRaises(ValueError):
            Match("Team A", "Team B", away_win_rate=-0.1)  # Invalid rate


class TestPrediction(unittest.TestCase):
    """Test Prediction model"""
    
    def test_prediction_creation(self):
        """Test creating a prediction"""
        match = Match("Team A", "Team B")
        prediction = Prediction(
            match=match,
            home_win_probability=0.5,
            away_win_probability=0.3,
            draw_probability=0.2,
            recommended_bet="home",
            confidence=0.5
        )
        self.assertEqual(prediction.recommended_bet, "home")
        self.assertEqual(prediction.confidence, 0.5)
    
    def test_prediction_validation(self):
        """Test prediction probability validation"""
        match = Match("Team A", "Team B")
        
        # Probabilities don't sum to 1
        with self.assertRaises(ValueError):
            Prediction(
                match=match,
                home_win_probability=0.5,
                away_win_probability=0.5,
                draw_probability=0.5,
                recommended_bet="home",
                confidence=0.5
            )


class TestBettingPredictor(unittest.TestCase):
    """Test BettingPredictor"""
    
    def test_predictor_creation(self):
        """Test creating a predictor"""
        predictor = BettingPredictor()
        self.assertFalse(predictor.is_trained)
        self.assertIsNone(predictor.model)
    
    def test_simple_prediction(self):
        """Test making a prediction without training"""
        predictor = BettingPredictor()
        match = Match(
            home_team="Team A",
            away_team="Team B",
            home_win_rate=0.7,
            away_win_rate=0.5
        )
        
        prediction = predictor.predict(match)
        
        self.assertIsInstance(prediction, Prediction)
        self.assertGreater(prediction.home_win_probability, 0)
        self.assertGreater(prediction.away_win_probability, 0)
        self.assertGreater(prediction.draw_probability, 0)
        
        # Check probabilities sum to approximately 1
        total = (prediction.home_win_probability + 
                prediction.away_win_probability + 
                prediction.draw_probability)
        self.assertAlmostEqual(total, 1.0, places=5)
    
    def test_training(self):
        """Test training the predictor"""
        predictor = BettingPredictor()
        
        # Create training data
        matches = [
            Match("A", "B", home_score=2, away_score=1, 
                  home_win_rate=0.6, away_win_rate=0.5,
                  home_avg_goals=1.8, away_avg_goals=1.5),
            Match("C", "D", home_score=1, away_score=1,
                  home_win_rate=0.55, away_win_rate=0.55,
                  home_avg_goals=1.6, away_avg_goals=1.6),
            Match("E", "F", home_score=0, away_score=2,
                  home_win_rate=0.45, away_win_rate=0.65,
                  home_avg_goals=1.2, away_avg_goals=2.0),
            Match("G", "H", home_score=3, away_score=0,
                  home_win_rate=0.7, away_win_rate=0.4,
                  home_avg_goals=2.2, away_avg_goals=1.1),
            Match("I", "J", home_score=1, away_score=3,
                  home_win_rate=0.5, away_win_rate=0.6,
                  home_avg_goals=1.5, away_avg_goals=1.9),
        ]
        
        predictor.train(matches)
        
        self.assertTrue(predictor.is_trained)
        self.assertIsNotNone(predictor.model)
    
    def test_batch_prediction(self):
        """Test predicting multiple matches"""
        predictor = BettingPredictor()
        
        matches = [
            Match("A", "B", home_win_rate=0.6, away_win_rate=0.5),
            Match("C", "D", home_win_rate=0.7, away_win_rate=0.4),
        ]
        
        predictions = predictor.predict_batch(matches)
        
        self.assertEqual(len(predictions), 2)
        self.assertIsInstance(predictions[0], Prediction)
        self.assertIsInstance(predictions[1], Prediction)


if __name__ == "__main__":
    unittest.main()
