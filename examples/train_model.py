"""
Example of training the AI model with historical data
"""
from aibetting import BettingPredictor, Match


def main():
    """Run training example"""
    print("AI Betting - Training Example\n")
    
    # Create historical matches with results
    print("Preparing historical match data...")
    historical_matches = [
        # Premier League matches
        Match("Liverpool", "Manchester City", home_score=3, away_score=1,
              home_win_rate=0.72, away_win_rate=0.75, 
              home_avg_goals=2.3, away_avg_goals=2.5),
        Match("Chelsea", "Arsenal", home_score=2, away_score=2,
              home_win_rate=0.65, away_win_rate=0.60,
              home_avg_goals=1.9, away_avg_goals=1.8),
        Match("Tottenham", "Manchester United", home_score=1, away_score=2,
              home_win_rate=0.58, away_win_rate=0.68,
              home_avg_goals=1.7, away_avg_goals=2.0),
        Match("Leicester", "West Ham", home_score=2, away_score=1,
              home_win_rate=0.55, away_win_rate=0.50,
              home_avg_goals=1.6, away_avg_goals=1.4),
        Match("Aston Villa", "Newcastle", home_score=1, away_score=1,
              home_win_rate=0.52, away_win_rate=0.53,
              home_avg_goals=1.5, away_avg_goals=1.5),
        Match("Brighton", "Wolves", home_score=3, away_score=0,
              home_win_rate=0.60, away_win_rate=0.48,
              home_avg_goals=1.8, away_avg_goals=1.3),
        Match("Everton", "Brentford", home_score=1, away_score=3,
              home_win_rate=0.45, away_win_rate=0.56,
              home_avg_goals=1.2, away_avg_goals=1.7),
        Match("Southampton", "Crystal Palace", home_score=0, away_score=2,
              home_win_rate=0.42, away_win_rate=0.51,
              home_avg_goals=1.1, away_avg_goals=1.4),
    ]
    
    print(f"Loaded {len(historical_matches)} historical matches\n")
    
    # Create and train predictor
    print("Training AI model...")
    predictor = BettingPredictor()
    predictor.train(historical_matches)
    print("Training complete!\n")
    
    # Now make predictions with the trained model
    print("Making predictions with trained model:\n")
    print("="*60)
    
    # Test matches
    test_matches = [
        Match("Liverpool", "Arsenal", 
              home_win_rate=0.70, away_win_rate=0.62,
              home_avg_goals=2.2, away_avg_goals=1.9),
        Match("Manchester City", "Chelsea",
              home_win_rate=0.75, away_win_rate=0.65,
              home_avg_goals=2.5, away_avg_goals=1.9),
    ]
    
    for match in test_matches:
        prediction = predictor.predict(match)
        print(prediction)
        print("-"*60 + "\n")
    
    print("Note: Predictions are now based on the trained ML model.")
    print("More training data will improve accuracy.")


if __name__ == "__main__":
    main()
