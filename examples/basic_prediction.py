"""
Basic example of using the AI betting predictor
"""
from aibetting import BettingPredictor, Match


def main():
    """Run basic prediction example"""
    print("AI Betting - Basic Example\n")
    
    # Create a predictor
    predictor = BettingPredictor()
    
    # Create a match to predict
    match = Match(
        home_team="Manchester United",
        away_team="Chelsea",
        home_win_rate=0.65,
        away_win_rate=0.55,
        home_avg_goals=2.1,
        away_avg_goals=1.8,
        sport="soccer"
    )
    
    # Get prediction
    print("Making prediction...\n")
    prediction = predictor.predict(match)
    
    # Display results
    print(prediction)
    print("\n" + "="*60)
    print("Analysis:")
    print(f"  - The model recommends betting on: {prediction.recommended_bet}")
    print(f"  - Confidence level: {prediction.confidence:.1%}")
    
    if prediction.confidence > 0.6:
        print(f"  - This is a HIGH confidence prediction")
    elif prediction.confidence > 0.4:
        print(f"  - This is a MEDIUM confidence prediction")
    else:
        print(f"  - This is a LOW confidence prediction")
    
    print("="*60)


if __name__ == "__main__":
    main()
