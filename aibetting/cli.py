"""
Command-line interface for AI betting system
"""
import argparse
import sys
from datetime import datetime
from aibetting.predictor import BettingPredictor
from aibetting.models import Match


def create_sample_match() -> Match:
    """Create a sample match for demonstration"""
    return Match(
        home_team="Manchester United",
        away_team="Chelsea",
        home_win_rate=0.65,
        away_win_rate=0.55,
        home_avg_goals=2.1,
        away_avg_goals=1.8,
        sport="soccer",
        date=datetime.now()
    )


def predict_command(args):
    """Handle predict command"""
    predictor = BettingPredictor()
    
    # Create match from arguments
    match = Match(
        home_team=args.home_team,
        away_team=args.away_team,
        home_win_rate=args.home_win_rate,
        away_win_rate=args.away_win_rate,
        home_avg_goals=args.home_avg_goals,
        away_avg_goals=args.away_avg_goals,
        sport=args.sport
    )
    
    # Make prediction
    prediction = predictor.predict(match)
    
    # Display results
    print("\n" + "="*60)
    print(prediction)
    print("="*60 + "\n")
    
    return 0


def demo_command(args):
    """Handle demo command"""
    print("\n" + "="*60)
    print("AI Betting System - Demo Mode")
    print("="*60 + "\n")
    
    predictor = BettingPredictor()
    
    # Create sample matches
    matches = [
        Match("Liverpool", "Arsenal", 0.70, 0.60, 2.3, 1.9),
        Match("Manchester City", "Tottenham", 0.75, 0.58, 2.5, 1.7),
        Match("Bayern Munich", "Dortmund", 0.72, 0.62, 2.4, 2.0),
    ]
    
    print("Sample Predictions:\n")
    
    for match in matches:
        prediction = predictor.predict(match)
        print(prediction)
        print("-" * 60 + "\n")
    
    print("\nNote: These are demonstration predictions using heuristic methods.")
    print("For better accuracy, train the model with historical data.\n")
    
    return 0


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="AI-powered betting prediction system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run demo with sample predictions
  aibetting demo
  
  # Predict a specific match
  aibetting predict --home-team "Liverpool" --away-team "Arsenal" \\
                    --home-win-rate 0.7 --away-win-rate 0.6 \\
                    --home-avg-goals 2.3 --away-avg-goals 1.9
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run demo with sample predictions")
    demo_parser.set_defaults(func=demo_command)
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict match outcome")
    predict_parser.add_argument("--home-team", required=True, help="Home team name")
    predict_parser.add_argument("--away-team", required=True, help="Away team name")
    predict_parser.add_argument("--home-win-rate", type=float, default=0.5,
                               help="Home team win rate (0-1, default: 0.5)")
    predict_parser.add_argument("--away-win-rate", type=float, default=0.5,
                               help="Away team win rate (0-1, default: 0.5)")
    predict_parser.add_argument("--home-avg-goals", type=float, default=1.5,
                               help="Home team average goals (default: 1.5)")
    predict_parser.add_argument("--away-avg-goals", type=float, default=1.5,
                               help="Away team average goals (default: 1.5)")
    predict_parser.add_argument("--sport", default="soccer",
                               help="Sport type (default: soccer)")
    predict_parser.set_defaults(func=predict_command)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
