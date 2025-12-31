# AI Betting

An AI-powered betting prediction system that uses machine learning to predict sports match outcomes and provide betting recommendations.

## Features

- 🤖 **AI-Powered Predictions**: Uses machine learning algorithms to predict match outcomes
- 📊 **Statistical Analysis**: Analyzes team statistics, win rates, and scoring patterns
- 🎯 **Confidence Scoring**: Provides confidence levels for each prediction
- 🔄 **Batch Processing**: Predict multiple matches at once
- 💻 **CLI Interface**: Easy-to-use command-line interface
- 🧪 **Demo Mode**: Try the system with sample data

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/andycheng11233/aibetting.git
cd aibetting

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Requirements

- Python 3.8+
- NumPy
- scikit-learn
- pandas

## Usage

### Demo Mode

Run the demo to see sample predictions:

```bash
aibetting demo
```

### Predict a Match

Predict the outcome of a specific match:

```bash
aibetting predict \
  --home-team "Liverpool" \
  --away-team "Arsenal" \
  --home-win-rate 0.7 \
  --away-win-rate 0.6 \
  --home-avg-goals 2.3 \
  --away-avg-goals 1.9
```

### Python API

Use the betting predictor in your Python code:

```python
from aibetting import BettingPredictor, Match

# Create a predictor
predictor = BettingPredictor()

# Create a match
match = Match(
    home_team="Manchester United",
    away_team="Chelsea",
    home_win_rate=0.65,
    away_win_rate=0.55,
    home_avg_goals=2.1,
    away_avg_goals=1.8
)

# Get prediction
prediction = predictor.predict(match)

print(prediction)
# Output:
# Prediction for Manchester United vs Chelsea:
#   Home Win: 52.34%
#   Away Win: 25.67%
#   Draw: 21.99%
#   Recommended: home
#   Confidence: 52.34%
```

### Training with Historical Data

For better accuracy, train the model with historical match data:

```python
from aibetting import BettingPredictor, Match

# Create predictor
predictor = BettingPredictor()

# Prepare historical matches with results
historical_matches = [
    Match("Team A", "Team B", home_score=2, away_score=1, 
          home_win_rate=0.6, away_win_rate=0.5, 
          home_avg_goals=1.8, away_avg_goals=1.5),
    Match("Team C", "Team D", home_score=1, away_score=1,
          home_win_rate=0.55, away_win_rate=0.55,
          home_avg_goals=1.6, away_avg_goals=1.6),
    # Add more historical matches...
]

# Train the model
predictor.train(historical_matches)

# Now predictions will use the trained model
prediction = predictor.predict(new_match)
```

## How It Works

The AI betting system uses multiple approaches:

1. **Feature Extraction**: Extracts relevant features from match data including:
   - Team win rates
   - Average goals scored
   - Historical performance metrics
   - Head-to-head statistics

2. **Machine Learning**: Uses Random Forest Classifier when trained with historical data for more accurate predictions

3. **Heuristic Fallback**: When no training data is available, uses statistical heuristics based on team strength indicators

4. **Confidence Scoring**: Provides confidence levels to help assess prediction reliability

## Data Models

### Match

Represents a sports match with team statistics:

- `home_team`: Name of the home team
- `away_team`: Name of the away team
- `home_score`: Actual home team score (for historical data)
- `away_score`: Actual away team score (for historical data)
- `home_win_rate`: Historical win rate (0-1)
- `away_win_rate`: Historical win rate (0-1)
- `home_avg_goals`: Average goals scored per match
- `away_avg_goals`: Average goals scored per match
- `sport`: Type of sport (default: soccer)

### Prediction

Contains the prediction results:

- `home_win_probability`: Probability of home team winning
- `away_win_probability`: Probability of away team winning
- `draw_probability`: Probability of a draw
- `recommended_bet`: Recommended betting choice
- `confidence`: Confidence level of the prediction

## Development

### Project Structure

```
aibetting/
├── aibetting/
│   ├── __init__.py      # Package initialization
│   ├── models.py        # Data models
│   ├── predictor.py     # AI prediction engine
│   └── cli.py          # Command-line interface
├── requirements.txt     # Dependencies
├── setup.py            # Package setup
└── README.md           # Documentation
```

## License

This project is open source and available for educational purposes.

## Disclaimer

⚠️ **Important**: This tool is for educational and research purposes only. Sports betting involves financial risk. This system does not guarantee accurate predictions or profitable outcomes. Always bet responsibly and within your means.
