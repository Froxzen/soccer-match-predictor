# ⚽ Premier League Soccer Match Predictor 

This project involves predicting the outcomes of Premier League soccer matches using a **Random Forest Classifier**. The dataset used for training and testing the model contains 1390 entries. The main steps include data preprocessing, feature engineering, training the model, and making predictions.

## 🛠️ Feature Engineering 

Several new features are engineered to improve the model's predictive power:
- `venue_code`: Encodes the match venue.
- `opp_code`: Encodes the opponent team.
- `hour`: Extracts the hour from the match time.
- `day_code`: Extracts the day of the week from the match date.
- `target`: Binary feature indicating whether the match was won (1) or not (0).

## 💻 Languages and Libraries Used

The project primarily uses the following languages and libraries:
- **Python**: For data preprocessing, feature engineering, model training, and evaluation.
- **Pandas**: For data manipulation and analysis.
- **Scikit-learn**: For implementing the Random Forest Classifier and evaluating the model.

## ✏️ Contributing
If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.
