import sys
import pandas as pd
import joblib
import numpy as np

if len(sys.argv) != 3:
    print("Usage: python3 ml.py <TEAM_ABBREV_1> <TEAM_ABBREV_2>")
    sys.exit(1)

team1 = sys.argv[1]
team2 = sys.argv[2]

# Load saved artifacts
model = joblib.load('best_logistic_regression_model.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')
metadata = joblib.load('metadata.pkl')

features_to_use = metadata['features_to_use']
categorical_features = metadata['categorical_features']
numerical_features = metadata['numerical_features']
encoder_feature_names = metadata['encoder_feature_names']

# Load the dataset
df = pd.read_csv('pace.csv')

# We need to find the most recent game involving team1 and team2.
# Check if there's a direct match where TEAM_ABBREVIATION=team1 and OPPONENT_TEAM_ABBREVIATION=team2
df_filtered = df[(df['TEAM_ABBREVIATION'] == team1) & (df['OPPONENT_TEAM_ABBREVIATION'] == team2)]

# If no direct match found, try the reverse
if df_filtered.empty:
    df_filtered = df[(df['TEAM_ABBREVIATION'] == team2) & (df['OPPONENT_TEAM_ABBREVIATION'] == team1)]

if df_filtered.empty:
    print(f"No recent game found between {team1} and {team2}.")
    sys.exit(1)

# Find the most recent game (max GAME_ID)
most_recent_game = df_filtered.loc[df_filtered['GAME_ID'].idxmax()]

# We'll now create a single-row DataFrame with the features needed
# Our model expects the same preprocessing: scaled numeric features and one-hot encoded categorical features
data_row = most_recent_game[features_to_use].to_frame().T  # Convert Series to DataFrame

# Separate numerical and categorical features
data_num = data_row[numerical_features]
data_cat = data_row[categorical_features]

# Scale numerical features using the saved scaler
data_num_scaled = scaler.transform(data_num)

# One-hot encode categorical features using the saved encoder
data_cat_encoded = encoder.transform(data_cat).toarray()

# Combine scaled numeric and encoded categorical
data_preprocessed = np.hstack([data_num_scaled, data_cat_encoded])

# Predict using the model
y_pred = model.predict(data_preprocessed)
y_proba = model.predict_proba(data_preprocessed)

print(f"Matchup: {team1} vs {team2}")
print(f"Predicted class: {y_pred[0]}")
# The order of classes from the model
classes = model.classes_
proba_str = ", ".join([f"{cls}: {prob:.4f}" for cls, prob in zip(classes, y_proba[0])])
print(f"Probabilities: {proba_str}")
