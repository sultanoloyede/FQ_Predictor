# Standard library imports
import os
import sys
from datetime import datetime

# Third-party imports
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn imports
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score
)
from sklearn.metrics import (
    classification_report,
    roc_auc_score
)
from sklearn.linear_model import LogisticRegression

# Imbalanced-learn import
from imblearn.over_sampling import SMOTE

# =============================================================================
# Function Definitions
# =============================================================================

def get_most_recent_game(df: pd.DataFrame, team_abbr: str) -> pd.Series:
    """
    Retrieves the most recent game data for the specified team.

    Parameters:
        df (pd.DataFrame): DataFrame containing game data.
        team_abbr (str): Team abbreviation.

    Returns:
        pd.Series: The most recent game data for the team.
    """
    # Filter DataFrame for the specified team
    team_games = df[df['TEAM_ABBREVIATION'] == team_abbr].copy()

    # Check if any games exist for the team
    if team_games.empty:
        print(f"Warning: No games found for team abbreviation '{team_abbr}'.")
        return None

    # Ensure 'GAME_DATE' is datetime type
    if not np.issubdtype(team_games['GAME_DATE'].dtype, np.datetime64):
        team_games['GAME_DATE'] = pd.to_datetime(team_games['GAME_DATE'])

    # Sort games by date in descending order
    team_games = team_games.sort_values(by='GAME_DATE', ascending=False)

    # Return the most recent game
    return team_games.iloc[0]


def get_most_recent_head_to_head_game(
    df: pd.DataFrame, team_abbr: str, opponent_abbr: str
) -> pd.Series:
    """
    Retrieves the most recent head-to-head game data between two teams.

    Parameters:
        df (pd.DataFrame): DataFrame containing game data.
        team_abbr (str): Team abbreviation.
        opponent_abbr (str): Opponent team abbreviation.

    Returns:
        pd.Series: The most recent head-to-head game data between the teams.
    """
    # Filter for head-to-head games between the two teams
    head_to_head_games = df[
        (
            (df['TEAM_ABBREVIATION'] == team_abbr)
            & (df['OPPONENT_TEAM_ABBREVIATION'] == opponent_abbr)
        )
        | (
            (df['TEAM_ABBREVIATION'] == opponent_abbr)
            & (df['OPPONENT_TEAM_ABBREVIATION'] == team_abbr)
        )
    ].copy()

    # Check if any head-to-head games exist
    if head_to_head_games.empty:
        print(f"Warning: No head-to-head games found between '{team_abbr}' and '{opponent_abbr}'.")
        return None

    # Ensure 'GAME_DATE' is datetime type
    if not np.issubdtype(head_to_head_games['GAME_DATE'].dtype, np.datetime64):
        head_to_head_games['GAME_DATE'] = pd.to_datetime(head_to_head_games['GAME_DATE'])

    # Sort games by date in descending order
    head_to_head_games = head_to_head_games.sort_values(by='GAME_DATE', ascending=False)

    # Return the most recent head-to-head game
    return head_to_head_games.iloc[0]


def construct_new_row(df: pd.DataFrame, team_abbr: str, opponent_abbr: str) -> pd.DataFrame:
    """
    Constructs a new row for prediction based on the most recent games of the specified teams
    and their most recent head-to-head game.

    Parameters:
        df (pd.DataFrame): DataFrame containing game data.
        team_abbr (str): Team abbreviation.
        opponent_abbr (str): Opponent team abbreviation.

    Returns:
        pd.DataFrame: A DataFrame containing the new row.
    """
    # Retrieve recent games and head-to-head data
    team_recent_game = get_most_recent_game(df, team_abbr)
    opponent_recent_game = get_most_recent_game(df, opponent_abbr)
    head_to_head_game = get_most_recent_head_to_head_game(df, team_abbr, opponent_abbr)

    # Initialize new row with default values
    new_row = {
        'GAME_DATE': pd.Timestamp.today().strftime('%Y-%m-%d'),
        'TEAM_ABBREVIATION': team_abbr,
        'OPPONENT_TEAM_ABBREVIATION': opponent_abbr,
        'Total_First_Quarter_Points': 0,
        'MONTH': pd.Timestamp.today().month,
        'Home_Away': 'Home',  # Assuming the team is playing at home
        'Head_to_Head_Q1': 0,
        'Season_Avg_Pace_Team': 0,
        'Season_Avg_Pace_Opponent': 0,
        'Average_Season_Avg_Pace': 0,
        'Season_Avg_FG_PCT_Team': 0,
        'Season_Avg_FG_PCT_Opponent': 0,
        'PPG_Team': 0,
        'PPG_Opponent': 0,
        'Off_Rating_Avg': 0,
        'Opp_Off_Rating_Avg': 0,
        'Recent_Trend_Team': '[0,0,0]',
        'Recent_Trend_Opponent': '[0,0,0]',
    }

    # Update new row with team's recent game data
    if team_recent_game is not None:
        new_row['Season_Avg_Pace_Team'] = team_recent_game.get('Season_Avg_Pace_Team', 0)
        new_row['Season_Avg_FG_PCT_Team'] = team_recent_game.get('Season_Avg_FG_PCT_Team', 0)
        new_row['PPG_Team'] = team_recent_game.get('PPG_Team', 0)
        new_row['Off_Rating_Avg'] = team_recent_game.get('Off_Rating_Avg', 0)
        new_row['Recent_Trend_Team'] = team_recent_game.get('Recent_Trend_Team', '[0,0,0]')

    # Update new row with opponent's recent game data
    if opponent_recent_game is not None:
        # Note: We use 'Season_Avg_Pace_Team' from opponent's perspective
        new_row['Season_Avg_Pace_Opponent'] = opponent_recent_game.get('Season_Avg_Pace_Team', 0)
        new_row['Season_Avg_FG_PCT_Opponent'] = opponent_recent_game.get('Season_Avg_FG_PCT_Team', 0)
        new_row['PPG_Opponent'] = opponent_recent_game.get('PPG_Team', 0)
        new_row['Opp_Off_Rating_Avg'] = opponent_recent_game.get('Off_Rating_Avg', 0)
        new_row['Recent_Trend_Opponent'] = opponent_recent_game.get('Recent_Trend_Team', '[0,0,0]')

    # Update new row with head-to-head data
    if head_to_head_game is not None:
        new_row['Total_First_Quarter_Points'] = head_to_head_game.get('Total_First_Quarter_Points', 0)
        new_row['Head_to_Head_Q1'] = head_to_head_game.get('Head_to_Head_Q1', 0)
    else:
        # Use team's recent game data if no head-to-head data is available
        if team_recent_game is not None:
            new_row['Total_First_Quarter_Points'] = team_recent_game.get('Total_First_Quarter_Points', 0)
            new_row['Head_to_Head_Q1'] = team_recent_game.get('Head_to_Head_Q1', 0)

    # Calculate average season pace
    new_row['Average_Season_Avg_Pace'] = np.mean([
        new_row['Season_Avg_Pace_Team'],
        new_row['Season_Avg_Pace_Opponent']
    ])

    return pd.DataFrame([new_row])


def convert_trend(trend: str) -> list:
    """
    Converts a string representation of a trend into a list of integers.

    Parameters:
        trend (str): String representation of a list, e.g., '[1,2,3]'.

    Returns:
        list: List of integers extracted from the string.
    """
    try:
        # Remove brackets and split by commas
        return [int(i) for i in trend.strip('[]').split(',') if i.strip().isdigit()]
    except Exception:
        # Return empty list if conversion fails
        return []

# =============================================================================
# Main Execution
# =============================================================================

def main():
    # Load the dataset
    data_path = 'final_training_dataset.csv'
    df = pd.read_csv(data_path)

    print("\n--- Last Row of the Dataset Before Adding New Row ---")
    print(df.iloc[-1])

    # Conditional row addition based on command-line arguments
    if len(sys.argv) == 3:
        team_abbr = sys.argv[1]
        opponent_abbr = sys.argv[2]

        # Validate team abbreviations
        unique_teams = df['TEAM_ABBREVIATION'].unique()
        unique_opponents = df['OPPONENT_TEAM_ABBREVIATION'].unique()

        if team_abbr not in unique_teams:
            print(f"Error: TEAM_ABBREVIATION '{team_abbr}' not found in the dataset.")
            sys.exit(1)
        if opponent_abbr not in unique_opponents:
            print(f"Error: OPPONENT_TEAM_ABBREVIATION '{opponent_abbr}' not found in the dataset.")
            sys.exit(1)

        # Construct and append new row
        new_row_df = construct_new_row(df, team_abbr, opponent_abbr)
        df = pd.concat([df, new_row_df], ignore_index=True)

        print("\n--- Last Row of the Dataset After Adding New Row ---")
        print(df.iloc[-1])
    else:
        print("\n--- No New Row Added ---")

    # Extract game date and team abbreviations from the last row
    first_game_date = df.loc[df.index[-1], 'GAME_DATE']
    team1 = df.loc[df.index[-1], 'TEAM_ABBREVIATION']
    team2 = df.loc[df.index[-1], 'OPPONENT_TEAM_ABBREVIATION']

    # Data cleaning and feature engineering
    df['Recent_Trend_Team'] = df['Recent_Trend_Team'].apply(convert_trend)
    df['Recent_Trend_Opponent'] = df['Recent_Trend_Opponent'].apply(convert_trend)

    # Generate statistical features from recent trends
    for trend_col in ['Recent_Trend_Team', 'Recent_Trend_Opponent']:
        df[f'{trend_col}_Avg'] = df[trend_col].apply(lambda x: np.mean(x) if len(x) > 0 else 0)
        df[f'{trend_col}_STD'] = df[trend_col].apply(lambda x: np.std(x) if len(x) > 0 else 0)
        df[f'{trend_col}_Max'] = df[trend_col].apply(lambda x: np.max(x) if len(x) > 0 else 0)
        df[f'{trend_col}_Min'] = df[trend_col].apply(lambda x: np.min(x) if len(x) > 0 else 0)
        df[f'{trend_col}_Trend_Diff'] = df[trend_col].apply(lambda x: (x[-1] - x[0]) if len(x) > 1 else 0)

    # Drop original trend columns
    df = df.drop(columns=['Recent_Trend_Team', 'Recent_Trend_Opponent'])

    # Handle missing values using median imputation
    columns_with_missing = [
        'Head_to_Head_Q1', 'PTS', 'OPP_PTS', 'Off_Rating',
        'PPG_Team', 'PPG_Opponent', 'Off_Rating_Avg', 'Opp_Off_Rating_Avg'
    ]

    imputer = SimpleImputer(strategy='median')
    df[columns_with_missing] = imputer.fit_transform(df[columns_with_missing])

    # Remove duplicate rows
    initial_shape = df.shape
    df = df.drop_duplicates()
    final_shape = df.shape
    duplicates_removed = initial_shape[0] - final_shape[0]
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate rows from the dataset.")

    # Feature selection and encoding
    columns_to_keep = [
        'TEAM_ABBREVIATION', 'OPPONENT_TEAM_ABBREVIATION', 'Total_First_Quarter_Points',
        'MONTH', 'Home_Away', 'Head_to_Head_Q1', 'Season_Avg_Pace_Team',
        'Season_Avg_Pace_Opponent', 'Average_Season_Avg_Pace', 'Season_Avg_FG_PCT_Team',
        'Season_Avg_FG_PCT_Opponent', 'PPG_Team', 'PPG_Opponent', 'Off_Rating_Avg',
        'Opp_Off_Rating_Avg', 'Recent_Trend_Team_Avg', 'Recent_Trend_Team_STD',
        'Recent_Trend_Team_Max', 'Recent_Trend_Team_Min', 'Recent_Trend_Team_Trend_Diff',
        'Recent_Trend_Opponent_Avg', 'Recent_Trend_Opponent_STD', 'Recent_Trend_Opponent_Max',
        'Recent_Trend_Opponent_Min', 'Recent_Trend_Opponent_Trend_Diff'
    ]

    df = df[columns_to_keep]

    # Encode categorical variables
    le = LabelEncoder()
    categorical_columns = ['TEAM_ABBREVIATION', 'OPPONENT_TEAM_ABBREVIATION', 'Home_Away']

    for col in categorical_columns:
        df[col] = le.fit_transform(df[col])

    # Ensure 'Total_First_Quarter_Points' is numeric
    df['Total_First_Quarter_Points'] = pd.to_numeric(df['Total_First_Quarter_Points'], errors='coerce')
    df['Total_First_Quarter_Points'].fillna(0, inplace=True)

    # Define the target variable based on a threshold
    threshold = 50
    df['target_over_50'] = (df['Total_First_Quarter_Points'] >= threshold).astype(int)
    df = df.drop(columns=['Total_First_Quarter_Points'])

    # Save the LabelEncoder for future use
    joblib.dump(le, 'label_encoder.pkl')

    # Feature engineering refinement
    df['Pace_Interaction'] = df['Season_Avg_Pace_Team'] * df['Season_Avg_Pace_Opponent']
    df['PPG_Interaction'] = df['PPG_Team'] * df['PPG_Opponent']
    df['FG_PCT_Interaction'] = df['Season_Avg_FG_PCT_Team'] * df['Season_Avg_FG_PCT_Opponent']

    # Polynomial features
    df['PPG_Team_Squared'] = df['PPG_Team'] ** 2
    df['PPG_Opponent_Squared'] = df['PPG_Opponent'] ** 2
    df['Head_to_Head_Q1_Squared'] = df['Head_to_Head_Q1'] ** 2
    df['Pace_Interaction_Squared'] = df['Pace_Interaction'] ** 2
    df['FG_PCT_Team_Squared'] = df['Season_Avg_FG_PCT_Team'] ** 2
    df['FG_PCT_Opponent_Squared'] = df['Season_Avg_FG_PCT_Opponent'] ** 2

    # Difference features
    df['FG_PCT_Difference'] = df['Season_Avg_FG_PCT_Team'] - df['Season_Avg_FG_PCT_Opponent']
    df['Pace_Difference'] = df['Season_Avg_Pace_Team'] - df['Season_Avg_Pace_Opponent']
    df['Off_Rating_Avg_Difference'] = df['Off_Rating_Avg'] - df['Opp_Off_Rating_Avg']
    df['Off_Rating_Avg_Interaction'] = df['Off_Rating_Avg'] * df['Opp_Off_Rating_Avg']

    # Drop less important features
    df.drop(columns=['Average_Season_Avg_Pace'], inplace=True)

    # Feature scaling
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    numerical_columns.remove('target_over_50')
    numerical_columns = [col for col in numerical_columns if col not in categorical_columns]

    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    # Save the scaler for future use
    joblib.dump(scaler, 'scaler.pkl')

    # Separate features and target variable
    X = df.drop(columns=['target_over_50'])
    y = df['target_over_50'].astype(int)

    # Verify the target variable
    print("Data type of y:", y.dtype)
    print("Unique values in y:", y.unique())

    # Model development and cross-validation
    model_path = 'best_nba_model.pkl'
    scaler_path = 'scaler.pkl'

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        print("\nExisting model and scaler found. Loading them...")

        best_model = joblib.load(model_path)
        print("Best model loaded successfully.")

        loaded_scaler = joblib.load(scaler_path)
        print("Scaler loaded successfully.")
    else:
        print("\nNo existing model or scaler found. Proceeding with model training...")

        # Handle class imbalance with SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
        )

        print("\nTraining and testing sets created.")
        print("Training set shape:", X_train.shape)
        print("Testing set shape:", X_test.shape)

        # Logistic Regression Classifier
        print("\n--- Logistic Regression Classifier ---")
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)

        print("\nLogistic Regression Classification Report:")
        print(classification_report(y_test, y_pred_lr))
        print(f"Logistic Regression ROC-AUC Score: {roc_auc_score(y_test, y_pred_lr):.4f}")

        # Prediction for the first row of the test set
        first_test_instance = X_test.iloc[0].values.reshape(1, -1)
        first_prediction = lr.predict(first_test_instance)
        first_prediction_proba = lr.predict_proba(first_test_instance)

        print("\n--- Prediction for the First Row of the Test Set ---")
        print(f"Predicted Class: {'Over' if first_prediction[0] == 1 else 'Under'}")
        print(f"Predicted Probability: {first_prediction_proba[0][first_prediction[0]]:.4f}")
        print(f"Actual Class: {'Over' if y_test.iloc[0] == 1 else 'Under'}")

        # Cross-validation
        print("\n--- Cross-Validation: Logistic Regression Classifier ---")
        lr_cv = LogisticRegression(random_state=42, max_iter=1000)
        cv_folds = 5
        lr_cv_scores = cross_val_score(
            lr_cv, X_resampled, y_resampled, cv=cv_folds, scoring='roc_auc', n_jobs=-1
        )

        print(f"Logistic Regression Cross-Validation ROC-AUC Scores: {lr_cv_scores}")
        print(f"Mean ROC-AUC Score: {lr_cv_scores.mean():.4f}")
        print(f"Standard Deviation of ROC-AUC Score: {lr_cv_scores.std():.4f}")

        # Hyperparameter tuning
        print("\n--- Hyperparameter Tuning: Logistic Regression Classifier ---")
        lr_param_grid = {
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'C': [0.01, 0.1, 1, 10],
            'solver': ['lbfgs', 'liblinear', 'saga'],
            'class_weight': ['balanced', None],
            'max_iter': [1000]
        }

        lr_grid = LogisticRegression(random_state=42)
        lr_grid_search = GridSearchCV(
            lr_grid, lr_param_grid, cv=cv_folds, scoring='roc_auc', n_jobs=-1
        )
        lr_grid_search.fit(X_resampled, y_resampled)

        print(f"Best Parameters for Logistic Regression: {lr_grid_search.best_params_}")
        print(f"Best ROC-AUC Score from Grid Search for Logistic Regression: {lr_grid_search.best_score_:.4f}")

        # Use the best estimator
        best_model = lr_grid_search.best_estimator_

        # Save the best model
        joblib.dump(best_model, model_path)
        print("Best model saved successfully.")

        # Save the scaler
        joblib.dump(scaler, scaler_path)
        print("Scaler saved successfully.")

        # Prediction using the best model
        best_first_prediction = best_model.predict(first_test_instance)
        best_first_prediction_proba = best_model.predict_proba(first_test_instance)

        print("\n--- Best Model Prediction for the First Row of the Test Set ---")
        print(f"Predicted Class: {'Over' if best_first_prediction[0] == 1 else 'Under'}")
        print(f"Predicted Probability: {best_first_prediction_proba[0][best_first_prediction[0]]:.4f}")
        print(f"Actual Class: {'Over' if y_test.iloc[0] == 1 else 'Under'}")

    # Prediction using the best loaded model
    print("\n--- Making Prediction Using the Best Loaded Model ---")

    # Load the best model and scaler
    best_model = joblib.load(model_path)
    print("Best model loaded successfully.")

    scaler = joblib.load(scaler_path)
    print("Scaler loaded successfully.")

    # Prepare the last instance for prediction
    last_instance = X.iloc[-1].values.reshape(1, -1)
    loaded_prediction = best_model.predict(last_instance)
    loaded_prediction_proba = best_model.predict_proba(last_instance)

    print("\n--- Prediction for the Newly Added Row Using Loaded Best Model ---")
    print(f"GAME_DATE: {first_game_date}")
    print(f"TEAM1: {team1}")
    print(f"TEAM2: {team2}")
    print(f"Predicted Class: {'Over' if loaded_prediction[0] == 1 else 'Under'}")
    print(f"Predicted Probability: {loaded_prediction_proba[0][loaded_prediction[0]]:.4f}")

    # Display actual class if available
    actual_class = y.iloc[-1]
    print(f"Actual Class: {'Over' if actual_class == 1 else 'Under'}")


if __name__ == '__main__':
    main()
