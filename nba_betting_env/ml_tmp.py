import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from datetime import datetime

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Function to retrieve the most recent game data for the specified team.
def get_most_recent_game(df, team_abbr):
    """
    Retrieves the most recent game data for the specified team.
    """
    team_games = df[df['TEAM_ABBREVIATION'] == team_abbr].copy()
    if team_games.empty:
        print(f"Warning: No games found for team abbreviation '{team_abbr}'.")
        return None
    if not np.issubdtype(team_games['GAME_DATE'].dtype, np.datetime64):
        team_games['GAME_DATE'] = pd.to_datetime(team_games['GAME_DATE'])
    team_games = team_games.sort_values(by='GAME_DATE', ascending=False)
    return team_games.iloc[0]

# Function to retrieve the most recent head-to-head game data between two teams.
def get_most_recent_head_to_head_game(df, team_abbr, opponent_abbr):
    """
    Retrieves the most recent head-to-head game data between two teams.
    """
    head_to_head_games = df[
        ((df['TEAM_ABBREVIATION'] == team_abbr) & (df['OPPONENT_TEAM_ABBREVIATION'] == opponent_abbr)) |
        ((df['TEAM_ABBREVIATION'] == opponent_abbr) & (df['OPPONENT_TEAM_ABBREVIATION'] == team_abbr))
    ].copy()
    
    if head_to_head_games.empty:
        print(f"Warning: No head-to-head games found between '{team_abbr}' and '{opponent_abbr}'.")
        return None
    
    if not np.issubdtype(head_to_head_games['GAME_DATE'].dtype, np.datetime64):
        head_to_head_games['GAME_DATE'] = pd.to_datetime(head_to_head_games['GAME_DATE'])
    
    head_to_head_games = head_to_head_games.sort_values(by='GAME_DATE', ascending=False)
    
    return head_to_head_games.iloc[0]

# Function to construct a new row based on recent games and head-to-head data.
def construct_new_row(df, team_abbr, opponent_abbr):
    """
    Constructs a new row for the DataFrame based on the most recent games of the specified teams
    and the most recent head-to-head game between them.
    """
    team_recent_game = get_most_recent_game(df, team_abbr)
    opponent_recent_game = get_most_recent_game(df, opponent_abbr)
    head_to_head_game = get_most_recent_head_to_head_game(df, team_abbr, opponent_abbr)
    
    new_row = {
        'GAME_DATE': pd.Timestamp.today().strftime('%Y-%m-%d'),
        'TEAM_ABBREVIATION': team_abbr,
        'OPPONENT_TEAM_ABBREVIATION': opponent_abbr,
        'Total_First_Quarter_Points': 0,
        'MONTH': pd.Timestamp.today().month,
        'Home_Away': 'Home',
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
    
    if team_recent_game is not None:
        new_row['Season_Avg_Pace_Team'] = team_recent_game.get('Season_Avg_Pace_Team', 0)
        new_row['Season_Avg_FG_PCT_Team'] = team_recent_game.get('Season_Avg_FG_PCT_Team', 0)
        new_row['PPG_Team'] = team_recent_game.get('PPG_Team', 0)
        new_row['Off_Rating_Avg'] = team_recent_game.get('Off_Rating_Avg', 0)
        new_row['Recent_Trend_Team'] = team_recent_game.get('Recent_Trend_Team', '[0,0,0]')
    
    if opponent_recent_game is not None:
        new_row['Season_Avg_Pace_Opponent'] = opponent_recent_game.get('Season_Avg_Pace_Team', 0)
        new_row['Season_Avg_FG_PCT_Opponent'] = opponent_recent_game.get('Season_Avg_FG_PCT_Team', 0)
        new_row['PPG_Opponent'] = opponent_recent_game.get('PPG_Team', 0)
        new_row['Recent_Trend_Opponent'] = opponent_recent_game.get('Recent_Trend_Team', '[0,0,0]')
        new_row['Opp_Off_Rating_Avg'] = opponent_recent_game.get('Off_Rating_Avg', 0)
    
    if head_to_head_game is not None:
        new_row['Total_First_Quarter_Points'] = head_to_head_game.get('Total_First_Quarter_Points', 0)
        new_row['Head_to_Head_Q1'] = head_to_head_game.get('Head_to_Head_Q1', 0)
    else:
        if team_recent_game is not None:
            new_row['Total_First_Quarter_Points'] = team_recent_game.get('Total_First_Quarter_Points', 0)
            new_row['Head_to_Head_Q1'] = team_recent_game.get('Head_to_Head_Q1', 0)
    
    new_row['Average_Season_Avg_Pace'] = np.mean([
        new_row['Season_Avg_Pace_Team'],
        new_row['Season_Avg_Pace_Opponent']
    ])
    
    return pd.DataFrame([new_row])

# Function to convert trend strings to lists of integers.
def convert_trend(trend):
    """
    Converts a string representation of a trend to a list of integers.
    """
    try:
        return [int(i) for i in trend.strip('[]').split(',') if i.strip().isdigit()]
    except Exception as e:
        return []

# === Main Script Execution ===

# Load the dataset.
data_path = 'final_training_dataset.csv'
df = pd.read_csv(data_path)

print("\n--- Last Row of the Dataset Before Adding New Row ---")
print(df.iloc[-1])

# === Conditional Row Addition ===
if len(sys.argv) == 3:
    team_abbr = sys.argv[1]
    opponent_abbr = sys.argv[2]
    
    unique_teams = df['TEAM_ABBREVIATION'].unique()
    unique_opponents = df['OPPONENT_TEAM_ABBREVIATION'].unique()
    
    if team_abbr not in unique_teams:
        print(f"Error: TEAM_ABBREVIATION '{team_abbr}' not found in the dataset.")
        sys.exit(1)
    if opponent_abbr not in unique_opponents:
        print(f"Error: OPPONENT_TEAM_ABBREVIATION '{opponent_abbr}' not found in the dataset.")
        sys.exit(1)
    
    new_row_df = construct_new_row(df, team_abbr, opponent_abbr)
    df = pd.concat([df, new_row_df], ignore_index=True)
    
    print("\n--- Last Row of the Dataset After Adding New Row ---")
    print(df.iloc[-1])
else:
    print("\n--- No New Row Added ---")

# Extract game date and team abbreviations from the last row.
first_game_date = df.loc[df.index[-1], 'GAME_DATE']
team1 = df.loc[df.index[-1], 'TEAM_ABBREVIATION']
team2 = df.loc[df.index[-1], 'OPPONENT_TEAM_ABBREVIATION']

# === Data Cleaning and Recent Trend Feature Engineering ===
df['Recent_Trend_Team'] = df['Recent_Trend_Team'].apply(convert_trend)
df['Recent_Trend_Opponent'] = df['Recent_Trend_Opponent'].apply(convert_trend)

for trend_col in ['Recent_Trend_Team', 'Recent_Trend_Opponent']:
    df[f'{trend_col}_Avg'] = df[trend_col].apply(lambda x: np.mean(x) if len(x) > 0 else 0)
    df[f'{trend_col}_STD'] = df[trend_col].apply(lambda x: np.std(x) if len(x) > 0 else 0)
    df[f'{trend_col}_Max'] = df[trend_col].apply(lambda x: np.max(x) if len(x) > 0 else 0)
    df[f'{trend_col}_Min'] = df[trend_col].apply(lambda x: np.min(x) if len(x) > 0 else 0)
    df[f'{trend_col}_Trend_Diff'] = df[trend_col].apply(lambda x: (x[-1] - x[0]) if len(x) > 1 else 0)

df = df.drop(columns=['Recent_Trend_Team', 'Recent_Trend_Opponent'])

# Handle missing values with imputation.
columns_with_missing = [
    'Head_to_Head_Q1', 'PTS', 'OPP_PTS', 'Off_Rating', 
    'PPG_Team', 'PPG_Opponent', 'Off_Rating_Avg', 'Opp_Off_Rating_Avg'
]

imputer = SimpleImputer(strategy='median')
df[columns_with_missing] = imputer.fit_transform(df[columns_with_missing])

# Remove duplicates from the dataset.
initial_shape = df.shape
df = df.drop_duplicates()
final_shape = df.shape
duplicates_removed = initial_shape[0] - final_shape[0]

# === Feature Selection and Encoding ===
columns_to_keep = [
    'TEAM_ABBREVIATION',
    'OPPONENT_TEAM_ABBREVIATION',
    'Total_First_Quarter_Points',
    'MONTH',
    'Home_Away',
    'Head_to_Head_Q1',
    'Season_Avg_Pace_Team',
    'Season_Avg_Pace_Opponent',
    'Average_Season_Avg_Pace',
    'Season_Avg_FG_PCT_Team',
    'Season_Avg_FG_PCT_Opponent',
    'PPG_Team',
    'PPG_Opponent',
    'Off_Rating_Avg',
    'Opp_Off_Rating_Avg',
    'Recent_Trend_Team_Avg',
    'Recent_Trend_Team_STD',
    'Recent_Trend_Team_Max',
    'Recent_Trend_Team_Min',
    'Recent_Trend_Team_Trend_Diff',
    'Recent_Trend_Opponent_Avg',
    'Recent_Trend_Opponent_STD',
    'Recent_Trend_Opponent_Max',
    'Recent_Trend_Opponent_Min',
    'Recent_Trend_Opponent_Trend_Diff'
]

df = df[columns_to_keep]

# Feature encoding using LabelEncoder for categorical variables.
le = LabelEncoder()
categorical_columns = ['TEAM_ABBREVIATION', 'OPPONENT_TEAM_ABBREVIATION', 'Home_Away']

for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# Ensure that 'Total_First_Quarter_Points' is numeric
df['Total_First_Quarter_Points'] = pd.to_numeric(df['Total_First_Quarter_Points'], errors='coerce')
df['Total_First_Quarter_Points'].fillna(0, inplace=True)

# Define target variable based on the new threshold of 52.
threshold = 48
df['target_over_52'] = (df['Total_First_Quarter_Points'] >= threshold).astype(int)
df = df.drop(columns=['Total_First_Quarter_Points'])

# Save the LabelEncoder for future use.
joblib.dump(le, 'label_encoder_tmp.pkl')

# === Feature Engineering Refinement ===
df['Pace_Interaction'] = df['Season_Avg_Pace_Team'] * df['Season_Avg_Pace_Opponent']
df['PPG_Interaction'] = df['PPG_Team'] * df['PPG_Opponent']
df['FG_PCT_Interaction'] = df['Season_Avg_FG_PCT_Team'] * df['Season_Avg_FG_PCT_Opponent']

df['PPG_Team_Squared'] = df['PPG_Team'] ** 2
df['PPG_Opponent_Squared'] = df['PPG_Opponent'] ** 2
df['Head_to_Head_Q1_Squared'] = df['Head_to_Head_Q1'] ** 2
df['Pace_Interaction_Squared'] = (df['Season_Avg_Pace_Team'] * df['Season_Avg_Pace_Opponent']) ** 2
df['FG_PCT_Team_Squared'] = df['Season_Avg_FG_PCT_Team'] ** 2
df['FG_PCT_Opponent_Squared'] = df['Season_Avg_FG_PCT_Opponent'] ** 2

df['FG_PCT_Difference'] = df['Season_Avg_FG_PCT_Team'] - df['Season_Avg_FG_PCT_Opponent']
df['Pace_Difference'] = df['Season_Avg_Pace_Team'] - df['Season_Avg_Pace_Opponent']
df['Off_Rating_Avg_Difference'] = df['Off_Rating_Avg'] - df['Opp_Off_Rating_Avg']
df['Off_Rating_Avg_Interaction'] = df['Off_Rating_Avg'] * df['Opp_Off_Rating_Avg']

# Drop features that may have little importance.
df.drop(columns=['Average_Season_Avg_Pace'], inplace=True)

# === Feature Scaling ===
categorical_columns = ['TEAM_ABBREVIATION', 'OPPONENT_TEAM_ABBREVIATION', 'Home_Away']
numerical_columns = [col for col in df.columns if col not in categorical_columns and col != 'target_over_52']

scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Save the scaler for future use.
joblib.dump(scaler, 'scaler_tmp.pkl')

# Separate features and the target variable.
X = df.drop(columns=['target_over_52'])
y = df['target_over_52'].astype(int)  # Ensure y is integer type


# Verify the target variable
print("Data type of y:", y.dtype)
print("Unique values in y:", y.unique())

# === Model Development and Cross-Validation ===
model_path = 'best_nba_model_tmp.pkl'
scaler_path = 'scaler.pkl_tmp'

if os.path.exists(model_path) and os.path.exists(scaler_path):
    print("\nExisting model and scaler found. Loading them...")
    
    loaded_scaler = joblib.load(scaler_path)
    print("Scaler loaded successfully.")
    
    best_model = joblib.load(model_path)
    print("Best model loaded successfully.")
else:
    print("\nNo existing model or scaler found. Proceeding with model training...")
    
    from imblearn.over_sampling import SMOTE
    
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )
    
    print("\nTraining and testing sets created.")
    print("Training set shape:", X_train.shape)
    print("Testing set shape:", X_test.shape)
    
    print("\n--- Logistic Regression Classifier ---")
    
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    
    print("\nLogistic Regression Classification Report:")
    print(classification_report(y_test, y_pred_lr))
    print(f"Logistic Regression ROC-AUC Score: {roc_auc_score(y_test, y_pred_lr):.4f}")
    
    # Prediction for the first row of the test set.
    first_test_instance = X_test.iloc[0].values.reshape(1, -1)
    first_prediction = lr.predict(first_test_instance)
    first_prediction_proba = lr.predict_proba(first_test_instance)
    
    print("\n--- Prediction for the First Row of the Test Set ---")
    print(f"Predicted Class: {'Over' if first_prediction[0] == 1 else 'Under'}")
    print(f"Predicted Probability: {first_prediction_proba[0][first_prediction[0]]:.4f}")
    print(f"Actual Class: {'Over' if y_test.iloc[0] == 1 else 'Under'}")
    
    print("\n--- Cross-Validation: Logistic Regression Classifier ---")
    
    lr_cv = LogisticRegression(random_state=42, max_iter=1000)
    cv_folds = 5
    lr_cv_scores = cross_val_score(lr_cv, X, y, cv=cv_folds, scoring='roc_auc', n_jobs=-1)
    
    print(f"Logistic Regression Cross-Validation ROC-AUC Scores: {lr_cv_scores}")
    print(f"Mean ROC-AUC Score: {lr_cv_scores.mean():.4f}")
    print(f"Standard Deviation of ROC-AUC Score: {lr_cv_scores.std():.4f}")
    
    print("\n--- Hyperparameter Tuning: Logistic Regression Classifier ---")
    
    lr_param_grid = {
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'C': [0.01, 0.1, 1, 10],
        'solver': ['lbfgs', 'liblinear', 'saga'],
        'class_weight': ['balanced', None],
        'max_iter': [1000]
    }
    
    lr_grid = LogisticRegression(random_state=42)
    lr_grid_search = GridSearchCV(lr_grid, lr_param_grid, cv=cv_folds, scoring='roc_auc', n_jobs=-1)
    lr_grid_search.fit(X, y)
    
    print(f"Best Parameters for Logistic Regression: {lr_grid_search.best_params_}")
    print(f"Best ROC-AUC Score from Grid Search for Logistic Regression: {lr_grid_search.best_score_:.4f}")
    
    best_model = lr_grid_search.best_estimator_
    
    # Save the best model and scaler.
    joblib.dump(best_model, model_path)
    print("Best model saved successfully.")
    
    joblib.dump(scaler, scaler_path)
    print("Scaler saved successfully.")
    
    # Prediction using the best model.
    best_first_prediction = best_model.predict(first_test_instance)
    best_first_prediction_proba = best_model.predict_proba(first_test_instance)
    
    print("\n--- Best Model Prediction for the First Row of the Test Set ---")
    print(f"Predicted Class: {'Over' if best_first_prediction[0] == 1 else 'Under'}")
    print(f"Predicted Probability: {best_first_prediction_proba[0][best_first_prediction[0]]:.4f}")
    print(f"Actual Class: {'Over' if y_test.iloc[0] == 1 else 'Under'}")

# === Prediction Using the Best Loaded Model ===
if os.path.exists(model_path) and os.path.exists(scaler_path):

    print("\n--- Making Prediction Using the Best Loaded Model ---")
    
    loaded_best_model = joblib.load(model_path)
    print("Best model loaded successfully.")
    
    loaded_scaler = joblib.load(scaler_path)
    print("Scaler loaded successfully.")
    
    last_instance = X.iloc[-1].values.reshape(1, -1)
    loaded_prediction = loaded_best_model.predict(last_instance)
    loaded_prediction_proba = loaded_best_model.predict_proba(last_instance)
    
    statement_one = "Over " + str(threshold)
    statement_two = "Under " + str(threshold + 1)

    print("\n--- Prediction for the Newly Added Row Using Loaded Best Model ---")
    print(f"GAME_DATE: {first_game_date}")
    print(f"TEAM1: {team1}")
    print(f"TEAM2: {team2}")
    print(f"Predicted Class: {statement_one if loaded_prediction[0] == 1 else statement_two}")
    print(f"Predicted Probability: {loaded_prediction_proba[0][loaded_prediction[0]]:.4f}")
    
    if len(sys.argv) == 3:
        actual_class = y.iloc[-1]
    else:
        actual_class = y.iloc[-1]
        print(f"Actual Class: {'Over' if actual_class == 1 else 'Under'}")
