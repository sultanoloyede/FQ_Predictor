import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from datetime import datetime

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve

# Function to retrieve the most recent game data for the specified team.
def get_most_recent_game(df, team_abbr):
    """
    Retrieves the most recent game data for the specified team.
    """
    team_games = df[df['TEAM_ABBREVIATION'] == team_abbr].copy()
    team_games = team_games.sort_values(by='GAME_DATE', ascending=False)
    return team_games.iloc[0] if not team_games.empty else None

# Function to construct a new row based on recent games and head-to-head data.
def construct_new_row(df, team_abbr, opponent_abbr):
    """
    Constructs a new row for the DataFrame based on the most recent games of the specified teams
    and the most recent head-to-head game between them.
    """
    team_recent_game = get_most_recent_game(df, team_abbr)
    opponent_recent_game = get_most_recent_game(df, opponent_abbr)
    
    new_row = {
        'TEAM_ABBREVIATION': team_abbr,
        'OPPONENT_TEAM_ABBREVIATION': opponent_abbr,
        'SEASON': '2024-25',
        'TOTAL_FQP': 0,
        'TEAM_LAST_10': 0,
        'TEAM_WORST_2': 0,
        'LAST_10_OPP': 0,
        'WORST_2_OPP': 0,
        'TEAM_FGA': 0,
        'FGA_OPP': 0,
        'TEAM_PACE': 0,
        'PACE_OPP': 0,
        'TEAM_LAST_10_AGAINST': 0,
        'TEAM_WORST_2_AGAINST': 0,
        'TEAM_EFG': 0,
        'EFG_OPP': 0,
        'TEAM_FG': 0,
        'FG_OPP': 0,
        'TEAM_3P_PCT': 0,
        '3P_PCT_OPP': 0,
        'TEAM_FT_PCT': 0,
        'FT_PCT_OPP': 0,
        'TEAM_FTA': 0,
        'FTA_OPP': 0,
        'TEAM_TOV': 0,
        'TOV_OPP': 0,
        'TEAM_OREB': 0,
        'OREB_OPP': 0,
        'TEAM_DREB': 0,
        'DREB_OPP': 0,
        'TEAM_OFF_RATING': 0,
        'OFF_RATING_OPP': 0,
        'TEAM_LAST_10_AGAINST': 0,
        'TEAM_WORST_2_AGAINST': 0,
        'LAST_10_OPP_AGAINST': 0,
        'WORST_2_OPP_AGAINST': 0,
        'TEAM_AST_TOV': 0,
        'AST_TOV_OPP': 0,
        'TEAM_LAST_3': 0,
        'LAST_3_OPP': 0,
        'LAST_3_H2H': 0,
        'LAST_3_PACE_H2H': 0,
        'LAST_3_EFG_H2H': 0,
        'xFQTP': 0,
        'xFQTP_OPP': 0,
        'GAME_DATE': pd.Timestamp.today().strftime('%Y-%m-%d'),
    }
    
    # Retrieve the most recent head-to-head game between the two teams
    head_to_head_games = df[
        (df['TEAM_ABBREVIATION'] == team_abbr) & 
        (df['OPPONENT_TEAM_ABBREVIATION'] == opponent_abbr)
    ].sort_values(by='GAME_DATE', ascending=False)
    
    if not head_to_head_games.empty:
        most_recent_h2h_game = head_to_head_games.iloc[0]
        new_row['TOTAL_FQP'] = most_recent_h2h_game.get('TOTAL_FQP', 0)
    
    # Populate team recent game features
    if team_recent_game is not None:
        new_row['TEAM_LAST_10'] = team_recent_game.get('TEAM_LAST_10', 0)
        new_row['TEAM_WORST_2'] = team_recent_game.get('TEAM_WORST_2', 0)
        new_row['TEAM_FGA'] = team_recent_game.get('TEAM_FGA', 0)
        new_row['TEAM_PACE'] = team_recent_game.get('TEAM_PACE', 0)
        new_row['TEAM_EFG'] = team_recent_game.get('TEAM_EFG', 0)
        new_row['TEAM_FG'] = team_recent_game.get('TEAM_FG', 0)
        new_row['TEAM_3P_PCT'] = team_recent_game.get('TEAM_3P_PCT', 0)
        new_row['TEAM_FT_PCT'] = team_recent_game.get('TEAM_FT_PCT', 0)
        new_row['TEAM_FTA'] = team_recent_game.get('TEAM_FTA', 0)
        new_row['TEAM_TOV'] = team_recent_game.get('TEAM_TOV', 0)
        new_row['TEAM_OREB'] = team_recent_game.get('TEAM_OREB', 0)
        new_row['TEAM_DREB'] = team_recent_game.get('TEAM_DREB', 0)
        new_row['TEAM_OFF_RATING'] = team_recent_game.get('TEAM_OFF_RATING', 0)
        new_row['TEAM_LAST_10_AGAINST'] = team_recent_game.get('TEAM_LAST_10_AGAINST', 0)
        new_row['TEAM_WORST_2_AGAINST'] = team_recent_game.get('TEAM_WORST_2_AGAINST', 0)
        new_row['TEAM_AST_TOV'] = team_recent_game.get('TEAM_AST_TOV', 0)
        new_row['TEAM_LAST_3'] = team_recent_game.get('TEAM_LAST_3', 0)
        new_row['xFQTP'] = team_recent_game.get('xFQTP', 0)
    
    # Populate opponent recent game features
    if opponent_recent_game is not None:
        new_row['LAST_10_OPP'] = opponent_recent_game.get('LAST_10_OPP', 0)
        new_row['WORST_2_OPP'] = opponent_recent_game.get('WORST_2_OPP', 0)
        new_row['FGA_OPP'] = opponent_recent_game.get('FGA_OPP', 0)
        new_row['PACE_OPP'] = opponent_recent_game.get('PACE_OPP', 0)
        new_row['EFG_OPP'] = opponent_recent_game.get('EFG_OPP', 0)
        new_row['FG_OPP'] = opponent_recent_game.get('FG_OPP', 0)
        new_row['3P_PCT_OPP'] = opponent_recent_game.get('3P_PCT_OPP', 0)
        new_row['FT_PCT_OPP'] = opponent_recent_game.get('FT_PCT_OPP', 0)
        new_row['FTA_OPP'] = opponent_recent_game.get('FTA_OPP', 0)
        new_row['TOV_OPP'] = opponent_recent_game.get('TOV_OPP', 0)
        new_row['OREB_OPP'] = opponent_recent_game.get('OREB_OPP', 0)
        new_row['DREB_OPP'] = opponent_recent_game.get('DREB_OPP', 0)
        new_row['OFF_RATING_OPP'] = opponent_recent_game.get('OFF_RATING_OPP', 0)
        new_row['LAST_10_OPP_AGAINST'] = opponent_recent_game.get('LAST_10_OPP_AGAINST', 0)
        new_row['WORST_2_OPP_AGAINST'] = opponent_recent_game.get('WORST_2_OPP_AGAINST', 0)
        new_row['AST_TOV_OPP'] = opponent_recent_game.get('AST_TOV_OPP', 0)
        new_row['LAST_3_OPP'] = opponent_recent_game.get('LAST_3_OPP', 0)
        new_row['xFQTP_OPP'] = opponent_recent_game.get('xFQTP_OPP', 0)
    
    new_row['Avg_Pace'] = np.mean([
        new_row['TEAM_PACE'],
        new_row['PACE_OPP']
    ])

    new_row['LAST_3_H2H'] = team_recent_game.get('LAST_3_H2H', 0) if team_recent_game is not None else 0
    new_row['LAST_3_PACE_H2H'] = team_recent_game.get('LAST_3_PACE_H2H', 0) if team_recent_game is not None else 0
    new_row['LAST_3_EFG_H2H'] = team_recent_game.get('LAST_3_EFG_H2H', 0) if team_recent_game is not None else 0
    
    return pd.DataFrame([new_row])

# Load dataset
df = pd.read_csv('xFQTP.csv')

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

team1 = df.loc[df.index[-1], 'TEAM_ABBREVIATION']
team2 = df.loc[df.index[-1], 'OPPONENT_TEAM_ABBREVIATION']

# === Feature Selection and Encoding ===
columns_to_keep = [
    'TEAM_ABBREVIATION',
    'OPPONENT_TEAM_ABBREVIATION',
    'TOTAL_FQP',
    'TEAM_LAST_10',
    'TEAM_WORST_2',
    'LAST_10_OPP',
    'WORST_2_OPP',
    'TEAM_FGA',
    'FGA_OPP',
    'TEAM_PACE',
    'PACE_OPP',
    'TEAM_EFG',
    'EFG_OPP',
    'TEAM_FG',
    'FG_OPP',
    'TEAM_3P_PCT',
    '3P_PCT_OPP',
    'TEAM_FT_PCT',
    'FT_PCT_OPP',
    'TEAM_FTA',
    'FTA_OPP',
    'TEAM_TOV',
    'TOV_OPP',
    'TEAM_OREB',
    'OREB_OPP',
    'TEAM_DREB',
    'DREB_OPP',
    'TEAM_OFF_RATING',
    'OFF_RATING_OPP',
    'TEAM_LAST_10_AGAINST',
    'TEAM_WORST_2_AGAINST',
    'LAST_10_OPP_AGAINST',
    'WORST_2_OPP_AGAINST',
    'TEAM_AST_TOV',
    'AST_TOV_OPP',
    'TEAM_LAST_3',
    'LAST_3_OPP',
    'LAST_3_H2H',
    'LAST_3_PACE_H2H',
    'LAST_3_EFG_H2H',
    'xFQTP',
    'xFQTP_OPP',
]

df = df[columns_to_keep]

# === Feature Encoding using One-Hot Encoding for Categorical Variables ===
categorical_columns = ['TEAM_ABBREVIATION', 'OPPONENT_TEAM_ABBREVIATION']
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Save the One-Hot Encoder columns for future use
one_hot_columns = df.columns.tolist()
joblib.dump(one_hot_columns, 'one_hot_columns.pkl')

# Ensure that 'TOTAL_FQP' is numeric
df['TOTAL_FQP'] = pd.to_numeric(df['TOTAL_FQP'], errors='coerce')
df['TOTAL_FQP'].fillna(0, inplace=True)

# Define target variable based on the new threshold of 52.5.
threshold = 52.5
df['target_over_52'] = (df['TOTAL_FQP'] > threshold).astype(int)
df = df.drop(columns=['TOTAL_FQP'])

# Compute correlation matrix
corr_matrix = df.corr()

# Extract correlation of each feature with the target
target_correlations = corr_matrix['target_over_52'].drop('target_over_52')

# Sort correlations by absolute value to see strongest associations first
target_correlations_sorted = target_correlations.abs().sort_values(ascending=False)

# Plot the correlations as a bar graph
plt.figure(figsize=(12, 8))
target_correlations_sorted.plot(kind='bar', color='skyblue')
plt.title('Feature Correlations with target_over_52')
plt.ylabel('Correlation Coefficient')
plt.xlabel('Features')
plt.tight_layout()
plt.show()

# === Feature Scaling ===
# Identify numerical columns (all except the target)
numerical_columns = [col for col in df.columns if col != 'target_over_52']

scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Save the scaler for future use.
joblib.dump(scaler, 'scl.pkl')

# Separate features and the target variable.
X = df.drop(columns=['target_over_52'])
y = df['target_over_52'].astype(int)  # Ensure y is integer type

# Verify the target variable
print("Data type of y:", y.dtype)
print("Unique values in y:", y.unique())
print(f"Class distribution:\n{y.value_counts(normalize=True) * 100}")

# === Model Development and Cross-Validation ===
model_path = 'best_nba_model.pkl'
scaler_path = 'scl.pkl'

if os.path.exists(model_path) and os.path.exists(scaler_path):
    print("\nExisting model and scaler found. Loading them...")
    
    loaded_scaler = joblib.load(scaler_path)
    print("Scaler loaded successfully.")
    
    best_model = joblib.load(model_path)
    print("Best model loaded successfully.")
else:
    print("\nNo existing model or scaler found. Proceeding with model training...")
    
    # === Split the data into training and testing sets ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("\nTraining and testing sets created.")
    print("Training set shape:", X_train.shape)
    print("Testing set shape:", X_test.shape)
    print(f"Training set class distribution:\n{y_train.value_counts(normalize=True) * 100}")
    print(f"Testing set class distribution:\n{y_test.value_counts(normalize=True) * 100}")
    
    # === Logistic Regression Classifier with Class Weights ===
    print("\n--- Logistic Regression Classifier ---")
    
    lr = LogisticRegression(
        random_state=42, 
        max_iter=1000, 
        class_weight='balanced'  # Using class weights to handle imbalance
    )
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    
    print("\nLogistic Regression Classification Report:")
    print(classification_report(y_test, y_pred_lr))
    print(f"Logistic Regression ROC-AUC Score: {roc_auc_score(y_test, lr.predict_proba(X_test)[:,1]):.4f}")
    
    # Prediction for the first row of the test set.
    first_test_instance = X_test.iloc[0].values.reshape(1, -1)
    first_prediction = lr.predict(first_test_instance)
    first_prediction_proba = lr.predict_proba(first_test_instance)
    
    print("\n--- Prediction for the First Row of the Test Set ---")
    print(f"Predicted Class: {'Over' if first_prediction[0] == 1 else 'Under'}")
    print(f"Predicted Probability of Over 52: {first_prediction_proba[0][1]:.4f}")
    print(f"Actual Class: {'Over' if y_test.iloc[0] == 1 else 'Under'}")
    
    # === Cross-Validation ===
    print("\n--- Cross-Validation: Logistic Regression Classifier ---")
    
    lr_cv = LogisticRegression(
        random_state=42, 
        max_iter=1000, 
        class_weight='balanced'
    )
    cv_folds = 5
    lr_cv_scores = cross_val_score(lr_cv, X, y, cv=cv_folds, scoring='roc_auc', n_jobs=-1)
    
    print(f"Logistic Regression Cross-Validation ROC-AUC Scores: {lr_cv_scores}")
    print(f"Mean ROC-AUC Score: {lr_cv_scores.mean():.4f}")
    print(f"Standard Deviation of ROC-AUC Score: {lr_cv_scores.std():.4f}")
    
    # === Hyperparameter Tuning: Logistic Regression ===
    print("\n--- Hyperparameter Tuning: Logistic Regression Classifier ---")
    
    lr_param_grid = {
        'penalty': ['l2'],  # 'l1' and 'elasticnet' are not supported with 'lbfgs' and 'balanced' class_weight
        'C': [0.01, 0.1, 1, 10],
        'solver': ['lbfgs', 'saga'],
        'class_weight': ['balanced'],
        'max_iter': [1000]
    }
    
    lr_grid = LogisticRegression(random_state=42)
    lr_grid_search = GridSearchCV(lr_grid, lr_param_grid, cv=cv_folds, scoring='roc_auc', n_jobs=-1)
    lr_grid_search.fit(X_train, y_train)
    
    print(f"Best Parameters for Logistic Regression: {lr_grid_search.best_params_}")
    print(f"Best ROC-AUC Score from Grid Search for Logistic Regression: {lr_grid_search.best_score_:.4f}")
    
    best_model_lr = lr_grid_search.best_estimator_
    
    # === Random Forest Classifier with Class Weights ===
    print("\n--- Random Forest Classifier ---")
    
    rf = RandomForestClassifier(
        random_state=42,
        class_weight='balanced',
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    
    print("\nRandom Forest Classification Report:")
    print(classification_report(y_test, y_pred_rf))
    print(f"Random Forest ROC-AUC Score: {roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]):.4f}")
    
    # === Cross-Validation for Random Forest ===
    print("\n--- Cross-Validation: Random Forest Classifier ---")
    
    rf_cv = RandomForestClassifier(
        random_state=42,
        class_weight='balanced'
    )
    rf_cv_scores = cross_val_score(rf_cv, X, y, cv=cv_folds, scoring='roc_auc', n_jobs=-1)
    
    print(f"Random Forest Cross-Validation ROC-AUC Scores: {rf_cv_scores}")
    print(f"Mean ROC-AUC Score: {rf_cv_scores.mean():.4f}")
    print(f"Standard Deviation of ROC-AUC Score: {rf_cv_scores.std():.4f}")
    
    # === Hyperparameter Tuning: Random Forest ===
    print("\n--- Hyperparameter Tuning: Random Forest Classifier ---")
    
    rf_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': ['balanced']
    }
    
    rf_grid = RandomForestClassifier(random_state=42)
    rf_grid_search = GridSearchCV(rf_grid, rf_param_grid, cv=cv_folds, scoring='roc_auc', n_jobs=-1)
    rf_grid_search.fit(X_train, y_train)
    
    print(f"Best Parameters for Random Forest: {rf_grid_search.best_params_}")
    print(f"Best ROC-AUC Score from Grid Search for Random Forest: {rf_grid_search.best_score_:.4f}")
    
    best_model_rf = rf_grid_search.best_estimator_
    
    # === Choose Best Model Based on ROC-AUC ===
    if lr_grid_search.best_score_ > rf_grid_search.best_score_:
        best_model = best_model_lr
        print("\nSelected Logistic Regression as the Best Model.")
    else:
        best_model = best_model_rf
        print("\nSelected Random Forest as the Best Model.")
    
    # Save the best model and scaler.
    joblib.dump(best_model, model_path)
    print("Best model saved successfully.")
    
    joblib.dump(scaler, scaler_path)
    print("Scaler saved successfully.")
    
    # === Calibration Curve ===
    print("\n--- Calibration Curve for the Best Model ---")
    
    if hasattr(best_model, "predict_proba"):
        prob_true, prob_pred = calibration_curve(y_test, best_model.predict_proba(X_test)[:,1], n_bins=10)
        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, marker='o', label='Best Model')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
        plt.xlabel('Predicted Probability')
        plt.ylabel('True Probability')
        plt.title('Calibration Curve')
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("The best model does not support probability predictions for calibration.")
    
    # === Prediction Interpretation ===
    print("\n--- Prediction for the First Row of the Test Set Using the Best Model ---")
    best_first_prediction = best_model.predict(first_test_instance)
    best_first_prediction_proba = best_model.predict_proba(first_test_instance)
    
    print(f"Predicted Class: {'Over 52.5' if best_first_prediction[0] == 1 else 'Under 52.5'}")
    print(f"Predicted Probability of Over 52.5: {best_first_prediction_proba[0][1]:.4f}")
    print(f"Actual Class: {'Over' if y_test.iloc[0] == 1 else 'Under'}")

# === Prediction Using the Best Loaded Model ===
if os.path.exists(model_path) and os.path.exists(scaler_path):
    print("\n--- Making Prediction Using the Best Loaded Model ---")
    
    loaded_best_model = joblib.load(model_path)
    print("Best model loaded successfully.")
    
    loaded_scaler = joblib.load(scaler_path)
    print("Scaler loaded successfully.")
    
    # Ensure that the new row is processed the same way as training data
    last_instance = X.iloc[-1].values.reshape(1, -1)
    
    loaded_prediction = loaded_best_model.predict(last_instance)
    loaded_prediction_proba = loaded_best_model.predict_proba(last_instance)
    
    print("\n--- Prediction for the Newly Added Row Using Loaded Best Model ---")
    print(f"TEAM1: {team1}")
    print(f"TEAM2: {team2}")
    print(f"Predicted Class: {'Over 52.5' if loaded_prediction[0] == 1 else 'Under 52.5'}")
    print(f"Predicted Probability of Over 52.5: {loaded_prediction_proba[0][1]:.4f}")
    
    if len(sys.argv) == 3:
        actual_class = y.iloc[-1]
        print(f"Actual Class: {'Over' if actual_class == 1 else 'Under'}")
    else:
        print(f"Actual Class: {'Over' if y.iloc[-1] == 1 else 'Under'}")
