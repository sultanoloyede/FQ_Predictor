# -------------------------------
# NBA Betting Predictor: Full Pipeline
# -------------------------------

# Step 0: Install Required Libraries (Optional)
# -------------------------------

# If you haven't installed the necessary libraries, uncomment the following lines and run them.
# !pip install scikit-learn pandas numpy matplotlib seaborn joblib imbalanced-learn shap

# Step 1: Import Necessary Libraries
# -------------------------------

import pandas as pd
import numpy as np
import joblib  # For saving and loading models and scalers
import os      # For checking file existence
import matplotlib.pyplot as plt
import seaborn as sns
import sys      # For command-line arguments
from datetime import datetime

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Import the classifiers used for model training
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# -------------------------------

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
        new_row['Season_Avg_Pace_Opponent'] = opponent_recent_game.get('Season_Avg_Pace_Opponent', 0)
        new_row['Season_Avg_FG_PCT_Opponent'] = opponent_recent_game.get('Season_Avg_FG_PCT_Opponent', 0)
        new_row['PPG_Opponent'] = opponent_recent_game.get('PPG_Opponent', 0)
        new_row['Recent_Trend_Opponent'] = opponent_recent_game.get('Recent_Trend_Opponent', '[0,0,0]')
    
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

# Define the path to your CSV file
data_path = 'final_training_dataset.csv'

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(data_path)

print("\n--- Last Row of the Dataset Before Adding New Row ---")
print(df.iloc[-1])

# ----------------------------------
# New Feature Integration: Conditional Row Addition with Recent Game Data
# ----------------------------------

# Check if two command-line arguments are provided (excluding the script name)
# Usage:
#   python AI.py BOS CHA  # Adds a new row with TEAM_ABBREVIATION='BOS' and OPPONENT_TEAM_ABBREVIATION='CHA'
#   python AI.py          # Does not add a new row
if len(sys.argv) == 3:
    team_abbr = sys.argv[1]
    opponent_abbr = sys.argv[2]
    
    # Validate that the provided team abbreviations exist in the dataset
    unique_teams = df['TEAM_ABBREVIATION'].unique()
    unique_opponents = df['OPPONENT_TEAM_ABBREVIATION'].unique()
    
    if team_abbr not in unique_teams:
        print(f"Error: TEAM_ABBREVIATION '{team_abbr}' not found in the dataset.")
        sys.exit(1)
    if opponent_abbr not in unique_opponents:
        print(f"Error: OPPONENT_TEAM_ABBREVIATION '{opponent_abbr}' not found in the dataset.")
        sys.exit(1)
    
    # Construct the new row using the most recent games
    new_row_df = construct_new_row(df, team_abbr, opponent_abbr)
    
    # Append the new row to the existing DataFrame using pd.concat
    df = pd.concat([df, new_row_df], ignore_index=True)
    
    print("\n--- Last Row of the Dataset After Adding New Row ---")
    print(df.iloc[-1])
else:
    print("\n--- No New Row Added ---")

# ----------------------------------
# Continue with the rest of your pipeline
# ----------------------------------

# Extract GAME_DATE from the last row
first_game_date = df.loc[df.index[-1], 'GAME_DATE']

# Extract TEAM_ABBREVIATION from the last row
team1 = df.loc[df.index[-1], 'TEAM_ABBREVIATION']

# Extract OPPONENT_TEAM_ABBREVIATION from the last row
team2 = df.loc[df.index[-1], 'OPPONENT_TEAM_ABBREVIATION']

# Display the first five rows of the dataset
print("\nFirst five rows of the dataset:")
print(df.head())

# Display DataFrame information
print("\nDataFrame Information:")
print(df.info())

# Display summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values Per Column:")
print(df.isnull().sum())

# -------------------------------
# Step 3: Data Cleaning and Recent Trend Feature Engineering
# -------------------------------

# Convert 'Recent_Trend_Team' and 'Recent_Trend_Opponent' from string representation of lists to actual lists of integers.
# Handle cases where there may be missing or invalid values gracefully.
def convert_trend(trend):
    try:
        # Remove square brackets, split by commas, and convert to integers while filtering out any invalid values
        return [int(i) for i in trend.strip('[]').split(',') if i.strip().isdigit()]
    except Exception as e:
        return []  # Return an empty list in case of an error

# Apply the conversion to both columns
df['Recent_Trend_Team'] = df['Recent_Trend_Team'].apply(convert_trend)
df['Recent_Trend_Opponent'] = df['Recent_Trend_Opponent'].apply(convert_trend)

# Create statistical summary features for Recent_Trend_Team and Recent_Trend_Opponent
for trend_col in ['Recent_Trend_Team', 'Recent_Trend_Opponent']:
    df[f'{trend_col}_Avg'] = df[trend_col].apply(lambda x: np.mean(x) if len(x) > 0 else 0)
    df[f'{trend_col}_STD'] = df[trend_col].apply(lambda x: np.std(x) if len(x) > 0 else 0)
    df[f'{trend_col}_Max'] = df[trend_col].apply(lambda x: np.max(x) if len(x) > 0 else 0)
    df[f'{trend_col}_Min'] = df[trend_col].apply(lambda x: np.min(x) if len(x) > 0 else 0)
    df[f'{trend_col}_Trend_Diff'] = df[trend_col].apply(lambda x: (x[-1] - x[0]) if len(x) > 1 else 0)

# Drop original Recent_Trend columns since we've derived summary features
df = df.drop(columns=['Recent_Trend_Team', 'Recent_Trend_Opponent'])

print("\nAdded statistical summary features for recent trends. Updated DataFrame columns:")
print(df.columns.tolist())

# Handle Missing Values
# Define columns with missing values that require imputation
columns_with_missing = [
    'Head_to_Head_Q1', 'PTS', 'OPP_PTS', 'Off_Rating', 
    'PPG_Team', 'PPG_Opponent', 'Off_Rating_Avg'
]

# Initialize SimpleImputer with median strategy
imputer = SimpleImputer(strategy='median')

# Apply imputer to the specified columns
df[columns_with_missing] = imputer.fit_transform(df[columns_with_missing])

print("\nMissing values imputed with median strategy.")

# Verify Missing Values After Imputation
print("\nMissing Values After Imputation:")
print(df.isnull().sum())

# Remove Duplicates
initial_shape = df.shape
df = df.drop_duplicates()
final_shape = df.shape
duplicates_removed = initial_shape[0] - final_shape[0]
print(f"\nDuplicates removed: {duplicates_removed}")

# -------------------------------
# Step 4: Feature Selection and Encoding (Optimized)
# -------------------------------

# Select Relevant Columns (as specified in your requirements)
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

# Keep only the specified columns in the DataFrame
df = df[columns_to_keep]

print("\nSelected relevant columns. Columns now in DataFrame:")
print(df.columns.tolist())

# -------------------------------
# Step 5: Feature Encoding (Optimized)
# -------------------------------

# Initialize LabelEncoder
le = LabelEncoder()

# Handle Categorical Variables with LabelEncoder
categorical_columns = ['TEAM_ABBREVIATION', 'OPPONENT_TEAM_ABBREVIATION', 'Home_Away']

for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

print("\nEncoded categorical columns. Updated DataFrame columns:")
print(df.columns.tolist())

# Define a threshold to classify games into high or low scoring in the first quarter
threshold = df['Total_First_Quarter_Points'].median()

# Create a new binary target variable
df['target_over_55'] = (df['Total_First_Quarter_Points'] >= threshold).astype(int)

# Drop the original target variable
df = df.drop(columns=['Total_First_Quarter_Points'])

print("\nConverted target variable to binary classification. Threshold used:", threshold)

# After label encoding during training
# Save the LabelEncoder
joblib.dump(le, 'label_encoder.pkl')
print("LabelEncoder saved successfully.")

# -------------------------------
# Phase 1: Feature Engineering Refinement
# -------------------------------

# Create new interaction features based on domain knowledge
df['Pace_Interaction'] = df['Season_Avg_Pace_Team'] * df['Season_Avg_Pace_Opponent']
df['PPG_Interaction'] = df['PPG_Team'] * df['PPG_Opponent']
df['FG_PCT_Interaction'] = df['Season_Avg_FG_PCT_Team'] * df['Season_Avg_FG_PCT_Opponent']

# Polynomial Features (e.g., Squared values to capture non-linear relationships)
df['PPG_Team_Squared'] = df['PPG_Team'] ** 2
df['PPG_Opponent_Squared'] = df['PPG_Opponent'] ** 2
df['Head_to_Head_Q1_Squared'] = df['Head_to_Head_Q1'] ** 2
# Additional interaction and polynomial features (Optional)
df['Pace_Interaction_Squared'] = (df['Season_Avg_Pace_Team'] * df['Season_Avg_Pace_Opponent']) ** 2
df['FG_PCT_Team_Squared'] = df['Season_Avg_FG_PCT_Team'] ** 2
df['FG_PCT_Opponent_Squared'] = df['Season_Avg_FG_PCT_Opponent'] ** 2

# Difference features to capture relative performance
df['FG_PCT_Difference'] = df['Season_Avg_FG_PCT_Team'] - df['Season_Avg_FG_PCT_Opponent']
df['Pace_Difference'] = df['Season_Avg_Pace_Team'] - df['Season_Avg_Pace_Opponent']

# Drop features that may have little importance based on analysis
df.drop(columns=['Average_Season_Avg_Pace'], inplace=True)

# Confirm the DataFrame after the feature engineering refinements
print("\nDataFrame after feature engineering refinements:")
print(df.head())

# -------------------------------
# Step 6: Feature Encoding and Scaling (Revised After Feature Engineering)
# -------------------------------

# Identify categorical and numerical columns
categorical_columns = ['TEAM_ABBREVIATION', 'OPPONENT_TEAM_ABBREVIATION', 'Home_Away']
numerical_columns = [col for col in df.columns if col not in categorical_columns and col != 'target_over_55']

# Apply StandardScaler to the numerical features
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Save the scaler for future use
joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved successfully.")

# Separate features and the target variable after adding new features
X = df.drop(columns=['target_over_55'])
y = df['target_over_55']

print("\nFeatures scaled using StandardScaler after feature engineering. Head of the updated DataFrame:")
print(X.head())

# -------------------------------
# Step 7 & 8: Model Development and Cross-Validation
# (Conditionally executed based on existing model and scaler)
# -------------------------------

# Define paths to model and scaler files
model_path = 'best_nba_model.pkl'
scaler_path = 'scaler.pkl'

# Check if both model and scaler files exist
if os.path.exists(model_path) and os.path.exists(scaler_path):
    print("\nExisting model and scaler found. Loading them...")
    
    # Load the scaler
    loaded_scaler = joblib.load(scaler_path)
    print("Scaler loaded successfully.")
    
    # Load the best model
    best_model = joblib.load(model_path)
    print("Best model loaded successfully.")
    
    # Since scaler is already applied to X, no need to scale again
    # However, to ensure consistency, you might want to verify or re-apply if necessary
    # For this script, we'll assume X is already scaled
else:
    print("\nNo existing model or scaler found. Proceeding with model training...")
    
    from imblearn.over_sampling import SMOTE
    
    # Apply SMOTE to balance the dataset
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Update train-test split using the balanced dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )
    
    print("\nTraining and testing sets created.")
    print("Training set shape:", X_train.shape)
    print("Testing set shape:", X_test.shape)
    
    # -------------------------------
    # Logistic Regression Classifier
    # -------------------------------
    print("\n--- Logistic Regression Classifier ---")
    
    # Initialize Logistic Regression Classifier
    lr = LogisticRegression(random_state=42, max_iter=1000)
    
    # Train the model
    lr.fit(X_train, y_train)
    
    # Predict on the test data
    y_pred_lr = lr.predict(X_test)
    
    # Evaluate the model
    print("\nLogistic Regression Classification Report:")
    print(classification_report(y_test, y_pred_lr))
    print(f"Logistic Regression ROC-AUC Score: {roc_auc_score(y_test, y_pred_lr):.4f}")
    
    # -------------------------------
    # Display Prediction for the First Row
    # -------------------------------
    
    # Extract the first row of the test set
    first_test_instance = X_test.iloc[0].values.reshape(1, -1)
    
    # Predict using the trained Logistic Regression model
    first_prediction = lr.predict(first_test_instance)
    
    # If you want to get the probability instead of the class label, use predict_proba
    first_prediction_proba = lr.predict_proba(first_test_instance)
    
    # Print the prediction results
    print("\n--- Prediction for the First Row of the Test Set ---")
    print(f"Predicted Class: {'Over' if first_prediction[0] == 1 else 'Under'}")
    print(f"Predicted Probability: {first_prediction_proba[0][first_prediction[0]]:.4f}")
    print(f"Actual Class: {'Over' if y_test.iloc[0] == 1 else 'Under'}")
    
    # -------------------------------
    # Cross-Validation for Logistic Regression Classifier
    # -------------------------------
    print("\n--- Cross-Validation: Logistic Regression Classifier ---")
    
    # Initialize Logistic Regression Classifier
    lr_cv = LogisticRegression(random_state=42, max_iter=1000)
    
    # Number of cross-validation folds
    cv_folds = 5
    
    # Perform cross-validation
    lr_cv_scores = cross_val_score(lr_cv, X, y, cv=cv_folds, scoring='roc_auc', n_jobs=-1)
    
    # Print cross-validation results
    print(f"Logistic Regression Cross-Validation ROC-AUC Scores: {lr_cv_scores}")
    print(f"Mean ROC-AUC Score: {lr_cv_scores.mean():.4f}")
    print(f"Standard Deviation of ROC-AUC Score: {lr_cv_scores.std():.4f}")
    
    # -------------------------------
    # Hyperparameter Tuning using GridSearchCV
    # -------------------------------
    
    print("\n--- Hyperparameter Tuning: Logistic Regression Classifier ---")
    
    # Define hyperparameter grid for Logistic Regression
    lr_param_grid = {
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'C': [0.01, 0.1, 1, 10],
        'solver': ['lbfgs', 'liblinear', 'saga'],
        'class_weight': ['balanced', None],
        'max_iter': [1000]
    }
    
    # Initialize Logistic Regression Classifier
    lr_grid = LogisticRegression(random_state=42)
    
    # Perform Grid Search
    lr_grid_search = GridSearchCV(lr_grid, lr_param_grid, cv=cv_folds, scoring='roc_auc', n_jobs=-1)
    lr_grid_search.fit(X, y)
    
    # Print best parameters and best score for Logistic Regression
    print(f"Best Parameters for Logistic Regression: {lr_grid_search.best_params_}")
    print(f"Best ROC-AUC Score from Grid Search for Logistic Regression: {lr_grid_search.best_score_:.4f}")
    
    # Assuming the best model is the Logistic Regression after hyperparameter tuning
    best_model = lr_grid_search.best_estimator_
    
    # Save the best model
    joblib.dump(best_model, model_path)
    print("Best model saved successfully.")
    
    # Save the scaler
    joblib.dump(scaler, scaler_path)
    print("Scaler saved successfully.")
    
    # -------------------------------
    # Display Prediction for the First Row Using Best Model
    # -------------------------------
    
    # Predict using the best model
    best_first_prediction = best_model.predict(first_test_instance)
    
    # Get prediction probabilities
    best_first_prediction_proba = best_model.predict_proba(first_test_instance)
    
    # Print the prediction results
    print("\n--- Best Model Prediction for the First Row of the Test Set ---")
    print(f"Predicted Class: {'Over' if best_first_prediction[0] == 1 else 'Under'}")
    print(f"Predicted Probability: {best_first_prediction_proba[0][best_first_prediction[0]]:.4f}")
    print(f"Actual Class: {'Over' if y_test.iloc[0] == 1 else 'Under'}")

# -------------------------------
# If Model Exists, Make Prediction Using Best Model
# -------------------------------
if os.path.exists(model_path) and os.path.exists(scaler_path):

    print("\n--- Making Prediction Using the Best Loaded Model ---")
    
    # Load the best model
    loaded_best_model = joblib.load(model_path)
    print("Best model loaded successfully.")
    
    # Load the scaler
    loaded_scaler = joblib.load(scaler_path)
    print("Scaler loaded successfully.")
    
    # Since the scaler is already applied to X, and X includes the new row if added,
    # we can directly use the last row for prediction
    last_instance = X.iloc[-1].values.reshape(1, -1)
    
    # Make prediction
    loaded_prediction = loaded_best_model.predict(last_instance)
    loaded_prediction_proba = loaded_best_model.predict_proba(last_instance)
    
    # Print the prediction results
    print("\n--- Prediction for the Newly Added Row Using Loaded Best Model ---")
    print(f"GAME_DATE: {first_game_date}")
    print(f"TEAM1: {team1}")
    print(f"TEAM2: {team2}")
    print(f"Predicted Class: {'Over' if loaded_prediction[0] == 1 else 'Under'}")
    print(f"Predicted Probability: {loaded_prediction_proba[0][loaded_prediction[0]]:.4f}")
    
    # Check if a new row was added
    if len(sys.argv) == 3:
        print(f"Actual Class: {'Unknown'}")  # Replace 'Unknown' with the actual class if available
    else:
        # If no new row was added, this is the actual class from the last row
        # Assuming 'target_over_55' is available in y
        actual_class = y.iloc[-1]
        print(f"Actual Class: {'Over' if actual_class == 1 else 'Under'}")
