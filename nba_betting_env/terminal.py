# train_model.py

import os
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score

# Define paths
DATA_PATH = 'data.csv'  # Replace with your actual data file path
MODELS_DIR = 'models'

# Create models directory if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)

# Feature Engineering
# Apply the same feature engineering steps as in prediction
df['Pace_Interaction'] = df['Season_Avg_Pace_Team'] * df['Season_Avg_Pace_Opponent']
df['PPG_Interaction'] = df['PPG_Team'] * df['PPG_Opponent']
df['FG_PCT_Interaction'] = df['Season_Avg_FG_PCT_Team'] * df['Season_Avg_FG_PCT_Opponent']

df['PPG_Team_Squared'] = df['PPG_Team'] ** 2
df['PPG_Opponent_Squared'] = df['PPG_Opponent'] ** 2
df['Head_to_Head_Q1_Squared'] = df['Head_to_Head_Q1'] ** 2

# Define target variable
TARGET_THRESHOLD = 55
df['target_over_55'] = (df['Total_First_Quarter_Points'] >= TARGET_THRESHOLD).astype(int)

# Define features and target
X = df.drop(columns=['Total_First_Quarter_Points', 'target_over_55'])
y = df['target_over_55']

# Define categorical and numerical columns
categorical_columns = ['TEAM_ABBREVIATION', 'OPPONENT_TEAM_ABBREVIATION', 'Home_Away']
numerical_columns = [col for col in X.columns if col not in categorical_columns]

# Define preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
        ('num', StandardScaler(), numerical_columns)
    ]
)

# Create Pipeline with preprocessing and classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
roc_auc = roc_auc_score(y_test, y_pred)
print(f"ROC-AUC Score: {roc_auc:.4f}")

# Cross-Validation
print("\nCross-Validation ROC-AUC Scores:")
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc', n_jobs=-1)
print(cv_scores)
print(f"Mean ROC-AUC: {cv_scores.mean():.4f}")
print(f"Std ROC-AUC: {cv_scores.std():.4f}")

# Save the trained pipeline
pipeline_path = os.path.join(MODELS_DIR, 'nba_predictor_pipeline.pkl')
joblib.dump(pipeline, pipeline_path)
print(f"\nPipeline saved at '{pipeline_path}'")
