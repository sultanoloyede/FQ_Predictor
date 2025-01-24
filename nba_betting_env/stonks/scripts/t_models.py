import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib

def load_and_merge_all_tickers(stocks_dir: str) -> pd.DataFrame:
    all_dfs = []
    for fname in os.listdir(stocks_dir):
        if not fname.endswith('.csv'):
            continue
        fpath = os.path.join(stocks_dir, fname)
        try:
            df = pd.read_csv(fpath)
            if df.empty:
                continue
            all_dfs.append(df)
        except Exception as e:
            print(f"Error reading {fname}: {e}")

    if not all_dfs:
        print("No CSV files found or all empty in", stocks_dir)
        return pd.DataFrame()

    merged_df = pd.concat(all_dfs, ignore_index=True)
    return merged_df

def create_nextrow_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by Ticker so we only shift within each ticker's rows.
    Shifts the columns Above1Pct, Above2Pct, Above3Pct upward (-1) to create new label columns:
      Label_Above1Pct_Next, Label_Above2Pct_Next, Label_Above3Pct_Next.
    Then removes the last row of each group (which has no next row).
    """
    df = df.sort_values(['Ticker', 'Year', 'Month', 'Day']).reset_index(drop=True)

    def shift_labels(group):
        group = group.copy()
        group['Label_Above1Pct_Next'] = group['Above1Pct'].shift(-1)
        group['Label_Above2Pct_Next'] = group['Above2Pct'].shift(-1)
        group['Label_Above3Pct_Next'] = group['Above3Pct'].shift(-1)
        # Drop the last row (NaN labels)
        if len(group) > 1:
            group = group.iloc[:-1]
        else:
            group = group.iloc[0:0]  # empty
        return group

    df_labeled = df.groupby('Ticker', group_keys=True).apply(shift_labels)
    df_labeled.reset_index(drop=True, inplace=True)
    return df_labeled

def check_and_apply_smote(X_train, y_train, minority_class_threshold=0.4):
    vc = y_train.value_counts(normalize=True)
    if len(vc) < 2:
        return X_train, y_train  # can't do SMOTE with only one class
    minority_class_ratio = vc.min()
    if minority_class_ratio < minority_class_threshold:
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        return X_res, y_res
    else:
        return X_train, y_train

def train_and_select_best_model(X_train, X_test, y_train, y_test, threshold_name):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "KNeighbors": KNeighborsClassifier(),
        "GradientBoosting": GradientBoostingClassifier(),
        "RandomForest": RandomForestClassifier()
    }

    best_model = None
    best_model_name = None
    best_score = -np.inf

    for mname, m in models.items():
        try:
            m.fit(X_train, y_train)
            score = m.score(X_test, y_test)
            print(f"[{threshold_name}] {mname} -> Test Accuracy: {score:.4f}")
            if score > best_score:
                best_score = score
                best_model_name = mname
                best_model = m
        except Exception as e:
            print(f"[{threshold_name}] Error training {mname}: {e}")
    return best_model_name, best_model, best_score

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    stocks_dir = os.path.join(data_dir, 'stocks')
    models_dir = os.path.join(script_dir, '..', 'models')
    os.makedirs(models_dir, exist_ok=True)

    # 1) Merge
    df_all = load_and_merge_all_tickers(stocks_dir)
    if df_all.empty:
        print("No data found, exiting.")
        return

    # 2) Remove 'Date' if it exists
    if 'Date' in df_all.columns:
        df_all.drop(columns=['Date'], inplace=True, errors='ignore')

    # 3) Create next-row labels
    df_all = create_nextrow_labels(df_all)
    if df_all.empty:
        print("After shifting, no data. Possibly each ticker had < 2 rows. Exiting.")
        return

    # 4) Save the consolidated dataset
    consolidated_csv_path = os.path.join(data_dir, 'consolidated_weekly.csv')
    df_all.to_csv(consolidated_csv_path, index=False)
    print(f"Saved consolidated dataset (with next-row labels) to {consolidated_csv_path}.")

    # 5) Train models for each threshold
    thresholds = {
        "Above1Pct_Next": "Label_Above1Pct_Next",
        "Above2Pct_Next": "Label_Above2Pct_Next",
        "Above3Pct_Next": "Label_Above3Pct_Next"
    }

    # Example feature selection:
    # We'll exclude the Ticker column from features (we won't OHE or encode it).
    # Also exclude Above1Pct, Above2Pct, Above3Pct, and the label columns themselves.
    exclude_cols = ['Ticker', 'Above1Pct', 'Above2Pct', 'Above3Pct'] + list(thresholds.values())

    all_cols = df_all.columns.tolist()
    feature_cols = [c for c in all_cols if c not in exclude_cols]

    from sklearn.model_selection import train_test_split

    for threshold_key, label_col in thresholds.items():
        if label_col not in df_all.columns:
            print(f"Label column {label_col} not found. Skipping.")
            continue

        # Build X, y
        X = df_all[feature_cols].copy()
        y = df_all[label_col].copy()

        # Drop rows with NaN
        data_merged = pd.concat([X, y], axis=1).dropna()
        X = data_merged[feature_cols]
        y = data_merged[label_col]

        if len(X) < 2:
            print(f"Not enough data for threshold {threshold_key} after dropna. Skipping.")
            continue

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # SMOTE if needed
        vc = y_train.value_counts(normalize=True)
        minority_ratio = vc.min() if len(vc) > 1 else 1.0
        if minority_ratio < 0.4:
            from imblearn.over_sampling import SMOTE
            sm = SMOTE(random_state=42)
            X_train, y_train = sm.fit_resample(X_train, y_train)
            print(f"[{threshold_key}] Applied SMOTE because minority ratio was {minority_ratio:.2f}")

        best_model_name, best_model, best_score = train_and_select_best_model(
            X_train, X_test, y_train, y_test, threshold_key
        )

        if best_model is None:
            print(f"No successful model found for {threshold_key}.")
            continue

        print(f"[{threshold_key}] Best Model: {best_model_name}, Score: {best_score:.4f}")

        # Save the best model
        model_filename = f"model_{threshold_key}_{best_model_name}_acc_{best_score:.4f}.pkl"
        model_path = os.path.join(models_dir, model_filename)
        joblib.dump(best_model, model_path)
        print(f"Saved best model for {threshold_key} => {model_path}")

if __name__ == "__main__":
    main()
