import os
import pandas as pd
import numpy as np
import joblib

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    models_dir = os.path.join(script_dir, '..', 'models')

    # 1) Load consolidated dataset (with next-row labels)
    consolidated_csv = os.path.join(data_dir, 'consolidated_weekly.csv')
    if not os.path.exists(consolidated_csv):
        print(f"No consolidated file at {consolidated_csv}. Exiting.")
        return

    df_all = pd.read_csv(consolidated_csv)
    if df_all.empty:
        print("Consolidated dataset is empty. Exiting.")
        return

    # 2) For each ticker, pick the MOST RECENT row
    #    We'll define "most recent" by sorting [Year, Month, Day].
    df_all.sort_values(['Ticker', 'Year', 'Month', 'Day'], inplace=True)
    df_all.reset_index(drop=True, inplace=True)

    # Group by ticker, take last row in each group
    def pick_last_row(group):
        # group is already sorted
        return group.iloc[-1:]  # last row only

    df_recent = df_all.groupby('Ticker', group_keys=True).apply(pick_last_row).reset_index(drop=True)

    # 3) Prepare the feature set (the same columns we used in training).
    thresholds = {
        "Above1Pct_Next": "Label_Above1Pct_Next",
        "Above2Pct_Next": "Label_Above2Pct_Next",
        "Above3Pct_Next": "Label_Above3Pct_Next"
    }
    exclude_cols = ['Ticker', 'Above1Pct', 'Above2Pct', 'Above3Pct'] + list(thresholds.values())

    all_cols = df_recent.columns.tolist()
    feature_cols = [c for c in all_cols if c not in exclude_cols]

    # We'll build X_input from df_recent
    # Note: If df_recent has only 1 row for some ticker, that's the row we predict on
    X_input = df_recent[feature_cols].copy()

    # Drop any NaNs
    # If a row has NaN in a feature, the model can't predict => drop that row
    data_merged = pd.concat([df_recent[['Ticker']], X_input], axis=1)
    data_merged.dropna(inplace=True)
    # keep Ticker column for final output
    X_input = data_merged[feature_cols]
    tickers_for_output = data_merged['Ticker'].values  # same index as X_input

    if X_input.empty:
        print("No valid rows to predict on after dropping NaNs. Exiting.")
        return

    # 4) Load each best model
    loaded_models = {}
    possible_thresholds = ["Above1Pct_Next", "Above2Pct_Next", "Above3Pct_Next"]

    for tkey in possible_thresholds:
        # find model file(s) in ../models
        pkl_files = [
            f for f in os.listdir(models_dir)
            if f.startswith(f"model_{tkey}_") and f.endswith(".pkl")
        ]
        if not pkl_files:
            print(f"No model file found for {tkey} in {models_dir}. Skipping.")
            continue
        # pick the "best" by name or the newest
        pkl_files.sort(reverse=True)
        best_file = pkl_files[0]
        best_path = os.path.join(models_dir, best_file)
        try:
            model = joblib.load(best_path)
            loaded_models[tkey] = model
            print(f"Loaded model for {tkey} => {best_file}")
        except Exception as e:
            print(f"Error loading model {best_file} for {tkey}: {e}")

    if not loaded_models:
        print("No models loaded. Exiting.")
        return

    # 5) Predict for each threshold
    # We'll store probabilities in a dictionary
    results = {
        'Ticker': tickers_for_output,
        'Prob_Above1Pct': np.nan,
        'Prob_Above2Pct': np.nan,
        'Prob_Above3Pct': np.nan,
    }

    # Convert results to DataFrame at the end
    # Then fill in columns if we have a model
    for tkey in loaded_models:
        model = loaded_models[tkey]
        # predict_proba => shape (n_samples, 2)
        prob_1 = model.predict_proba(X_input)[:, 1]
        if tkey == "Above1Pct_Next":
            results['Prob_Above1Pct'] = prob_1
        elif tkey == "Above2Pct_Next":
            results['Prob_Above2Pct'] = prob_1
        elif tkey == "Above3Pct_Next":
            results['Prob_Above3Pct'] = prob_1

    pred_df = pd.DataFrame(results)

    # 6) Save final CSV => Ticker, Prob_Above1Pct, Prob_Above2Pct, Prob_Above3Pct
    predictions_dir = os.path.join(data_dir, 'predictions')
    os.makedirs(predictions_dir, exist_ok=True)

    out_path = os.path.join(predictions_dir, 'weekly_predictions_recent.csv')
    pred_df.to_csv(out_path, index=False)
    print(f"Predictions for most recent rows saved to {out_path}")


if __name__ == "__main__":
    main()
