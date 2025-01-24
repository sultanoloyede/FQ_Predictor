import pandas as pd
import numpy as np


def rolling_avg_10(df, team):
    df = df[df['TEAM_ABBREVIATION'] == team].reset_index(drop=True).sort_values(by='GAME_ID')

    # Initialize new columns
    df['TEAM_LAST_10'] = df.iloc[0]['PTS_QTR1']
    df['TEAM_WORST_2'] = df.iloc[0]['PTS_QTR1']

    for i in range(len(df)):
        if i == 0:
            continue

        # Check if the value in col2 changes
        if df.loc[i, 'SEASON'] != df.loc[i - 1, 'SEASON']:
            # Scenario 1: Value changes in col2
            df.loc[i, 'TEAM_LAST_10'] = df.loc[i, 'PTS_QTR1']
            df.loc[i, 'TEAM_WORST_2'] = df.loc[i, 'PTS_QTR1']
            if i + 1 < len(df):
                df.loc[i + 1, 'TEAM_LAST_10'] = df.loc[i, 'PTS_QTR1']
                df.loc[i + 1, 'TEAM_WORST_2'] = df.loc[i, 'PTS_QTR1']
        else:
            # Identify valid data slices in col1 where col2 remains the same
            prev_rows = df.loc[:i - 1]
            valid_data_avg_10 = prev_rows[prev_rows['SEASON'] == df.loc[i, 'SEASON']].iloc[-10:]['PTS_QTR1']
            
            # avg_10: Take the average of up to 10 previous valid values
            if len(valid_data_avg_10) > 0:
                df.loc[i, 'TEAM_LAST_10'] = valid_data_avg_10.mean()
            
            # lowest_2: Use the avg_10 slice, find the lowest 2 values, and calculate their average
            if len(valid_data_avg_10) > 1:
                lowest_2_values = valid_data_avg_10.nsmallest(2)
                df.loc[i, 'TEAM_WORST_2'] = lowest_2_values.mean()
            elif len(valid_data_avg_10) == 1:
                df.loc[i, 'TEAM_WORST_2'] = valid_data_avg_10.iloc[0]

    # Print the updated DataFrame
    return df




csv = 'merge.csv'
csv2 = 'last10_database.csv'
df = pd.read_csv(csv)

unique_values = df['TEAM_ABBREVIATION'].unique()

combined_df = pd.DataFrame()

for team in unique_values:
    x = rolling_avg_10(df, team)
    combined_df = pd.concat([combined_df, x], ignore_index=True)

combined_df = combined_df.sort_values(by='GAME_ID')

combined_df.to_csv(csv2, index=False)

# Shift the columns
combined_df['LAST_10_OPP'] = combined_df['TEAM_LAST_10'].shift(-1)
combined_df['WORST_2_OPP'] = combined_df['TEAM_WORST_2'].shift(-1)

# Assign values to specific rows ensuring alignment of shapes
even_indices = combined_df.index[::2]  # Indices for even rows
odd_indices = combined_df.index[1::2]  # Indices for odd rows (to assign values)

combined_df.loc[odd_indices, 'LAST_10_OPP'] = combined_df.loc[even_indices, 'TEAM_LAST_10'].values
combined_df.loc[odd_indices, 'WORST_2_OPP'] = combined_df.loc[even_indices, 'TEAM_WORST_2'].values

# Remove rows with NaN caused by the shift
combined_df = combined_df.dropna().reset_index(drop=True)

combined_df.drop(columns=['Total_FQP', 'index'], inplace=True)

csv3 = 'last10_tmp_database.csv'
combined_df.to_csv(csv3, index=False)