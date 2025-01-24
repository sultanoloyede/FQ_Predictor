import pandas as pd

csv = 'fga.csv'
main = pd.read_csv(csv).sort_values(by='GAME_ID')

csv2 = 'adv2_database.csv'
adv2 = pd.read_csv(csv2).sort_values(by='GAME_ID')

season = main['SEASON']
adv2 = pd.concat([season, adv2], axis=1)

def rolling_avg_10(df1, df2, team, team_id):

    df1 = df1[df1['TEAM_ABBREVIATION'] == team].reset_index(drop=True).sort_values(by='GAME_ID')
    df2 = df2[df2['TEAM_ID'] == team_id].reset_index(drop=True).sort_values(by='GAME_ID')

    # Initialize new columns
    df1['TEAM_PACE'] = df2.iloc[0]['PACE']

    for i in range(len(df1)):
        if i == 0:
            continue

        if df2.loc[i, 'SEASON'] != df2.loc[i - 1, 'SEASON']:
            # Scenario 1: Value changes in col2
            df1.loc[i, 'TEAM_PACE'] = df2.loc[i, 'PACE']
            if i + 1 < len(df2):
                df1.loc[i + 1, 'TEAM_PACE'] = df2.loc[i, 'PACE']

        else:
            # Identify valid data slices in col1 where col2 remains the same
            prev_rows = df2.loc[:i - 1]
            valid_data_avg_10 = prev_rows[prev_rows['SEASON'] == df2.loc[i, 'SEASON']].iloc[-10:]['PACE']

            # avg_10: Take the average of up to 10 previous valid values
            if len(valid_data_avg_10) > 0:
                df1.loc[i, 'TEAM_PACE'] = valid_data_avg_10.mean()

    return df1

unique_values = main['TEAM_ABBREVIATION'].unique()

combined_df = pd.DataFrame()

team_abv = unique_values[0]

for team_abv in unique_values:
    team_id = main.loc[main['TEAM_ABBREVIATION'] == team_abv, 'TEAM_ID'].values[0]
    x = rolling_avg_10(main, adv2, team_abv, team_id)
    if team_abv == 'ATL':
        print(x)
    combined_df = pd.concat([combined_df, x], ignore_index=True)

csv3 = 'pace.csv'

combined_df = combined_df.sort_values(by='GAME_ID')

# Shift the columns
combined_df['PACE_OPP'] = combined_df['TEAM_PACE'].shift(-1)

# Assign values to specific rows ensuring alignment of shapes
even_indices = combined_df.index[::2]  # Indices for even rows
odd_indices = combined_df.index[1::2]  # Indices for odd rows (to assign values)

combined_df.loc[odd_indices, 'PACE_OPP'] = combined_df.loc[even_indices, 'TEAM_PACE'].values

# Remove rows with NaN caused by the shift
combined_df = combined_df.dropna().reset_index(drop=True)

print(combined_df.columns)

combined_df.to_csv(csv3, index=False)