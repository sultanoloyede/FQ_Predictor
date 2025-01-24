import pandas as pd
from statistics import mean

# Step 5: Define the function to compute LAST_3_H2H
def compute_last_3_h2h(group):
    """
    Computes the LAST_3_H2H for each game in the group.
    
    Parameters:
    - group (pd.DataFrame): A DataFrame group representing a unique matchup.
    
    Returns:
    - pd.DataFrame: The group with an added 'LAST_3_H2H' column.
    """
    # Ensure the group is sorted by GAME_ID
    group = group.sort_values('GAME_ID').reset_index(drop=True)
    
    # Extract the 'TOTAL_FQP' as a list for easy access
    total_fqp = group['TOTAL_FQP'].tolist()
    n = len(group)
    
    # Initialize a list to store the mean values
    last_3_h2h = []
    
    for i in range(n):
        if i >= 1:
            # There are prior matchups
            # Get up to the last three prior 'TOTAL_FQP' values
            prior_fqps = total_fqp[max(0, i-3):i]
            mean_val = mean(prior_fqps)
        else:
            # First occurrence; get up to the next three 'TOTAL_FQP' values
            next_fqps = total_fqp[i+1:i+4]
            if next_fqps:
                mean_val = mean(next_fqps)
            else:
                # If no next games exist, use the current 'TOTAL_FQP'
                mean_val = total_fqp[i]
        last_3_h2h.append(mean_val)
    
    # Assign the computed means to the 'LAST_3_H2H' column
    group['LAST_3_H2H'] = last_3_h2h
    
    return group

def compute_h2h():
    df = pd.read_csv('last3_database.csv')

    # Step 1: Sort by GAME_ID ascending
    df_sorted = df.sort_values('GAME_ID').reset_index(drop=True)

    # Step 2: Create MATCHUP_KEY by sorting TEAM and OPPONENT abbreviations
    df_sorted['MATCHUP_KEY'] = df_sorted.apply(
        lambda row: '_vs_'.join(sorted([row['TEAM_ABBREVIATION'], row['OPPONENT_TEAM_ABBREVIATION']])),
        axis=1
    )

    # Step 3: Extract unique games based on GAME_ID and MATCHUP_KEY
    df_unique = df_sorted.drop_duplicates(subset=['GAME_ID', 'MATCHUP_KEY']).copy()

    # Step 4: Sort unique games by GAME_ID ascending
    df_unique_sorted = df_unique.sort_values('GAME_ID').reset_index(drop=True)

    # Step 6: Group by MATCHUP_KEY and apply the function
    df_unique_with_h2h = df_unique_sorted.groupby('MATCHUP_KEY').apply(compute_last_3_h2h).reset_index(drop=True)

    # Step 7: Merge 'LAST_3_H2H' back to the original sorted DataFrame
    df_unique_h2h = df_unique_with_h2h[['GAME_ID', 'MATCHUP_KEY', 'LAST_3_H2H']]

    # Merge back to the original sorted DataFrame
    df_final = df_sorted.merge(
        df_unique_h2h,
        on=['GAME_ID', 'MATCHUP_KEY'],
        how='left'
    )

    # Drop the 'MATCHUP_KEY' if not needed
    df_final = df_final.drop(columns=['MATCHUP_KEY'])

    print(df_final)