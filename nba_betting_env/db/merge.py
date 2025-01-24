import pandas as pd

def sum_fqp(csv):
    database = pd.read_csv(csv)

    database['TOTAL_FQP'] = database['PTS_QTR1'] + database['PTS_QTR1_OPP']

    csv_new = 'merge.csv'
    database.to_csv(csv_new, index=False)
    return database


csv_filename = 'basic_database.csv'

sum_fqp(csv_filename)